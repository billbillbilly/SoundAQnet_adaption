"""
soundaqnet.emosoundscape
========================
Valence/arousal inference for models trained on the Emo-Soundscapes dataset.

The reference papers in ``docs/emosoundscape`` use the Russell circumplex:
valence corresponds to pleasantness and arousal corresponds to eventfulness.
The 2018 deep-learning paper maps Emo-Soundscapes rankings to continuous
ratings from ``1.0`` to ``-1.0`` and trains independent regressors.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from soundaqnet.framework.emosoundscape_models import EmoSoundscapeCNN

EMOSOUNDSCAPE_TARGETS: tuple[str, str] = ("valence", "arousal")
DEFAULT_EMOSOUNDSCAPE_ARCH = "cnn_54x30"
DEFAULT_EMOSOUNDSCAPE_MODEL = "EmoS_gradient_boosting"
EMOSOUNDSCAPE_MODEL_ALIASES: dict[str, str] = {
    "EmoSoundscape_native_audio_gradient_boosting": DEFAULT_EMOSOUNDSCAPE_MODEL,
}
USER_FACING_EMOSOUNDSCAPE_MODELS: tuple[str, ...] = (DEFAULT_EMOSOUNDSCAPE_MODEL,)
_SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".opus")


def _available_bundled_models() -> list[str]:
    models_dir = Path(str(_pkg_files("soundaqnet") / "models"))
    if not models_dir.exists():
        return []
    bundled = sorted(
        p.stem
        for p in models_dir.iterdir()
        if p.is_file()
        and p.suffix in {".pth", ".pt", ".pkl", ".pickle"}
        and (p.stem.startswith("EmoSoundscape") or p.stem.startswith("EmoS"))
    )
    return [name for name in bundled if name in USER_FACING_EMOSOUNDSCAPE_MODELS]


def resolve_emosoundscape_model_path(model_arg: str | Path) -> str:
    """Resolve an Emo-Soundscape checkpoint path or bundled model name."""
    p = Path(model_arg)
    stem = p.stem
    if stem in EMOSOUNDSCAPE_MODEL_ALIASES and not p.exists():
        stem = EMOSOUNDSCAPE_MODEL_ALIASES[stem]

    if p.suffix == "" and not p.exists():
        for suffix in (".pth", ".pt", ".pkl", ".pickle"):
            candidate = p.with_suffix(suffix)
            if candidate.exists():
                return str(candidate.resolve())
    if p.exists():
        return str(p.resolve())

    for suffix in (".pth", ".pt", ".pkl", ".pickle"):
        bundled = Path(str(_pkg_files("soundaqnet") / "models" / f"{stem}{suffix}"))
        if bundled.exists():
            return str(bundled)

    bundled_names = _available_bundled_models()
    extra = ""
    if bundled_names:
        extra = "\nBundled Emo-Soundscape models:\n" + "\n".join(f"  {m}" for m in bundled_names)
    raise FileNotFoundError(
        f"Emo-Soundscape model not found: {model_arg!r}.\n"
        "Supply a path to a .pth/.pt/.pkl checkpoint or add a bundled "
        "EmoS*.pkl or EmoSoundscape*.pth model under soundaqnet/models/."
        f"{extra}"
    )


def _torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a PyTorch state_dict")


def _normalise_features(
    features: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    require_patch: bool = True,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if require_patch and x.shape[-2:] != (54, 30):
        raise ValueError(f"Expected feature shape (..., 54, 30), got {x.shape}")
    if mean is not None and std is not None:
        x = (x - mean) / np.maximum(std, 1e-8)
    return x.astype(np.float32)


class EmoSoundscape:
    """Predict valence and arousal from Emo-Soundscapes-style features.

    Parameters
    ----------
    model:
        Path to a PyTorch checkpoint, TorchScript module, or pickle containing
        sklearn estimators. PyTorch checkpoints use 54 x 30 feature patches.
        Pickle checkpoints may use flat tabular features. The bundled default
        model uses package-native audio features extracted by
        :mod:`soundaqnet.feature_extraction.emosoundscape_features`.
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"``, or any ``torch.device``. Auto-selects
        CUDA, then MPS, then CPU when omitted.
    arch:
        Currently ``"cnn_54x30"``, matching the CNN trained from scratch in Fan
        et al. (2018). TorchScript checkpoints ignore this value.
    """

    def __init__(
        self,
        model: str | Path = DEFAULT_EMOSOUNDSCAPE_MODEL,
        device: str | torch.device | None = None,
        arch: str = DEFAULT_EMOSOUNDSCAPE_ARCH,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model_path = resolve_emosoundscape_model_path(model)
        self.arch = arch
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self._sklearn_model: Any | None = None
        self._torch_model: torch.nn.Module | None = None
        self._target_models: dict[str, torch.nn.Module] | None = None

        suffix = Path(self.model_path).suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            with open(self.model_path, "rb") as fh:
                self._sklearn_model = pickle.load(fh)
            return

        self._load_torch_model()

    @classmethod
    def load(
        cls,
        model: str | Path = DEFAULT_EMOSOUNDSCAPE_MODEL,
        device: str | torch.device | None = None,
        arch: str = DEFAULT_EMOSOUNDSCAPE_ARCH,
    ) -> "EmoSoundscape":
        return cls(model=model, device=device, arch=arch)

    def _new_arch_model(self, out_dim: int = 1) -> torch.nn.Module:
        if self.arch != "cnn_54x30":
            raise ValueError(f"Unsupported Emo-Soundscape architecture: {self.arch!r}")
        return EmoSoundscapeCNN(out_dim=out_dim)

    def _load_torch_model(self) -> None:
        if Path(self.model_path).suffix.lower() == ".pt":
            try:
                model = torch.jit.load(self.model_path, map_location="cpu")
            except RuntimeError:
                model = None
            if model is not None:
                self._torch_model = model.to(self.device).eval()
                return

        checkpoint = _torch_load(self.model_path)
        if isinstance(checkpoint, dict):
            if "feature_mean" in checkpoint and "feature_std" in checkpoint:
                self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
                self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

            if "valence" in checkpoint and "arousal" in checkpoint:
                self._target_models = {}
                for target in EMOSOUNDSCAPE_TARGETS:
                    model = self._new_arch_model(out_dim=1)
                    state = checkpoint[target]
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    model.load_state_dict(state)
                    self._target_models[target] = model.to(self.device).eval()
                return

            out_dim = int(checkpoint.get("out_dim", 2)) if isinstance(checkpoint, dict) else 2
        else:
            out_dim = 2

        model = self._new_arch_model(out_dim=out_dim)
        model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
        self._torch_model = model.to(self.device).eval()

    def _predict_numpy_batch(self, batch: np.ndarray) -> np.ndarray:
        if self._sklearn_model is not None:
            flat = batch.reshape(batch.shape[0], -1)
            pred = self._sklearn_model.predict(flat)
            pred = np.asarray(pred, dtype=np.float32)
            if pred.ndim == 1:
                raise ValueError("A pickle model must return both valence and arousal columns")
            return pred[:, :2]

        x = torch.from_numpy(batch).float().to(self.device)
        with torch.no_grad():
            if self._target_models is not None:
                pred = torch.cat([self._target_models[t](x) for t in EMOSOUNDSCAPE_TARGETS], dim=1)
            elif self._torch_model is not None:
                pred = self._torch_model(x)
            else:
                raise RuntimeError("No Emo-Soundscape model is loaded")
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        if pred_np.ndim == 1:
            pred_np = pred_np[:, None]
        if pred_np.shape[1] == 1:
            raise ValueError("Single-output checkpoints must be provided as valence/arousal pair")
        return pred_np[:, :2]

    @staticmethod
    def _records(names: list[str], pred: np.ndarray) -> list[dict[str, float | str]]:
        return [
            {"clip_id": name, "valence": float(row[0]), "arousal": float(row[1])}
            for name, row in zip(names, pred)
        ]

    def predict_sample(
        self, features: np.ndarray, clip_id: str = "sample"
    ) -> dict[str, float | str]:
        """Predict one feature vector or ``(54, 30)`` feature patch."""
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            batch = x[None, :]
        else:
            batch = x[None]
        batch = _normalise_features(
            batch,
            self.feature_mean,
            self.feature_std,
            require_patch=self._sklearn_model is None,
        )
        return self._records([clip_id], self._predict_numpy_batch(batch))[0]

    def predict(
        self,
        feature_dir: str | Path,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Predict valence/arousal for a directory of ``.npy`` feature arrays."""
        feature_dir = Path(feature_dir)
        files = sorted(feature_dir.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy feature files found in {feature_dir}")

        records: list[dict[str, float | str]] = []
        batch: list[np.ndarray] = []
        names: list[str] = []
        iterator = tqdm(files, desc="Emo-Soundscape inference", disable=not show_progress)
        for file in iterator:
            batch.append(np.load(str(file)).astype(np.float32))
            names.append(file.stem)
            if len(batch) >= batch_size:
                arr = _normalise_features(
                    np.stack(batch),
                    self.feature_mean,
                    self.feature_std,
                    require_patch=self._sklearn_model is None,
                )
                records.extend(self._records(names, self._predict_numpy_batch(arr)))
                batch, names = [], []

        if batch:
            arr = _normalise_features(
                np.stack(batch),
                self.feature_mean,
                self.feature_std,
                require_patch=self._sklearn_model is None,
            )
            records.extend(self._records(names, self._predict_numpy_batch(arr)))
        return pd.DataFrame(records)

    def predict_from_audio(
        self,
        audio_dir: str | Path | None = None,
        audio_files: list[str | Path] | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Extract package-native features from audio, then predict valence/arousal.

        This method is intended for the bundled ``EmoS_gradient_boosting``
        model and compatible custom models trained on the package-native
        122-feature audio representation.
        """
        from soundaqnet.feature_extraction.emosoundscape_features import (
            extract_emosoundscape_features_from_file,
        )

        if audio_files is None:
            if audio_dir is None:
                raise ValueError("Specify either audio_dir or audio_files.")
            files = sorted(
                p
                for p in Path(audio_dir).rglob("*")
                if p.is_file() and p.suffix.lower() in _SUPPORTED_AUDIO_EXTS
            )
        else:
            files = [Path(p) for p in audio_files]

        if not files:
            raise ValueError(f"No supported audio files found in {audio_dir or audio_files}")

        records: list[dict[str, float | str]] = []
        batch: list[np.ndarray] = []
        names: list[str] = []
        iterator = tqdm(files, desc="Extracting Emo features", disable=not show_progress)
        for file in iterator:
            batch.append(extract_emosoundscape_features_from_file(file))
            names.append(file.stem)
            if len(batch) >= batch_size:
                arr = _normalise_features(
                    np.stack(batch),
                    self.feature_mean,
                    self.feature_std,
                    require_patch=False,
                )
                records.extend(self._records(names, self._predict_numpy_batch(arr)))
                batch, names = [], []

        if batch:
            arr = _normalise_features(
                np.stack(batch),
                self.feature_mean,
                self.feature_std,
                require_patch=False,
            )
            records.extend(self._records(names, self._predict_numpy_batch(arr)))
        return pd.DataFrame(records)

    def __repr__(self) -> str:
        arch = "sklearn_features" if self._sklearn_model is not None else self.arch
        return (
            f"EmoSoundscape(model='{Path(self.model_path).stem}', "
            f"arch='{arch}', device='{self.device}')"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Emo-Soundscape valence/arousal inference on audio or .npy features."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMOSOUNDSCAPE_MODEL,
        help=f"Path or bundled model name (default: {DEFAULT_EMOSOUNDSCAPE_MODEL}).",
    )
    parser.add_argument("--feature_dir", help="Directory of .npy feature files.")
    parser.add_argument("--audio_dir", help="Directory of audio files.")
    parser.add_argument("--audio_file", action="append", help="Audio file; can be repeated.")
    parser.add_argument("--output_csv", default="emosoundscape_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        names = _available_bundled_models()
        print("Bundled Emo-Soundscape models:")
        print("\n".join(f"  {name}" for name in names) if names else "  (none found)")
        return 0

    if args.feature_dir is None and args.audio_dir is None and args.audio_file is None:
        parser.error(
            "Specify --audio_dir, --audio_file, or --feature_dir unless --list-models is used"
        )

    try:
        model = EmoSoundscape(args.model, device=args.device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.audio_dir is not None or args.audio_file is not None:
        df = model.predict_from_audio(
            audio_dir=args.audio_dir,
            audio_files=args.audio_file,
            batch_size=args.batch_size,
        )
    else:
        df = model.predict(args.feature_dir, batch_size=args.batch_size)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

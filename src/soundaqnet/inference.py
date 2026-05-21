"""
soundaqnet.inference
====================
High-level SoundAQnet inference interface.

Public API
----------
    model = SoundAQnet()                              # load default bundled model
    model = SoundAQnet("SoundAQnet_ASC96_AEC94_PAQ1039")
    model = SoundAQnet("/path/to/custom.pth")

    # From pre-extracted feature directories (.npy files)
    df = model.predict(mel_dir="mel/", loudness_dir="loudness/")

    # From a single pair of numpy arrays
    result = model.predict_sample(mel_array, loudness_array)

    # End-to-end from an audio directory (extracts features internally)
    df = model.predict_from_audio("audio/")
    df = model.predict_from_audio(audio_files=["a.wav", "b.wav"])

All methods return a ``pandas.DataFrame`` (multi-sample) or a plain ``dict``
(single sample).  Every row / dict contains:

    clip_id      str   – file stem (no extension)
    scene        str   – predicted acoustic scene label
    isop         float – ISO Pleasantness  (-1 … +1)
    isoe         float – ISO Eventfulness  (-1 … +1)
    pleasant     float – PAQ pleasant       score
    eventful     float – PAQ eventful       score
    chaotic      float – PAQ chaotic        score
    vibrant      float – PAQ vibrant        score
    uneventful   float – PAQ uneventful     score
    calm         float – PAQ calm           score
    annoying     float – PAQ annoying       score
    monotonous   float – PAQ monotonous     score
    top_events   list  – top-5 event labels sorted by probability (high→low)
    event_probs  dict  – {event_label: probability} for all 15 event classes

CLI
---
    soundaqnet-infer \\
        --dataset_mel  <mel_npy_dir>        \\
        --dataset_wav_loudness <loud_dir>   \\
        [--model <name_or_path>]            \\
        [--list-models]

Bundled model names (pass to --model or SoundAQnet()):
    SoundAQnet_ASC96_AEC94_PAQ1027   (default)
    SoundAQnet_ASC96_AEC94_PAQ1039
    SoundAQnet_ASC96_AEC94_PAQ1041
    SoundAQnet_ASC96_AEC95_PAQ1052
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
import tempfile
from pathlib import Path
from importlib.resources import files as _pkg_files
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Internal PyTorch model (renamed to avoid clash with the public SoundAQnet class)
from soundaqnet.framework.models_pytorch import SoundAQnet as _SoundAQnetModel
from soundaqnet.framework import config
from soundaqnet.framework.utilities import create_folder, calculate_scalar, scale


# ── constants ─────────────────────────────────────────────────────────────────

#: Labels for the 15 audio-event output nodes.
EVENT_LABELS: list[str] = [
    "Silence", "Human sounds", "Wind", "Water", "Natural sounds", "Traffic",
    "Sounds of things", "Vehicle", "Bird", "Outside, rural or natural",
    "Environment and background", "Speech", "Music", "Noise", "Animal",
]

#: Short names that can be passed to SoundAQnet() instead of a full path.
BUNDLED_MODELS: list[str] = [
    "SoundAQnet_ASC96_AEC94_PAQ1027",
    "SoundAQnet_ASC96_AEC94_PAQ1039",
    "SoundAQnet_ASC96_AEC94_PAQ1041",
    "SoundAQnet_ASC96_AEC95_PAQ1052",
]

DEFAULT_MODEL: str = BUNDLED_MODELS[0]

# Audio formats accepted by predict_from_audio
_SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".opus")


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_model_path(model_arg: str) -> str:
    """Return an absolute path to a .pth file.

    Accepts either:
    * A filesystem path (absolute or relative) to a .pth file, or
    * A short model name (without extension) from BUNDLED_MODELS.

    Raises ``FileNotFoundError`` if neither resolves.
    """
    p = Path(model_arg)
    if p.suffix == "" and not p.exists():
        p = p.with_suffix(".pth")
    if p.exists():
        return str(p.resolve())

    stem = Path(model_arg).stem
    if stem in BUNDLED_MODELS:
        bundled = Path(str(_pkg_files("soundaqnet") / "models" / f"{stem}.pth"))
        if bundled.exists():
            return str(bundled)
        raise FileNotFoundError(
            f"Bundled model '{stem}.pth' not found inside the installed package.\n"
            "Re-install soundaqnet to restore it, or supply a full path."
        )

    raise FileNotFoundError(
        f"Model not found: '{model_arg}'.\n"
        f"Supply a path to a .pth file, or use one of the bundled names:\n"
        + "\n".join(f"  {m}" for m in BUNDLED_MODELS)
    )


def _load_norm_stats() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load bundled normalisation statistics.

    Returns (mean_mel, std_mel, mean_loudness, std_loudness).
    """
    data_pkg = _pkg_files("soundaqnet.data")
    with open(str(data_pkg / "norm_log_mel.pickle"), "rb") as fh:
        mel_data = pickle.load(fh, encoding="bytes")
    with open(str(data_pkg / "norm_loudness.pickle"), "rb") as fh:
        loud_data = pickle.load(fh, encoding="bytes")
    mean_mel,   std_mel   = calculate_scalar(mel_data)
    mean_loud,  std_loud  = calculate_scalar(loud_data)
    return mean_mel, std_mel, mean_loud, std_loud


# ── SoundAQnet high-level class ───────────────────────────────────────────────

class SoundAQnet:
    """High-level SoundAQnet soundscape inference interface.

    Parameters
    ----------
    model  : bundled model name (str) **or** path to a ``.pth`` checkpoint.
             Defaults to ``"SoundAQnet_ASC96_AEC94_PAQ1027"``.
    device : ``"cpu"``, ``"cuda"``, ``"mps"``, or any ``torch.device``.
             Auto-selects CUDA → MPS → CPU if not specified.

    Examples
    --------
    >>> from soundaqnet import SoundAQnet
    >>> model = SoundAQnet()                    # default bundled weights
    >>> model = SoundAQnet("SoundAQnet_ASC96_AEC95_PAQ1052")
    >>> model = SoundAQnet("/my/trained/model.pth")

    Predict from pre-extracted feature directories::

        df = model.predict(mel_dir="mel/", loudness_dir="loudness/")

    Predict a single sample from numpy arrays::

        result = model.predict_sample(mel_array, loudness_array)

    End-to-end from audio files (feature extraction + inference)::

        df = model.predict_from_audio("audio/")
        df = model.predict_from_audio(audio_files=["clip1.wav", "clip2.wav"])
    """

    # Model architecture hyper-parameters (fixed for all pretrained checkpoints)
    _NODE_EMB_DIM = 64
    _HIDDEN_DIM   = 32
    _OUT_DIM      = 64
    _NUM_NODES    = 8

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        device: str | torch.device | None = None,
    ) -> None:
        # ── device ───────────────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)

        # ── load PyTorch model ────────────────────────────────────────────────
        model_path = resolve_model_path(str(model))
        self._net = _SoundAQnetModel(
            max_node_num=self._NUM_NODES,
            node_emb_dim=self._NODE_EMB_DIM,
            hidden_dim=self._HIDDEN_DIM,
            out_dim=self._OUT_DIM,
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self._net.load_state_dict(checkpoint["state_dict"])
        else:
            self._net.load_state_dict(checkpoint)
        self._net.to(self.device)
        self._net.eval()

        # ── normalisation statistics ──────────────────────────────────────────
        self._mean_mel, self._std_mel, self._mean_loud, self._std_loud = \
            _load_norm_stats()

        self.model_path = model_path

    # ── convenience constructor ───────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model: str | Path = DEFAULT_MODEL,
        device: str | torch.device | None = None,
    ) -> "SoundAQnet":
        """Convenience alternative to ``SoundAQnet(model, device)``.

        Examples
        --------
        >>> model = SoundAQnet.load("SoundAQnet_ASC96_AEC94_PAQ1039")
        """
        return cls(model=model, device=device)

    # ── normalisation helpers ─────────────────────────────────────────────────

    def _norm_mel(self, x: np.ndarray) -> np.ndarray:
        return scale(x, self._mean_mel, self._std_mel).astype(np.float32)

    def _norm_loud(self, x: np.ndarray) -> np.ndarray:
        return scale(x, self._mean_loud, self._std_loud).astype(np.float32)

    # ── core forward pass ─────────────────────────────────────────────────────

    def _forward_batch(
        self,
        mel_batch:  np.ndarray,   # (B, T, 64)
        loud_batch: np.ndarray,   # (B, T, 1)
    ) -> dict[str, np.ndarray]:
        """Run the model on one numpy batch; return raw numpy outputs."""
        x      = torch.from_numpy(mel_batch).float().to(self.device)
        x_loud = torch.from_numpy(loud_batch).float().to(self.device)

        with torch.no_grad():
            scene, event, ISOPls, ISOEvs, \
            pleasant, eventful, chaotic, vibrant, \
            uneventful, calm, annoying, monotonous = self._net(x, x_loud)

            event = F.sigmoid(event)

        def _np(t: torch.Tensor) -> np.ndarray:
            return t.cpu().numpy()

        return {
            "scene":      _np(scene),       # (B, 3)  logits → softmax later
            "event":      _np(event),        # (B, 15) probabilities
            "ISOPls":     _np(ISOPls),       # (B, 1)
            "ISOEvs":     _np(ISOEvs),       # (B, 1)
            "pleasant":   _np(pleasant),
            "eventful":   _np(eventful),
            "chaotic":    _np(chaotic),
            "vibrant":    _np(vibrant),
            "uneventful": _np(uneventful),
            "calm":       _np(calm),
            "annoying":   _np(annoying),
            "monotonous": _np(monotonous),
        }

    # ── record builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_record(clip_id: str, raw: dict[str, np.ndarray], idx: int) -> dict:
        """Convert model output arrays at position *idx* into a result dict."""
        scene_idx = int(np.argmax(raw["scene"][idx]))
        event_probs = {
            label: float(raw["event"][idx, i])
            for i, label in enumerate(EVENT_LABELS)
        }
        top_events = sorted(event_probs, key=event_probs.get, reverse=True)[:5]

        return {
            "clip_id":     clip_id,
            "scene":       config.scene_labels[scene_idx],
            "isop":        float(raw["ISOPls"][idx, 0]),
            "isoe":        float(raw["ISOEvs"][idx, 0]),
            "pleasant":    float(raw["pleasant"][idx, 0]),
            "eventful":    float(raw["eventful"][idx, 0]),
            "chaotic":     float(raw["chaotic"][idx, 0]),
            "vibrant":     float(raw["vibrant"][idx, 0]),
            "uneventful":  float(raw["uneventful"][idx, 0]),
            "calm":        float(raw["calm"][idx, 0]),
            "annoying":    float(raw["annoying"][idx, 0]),
            "monotonous":  float(raw["monotonous"][idx, 0]),
            "top_events":  top_events,
            "event_probs": event_probs,
        }

    # ── batch accumulator helper ──────────────────────────────────────────────

    def _flush_batch(
        self,
        mel_list:   list[np.ndarray],
        loud_list:  list[np.ndarray],
        names:      list[str],
    ) -> list[dict]:
        mel_arr  = np.stack(mel_list,  axis=0)
        loud_arr = np.stack(loud_list, axis=0)
        raw = self._forward_batch(mel_arr, loud_arr)
        return [self._build_record(names[i], raw, i) for i in range(len(names))]

    # ── public inference methods ──────────────────────────────────────────────

    def predict_sample(
        self,
        mel:      np.ndarray,
        loudness: np.ndarray,
    ) -> dict:
        """Run inference on a **single** pre-extracted feature sample.

        Parameters
        ----------
        mel      : log-mel spectrogram ``(T, 64)`` as returned by
                   :func:`~soundaqnet.feature_extraction.extract_mel_from_file`.
        loudness : ISO 532-1 loudness ``(T, 1)`` as returned by
                   :func:`~soundaqnet.feature_extraction.extract_loudness_from_file`.

        Returns
        -------
        dict with keys: ``clip_id``, ``scene``, ``isop``, ``isoe``,
        ``pleasant``, ``eventful``, ``chaotic``, ``vibrant``,
        ``uneventful``, ``calm``, ``annoying``, ``monotonous``,
        ``top_events``, ``event_probs``.

        Examples
        --------
        >>> from soundaqnet.feature_extraction import (
        ...     extract_mel_from_file, extract_loudness_from_file)
        >>> mel  = extract_mel_from_file("my_clip.wav")
        >>> loud = extract_loudness_from_file("my_clip.wav")
        >>> result = model.predict_sample(mel, loud)
        >>> result["scene"]
        'park'
        >>> result["isop"]
        0.62
        >>> result["top_events"]
        ['Bird', 'Wind', 'Natural sounds', 'Silence', 'Water']
        """
        mel_n  = self._norm_mel(mel[None])    # (1, T, 64)
        loud_n = self._norm_loud(loudness[None])  # (1, T, 1)
        raw    = self._forward_batch(mel_n, loud_n)
        return self._build_record("sample", raw, 0)

    def predict(
        self,
        mel_dir:      str | Path,
        loudness_dir: str | Path,
        batch_size:   int = 32,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Run inference on directories of pre-extracted ``.npy`` feature files.

        Parameters
        ----------
        mel_dir        : directory produced by
                         ``soundaqnet-extract-mel`` / :func:`extract_mel`.
        loudness_dir   : directory produced by
                         ``soundaqnet-extract-loudness`` / :func:`extract_loudness`.
        batch_size     : number of clips forwarded in one model call.
                         Larger values are faster (up to GPU memory limits).
                         Default 32.  Try 64–128 for a GPU, 8–16 for CPU.
        show_progress  : show a ``tqdm`` progress bar (default ``True``).

        Returns
        -------
        pandas.DataFrame  — one row per clip, columns as described in the
        class docstring.  ``event_probs`` column contains dicts.

        Raises
        ------
        ValueError   if no ``.npy`` files are found in *mel_dir*.
        FileNotFoundError  if a loudness file is missing for a mel file.

        Examples
        --------
        >>> df = model.predict(mel_dir="mel/", loudness_dir="loudness/")
        >>> # Larger batch for GPU:
        >>> df = model.predict(mel_dir="mel/", loudness_dir="loudness/",
        ...                    batch_size=64)
        >>> df.columns.tolist()
        ['clip_id', 'scene', 'isop', 'isoe', 'pleasant', ..., 'event_probs']
        """
        mel_dir      = Path(mel_dir)
        loudness_dir = Path(loudness_dir)

        mel_files = sorted(mel_dir.glob("*.npy"))
        if not mel_files:
            raise ValueError(f"No .npy files found in {mel_dir}")

        records:   list[dict]       = []
        mel_list:  list[np.ndarray] = []
        loud_list: list[np.ndarray] = []
        names:     list[str]        = []

        with tqdm(total=len(mel_files), desc="Inference", disable=not show_progress) as pbar:
            for mel_file in mel_files:
                loud_file = loudness_dir / mel_file.name
                if not loud_file.exists():
                    print(f"[warn] Missing loudness file for {mel_file.name} — skipped.")
                    pbar.update(1)
                    continue

                mel_n  = self._norm_mel(np.load(str(mel_file)).astype(np.float32))
                loud_n = self._norm_loud(np.load(str(loud_file)).astype(np.float32))

                mel_list.append(mel_n)
                loud_list.append(loud_n)
                names.append(mel_file.stem)

                if len(mel_list) == batch_size:
                    records += self._flush_batch(mel_list, loud_list, names)
                    mel_list, loud_list, names = [], [], []

                pbar.update(1)

            if mel_list:
                records += self._flush_batch(mel_list, loud_list, names)

        return pd.DataFrame(records)

    def predict_from_audio(
        self,
        audio_dir:     str | Path | None  = None,
        audio_files:   list[str | Path] | None = None,
        batch_size:    int = 32,
        num_workers:   int = 4,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """End-to-end inference: **extract features then predict**.

        Runs mel and loudness extraction on each file then immediately runs
        the model in batches.  At least one of *audio_dir* or *audio_files*
        must be provided.

        Parameters
        ----------
        audio_dir      : directory of audio files.  All supported formats
                         (.wav, .mp3, .flac, .ogg, .aiff, .m4a, .opus) are used.
        audio_files    : explicit list of audio file paths.
        batch_size     : number of clips forwarded in one model call.
                         Larger values are faster (up to GPU memory limits).
                         Default 32.  Try 64–128 for a GPU, 8–16 for CPU.
        num_workers    : parallel worker threads for feature extraction
                         (mel + loudness).  Default 4.  Pass ``1`` for
                         sequential extraction.
        show_progress  : show a ``tqdm`` progress bar for extraction and
                         inference (default ``True``).

        Returns
        -------
        pandas.DataFrame — same columns as :meth:`predict`.

        Examples
        --------
        >>> df = model.predict_from_audio("audio/")
        >>> # Fast GPU run with more workers and larger batch:
        >>> df = model.predict_from_audio("audio/", batch_size=64, num_workers=8)
        >>> df = model.predict_from_audio(audio_files=["a.wav", "b.mp3"])
        """
        from soundaqnet.feature_extraction.mel_spectrogram import extract_mel_from_file
        from soundaqnet.feature_extraction.loudness import extract_loudness_from_file

        if audio_dir is not None:
            files = sorted(
                p for p in Path(audio_dir).iterdir()
                if p.suffix.lower() in _SUPPORTED_EXTS
            )
        elif audio_files is not None:
            files = [Path(f) for f in audio_files]
        else:
            raise ValueError("Specify either audio_dir or audio_files.")

        if not files:
            raise ValueError(f"No supported audio files found in {audio_dir or audio_files}")

        records:   list[dict]       = []
        mel_list:  list[np.ndarray] = []
        loud_list: list[np.ndarray] = []
        names:     list[str]        = []

        print(f"Extracting features for {len(files)} file(s) "
              f"(num_workers={num_workers}, batch_size={batch_size})…")

        with tqdm(total=len(files), desc="Extracting + inference",
                  disable=not show_progress) as pbar:
            for audio_file in files:
                mel  = extract_mel_from_file(audio_file, num_workers=1)
                loud = extract_loudness_from_file(audio_file, num_workers=1)

                mel_list.append(self._norm_mel(mel))
                loud_list.append(self._norm_loud(loud))
                names.append(audio_file.stem)

                if len(mel_list) == batch_size:
                    records += self._flush_batch(mel_list, loud_list, names)
                    mel_list, loud_list, names = [], [], []

                pbar.update(1)

            if mel_list:
                records += self._flush_batch(mel_list, loud_list, names)

        return pd.DataFrame(records)

    def __repr__(self) -> str:
        return (
            f"SoundAQnet("
            f"model='{Path(self.model_path).stem}', "
            f"device='{self.device}')"
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> int:
    """CLI: soundaqnet-infer"""
    parser = argparse.ArgumentParser(
        description="Run SoundAQnet inference on pre-extracted features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(os.getcwd(), "Dataset"),
        help="(Legacy) Path to Dataset directory for normalization files. "
             "Bundled stats are used automatically; this flag is ignored.",
    )
    parser.add_argument(
        "--dataset_mel",
        type=str,
        required=True,
        help="Directory of mel .npy feature files.",
    )
    parser.add_argument(
        "--dataset_wav_loudness",
        type=str,
        required=True,
        help="Directory of loudness .npy feature files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            f"Bundled model name (default: {DEFAULT_MODEL!r}) or path to a "
            ".pth checkpoint.  Run --list-models to see all bundled options."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available bundled model names and exit.",
    )
    parser.add_argument(
        "--event_output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "SoundAQnet_event_probability"),
        help="Directory to save per-clip event probability .txt files.",
    )
    parser.add_argument(
        "--paq_output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs"),
        help="Directory to save per-clip scene / ISO / PAQ .txt files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. 'cpu', 'cuda', 'mps'.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Bundled SoundAQnet models:")
        for name in BUNDLED_MODELS:
            print(f"  {name}")
        return 0

    try:
        model = SoundAQnet(model=args.model, device=args.device)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded: {model}")

    df = model.predict(
        mel_dir=args.dataset_mel,
        loudness_dir=args.dataset_wav_loudness,
    )

    # ── Write legacy text-file outputs (backward compatible) ──────────────────
    create_folder(args.event_output_dir)
    create_folder(args.paq_output_dir)

    for _, row in df.iterrows():
        name = row["clip_id"]
        print(f"\nSoundscape audio clip: {name}")

        # Event probabilities
        event_vals = list(row["event_probs"].values())
        txt_event = os.path.join(args.event_output_dir, f"{name}_event.txt")
        np.savetxt(txt_event, event_vals)
        print("Audio event probabilities:", event_vals)

        # Scene / ISO / PAQ
        txt_paq = os.path.join(args.paq_output_dir, f"{name}_scene_PAQ.txt")
        with open(txt_paq, "w") as fh:
            fh.write(row["scene"] + "\n")
            fh.write(f"{row['isop']}\t{row['isoe']}\n")
            paq_vals = [
                row["pleasant"], row["eventful"], row["chaotic"], row["vibrant"],
                row["uneventful"], row["calm"], row["annoying"], row["monotonous"],
            ]
            fh.write("\t".join(str(v) for v in paq_vals) + "\n")
        print(f"Scene: {row['scene']}")
        print(f"ISOP / ISOE: {row['isop']:.4f} / {row['isoe']:.4f}")
        print(f"Top events: {', '.join(row['top_events'])}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)

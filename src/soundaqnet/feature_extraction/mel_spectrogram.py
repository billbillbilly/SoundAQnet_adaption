"""
soundaqnet.feature_extraction.mel_spectrogram
==============================================
Log-mel spectrogram extraction from mono audio clips.

Produces one ``.npy`` file per audio file, shape (frames, 64) at 16 kHz.

Public API
----------
    extract_mel(input_dir, output_dir, num_workers=4)   — batch: directory → .npy files
    extract_mel_from_file(audio_path, num_workers=4)    — single/multi file → numpy array(s)

CLI
---
    soundaqnet-extract-mel --input_dir <wav_dir> --output_dir <out_dir> [--num_workers N]

Parallelism
-----------
    Both batch and multi-file APIs use ``ThreadPoolExecutor`` so that disk I/O
    (librosa.load) from multiple files overlaps with STFT/mel computation.
    Each worker thread keeps its own extractor instance (via ``threading.local``)
    so there is never concurrent mutation of shared module state.  On CUDA the
    threads share the same device; on CPU they can run torch ops independently.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import torch
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from tqdm import tqdm

# Audio formats supported as input — librosa handles all of these.
SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".opus")


# ── helpers ───────────────────────────────────────────────────────────────────


def create_folder(fd: str) -> None:
    if not os.path.exists(fd):
        os.makedirs(fd, exist_ok=True)


def listFnames(dirName: str) -> list[str]:
    """Return all supported audio files under *dirName* (recursive)."""
    fnames = []
    for rootDir, _, filesList in os.walk(dirName):
        fnames += [
            os.path.join(rootDir, f) for f in filesList if Path(f).suffix.lower() in SUPPORTED_EXTS
        ]
    return fnames


def pad_or_truncate(x: np.ndarray, audio_length: int) -> np.ndarray:
    """Pad all audio to a specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0:audio_length]


def move_data_to_device(x: np.ndarray, device: torch.device) -> torch.Tensor:
    if "float" in str(x.dtype):
        x_t = torch.Tensor(x)
    elif "int" in str(x.dtype):
        x_t = torch.LongTensor(x)
    else:
        return x  # type: ignore[return-value]
    return x_t.to(device)


# ── Per-thread extractor cache ────────────────────────────────────────────────
# Each thread keeps its own Spectrogram + LogmelFilterBank so that parallel
# workers never mutate shared module state.  Extractors are cheap to create
# (they are fixed linear filters) and reused across all files in a thread.

_MEL_BINS = 64
_SAMPLE_RATE = 16_000
_WINDOW_SIZE = 512
_HOP_SIZE = 160

_thread_local_mel = threading.local()


def _get_thread_extractors(device: torch.device) -> tuple:
    """Return the (spec_ext, mel_ext) pair bound to the calling thread.

    Creates a new pair on first call per thread, or when *device* changes.
    """
    if getattr(_thread_local_mel, "device", None) != device:
        _thread_local_mel.spec_ext = Spectrogram(
            n_fft=_WINDOW_SIZE,
            hop_length=_HOP_SIZE,
            win_length=_WINDOW_SIZE,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        ).to(device)
        _thread_local_mel.mel_ext = LogmelFilterBank(
            sr=_SAMPLE_RATE,
            n_fft=_WINDOW_SIZE,
            n_mels=_MEL_BINS,
            fmin=50,
            fmax=_SAMPLE_RATE // 2,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        ).to(device)
        _thread_local_mel.device = device
    return _thread_local_mel.spec_ext, _thread_local_mel.mel_ext


# ── Module-level cache for single-call API (backward compatible) ──────────────
# Used by extract_mel_from_file when called from the main thread or when
# num_workers == 1.  Protected with a lock so it remains safe if the caller
# inadvertently shares the module across threads.

_MAIN_LOCK = threading.Lock()
_spectrogram_extractor = None
_logmel_extractor = None
_extractor_device = None


def _get_extractors(device: "torch.device | None" = None) -> tuple:
    """Return (spectrogram_extractor, logmel_extractor) on *device*.

    Extractors are cached at module level and re-created only if the device
    changes (e.g. the caller switches from CPU to GPU between calls).
    Thread-safe via ``_MAIN_LOCK``.
    """
    global _spectrogram_extractor, _logmel_extractor, _extractor_device

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with _MAIN_LOCK:
        if _spectrogram_extractor is None or device != _extractor_device:
            _spectrogram_extractor = Spectrogram(
                n_fft=_WINDOW_SIZE,
                hop_length=_HOP_SIZE,
                win_length=_WINDOW_SIZE,
                window="hann",
                center=True,
                pad_mode="reflect",
                freeze_parameters=True,
            ).to(device)
            _logmel_extractor = LogmelFilterBank(
                sr=_SAMPLE_RATE,
                n_fft=_WINDOW_SIZE,
                n_mels=_MEL_BINS,
                fmin=50,
                fmax=_SAMPLE_RATE // 2,
                ref=1.0,
                amin=1e-10,
                top_db=None,
                freeze_parameters=True,
            ).to(device)
            _extractor_device = device

    return _spectrogram_extractor, _logmel_extractor


# ── extraction ────────────────────────────────────────────────────────────────


def _extract_one(
    audio_path: str,
    output_dir: "str | None",
    device: torch.device,
) -> np.ndarray:
    """Extract log-mel spectrogram for a single file.  Thread-safe."""
    spec_ext, mel_ext = _get_thread_extractors(device)

    audiodata, _ = librosa.load(str(audio_path), sr=_SAMPLE_RATE, mono=True)
    x = torch.from_numpy(audiodata).float().unsqueeze(0).to(device)

    with torch.no_grad():
        spectrogram = spec_ext(x)
        logmel = mel_ext(spectrogram)  # (1, 1, frames, mel_bins)

    arr = logmel[0, 0].cpu().numpy().astype(np.float32)

    if output_dir is not None:
        np.save(os.path.join(output_dir, Path(audio_path).stem + ".npy"), arr)

    return arr


def run_jobs(input_dir: str, output_dir: str, num_workers: int = 4) -> None:
    """Extract log-mel spectrograms for every audio file in *input_dir*.

    Parameters
    ----------
    input_dir   : directory that is searched recursively for audio files.
    output_dir  : directory where ``.npy`` outputs are written.
    num_workers : number of parallel worker threads.  Disk I/O and librosa
                  loading run in parallel across workers; CUDA/CPU forward
                  passes are interleaved per thread.  Default 4.
    """
    create_folder(output_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    audio_files = listFnames(input_dir)
    print(f"Found {len(audio_files)} audio files ({', '.join(SUPPORTED_EXTS)})")

    # Skip already-completed outputs
    audio_files = [
        f
        for f in audio_files
        if not os.path.exists(os.path.join(output_dir, Path(f).stem + ".npy"))
    ]
    print(f"Skipping already-complete files; {len(audio_files)} remaining")

    if not audio_files:
        return

    if num_workers <= 1:
        for audio_path in tqdm(audio_files, desc="Extracting mel"):
            _extract_one(audio_path, output_dir, device)
    else:
        print(f"Using {num_workers} worker threads.")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_extract_one, f, output_dir, device): f for f in audio_files}
            with tqdm(total=len(futures), desc="Extracting mel") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"\n[warn] {futures[future]}: {exc}")
                    pbar.update(1)


# ── Public API ────────────────────────────────────────────────────────────────


def extract_mel_from_file(
    audio_path: "str | Path | list[str | Path]",
    output_dir: "str | Path | None" = None,
    device: "torch.device | None" = None,
    num_workers: int = 4,
) -> "np.ndarray | dict[str, np.ndarray]":
    """Extract log-mel spectrogram(s) from one or more audio files.

    Parameters
    ----------
    audio_path  : a single file path **or** a list of file paths.
                  Accepts any format supported by librosa
                  (.wav, .mp3, .flac, .ogg, .aiff, .m4a, .opus, …).
    output_dir  : if provided, each result is saved as ``<stem>.npy`` in
                  this directory (created automatically if it does not exist).
                  Useful for producing inputs that :meth:`SoundAQnet.predict`
                  can read via its *mel_dir* argument.
    device      : torch device for the STFT/mel-filterbank computation.
                  Auto-selects CUDA if available, otherwise CPU.
    num_workers : number of parallel worker threads when *audio_path* is a
                  list.  Pass ``1`` to process sequentially.  Ignored for a
                  single file.  Default ``4``.

    Returns
    -------
    * **Single path** → ``np.ndarray`` of shape ``(frames, 64)``, dtype float32.
    * **List of paths** → ``dict[str, np.ndarray]`` mapping each file stem to
      its ``(frames, 64)`` array.

    Examples
    --------
    Single file::

        mel = extract_mel_from_file("clip.wav")
        # shape: (T, 64)

    Multiple files, parallel, with export::

        mels = extract_mel_from_file(
            ["clip1.wav", "clip2.wav", "clip3.wav"],
            output_dir="mel_features/",
            num_workers=4,
        )
        # mels == {"clip1": array(...), "clip2": array(...), "clip3": array(...)}
        # mel_features/clip1.npy, clip2.npy, clip3.npy are also written

        # The output_dir can then be passed directly to SoundAQnet.predict():
        # df = model.predict(mel_dir="mel_features/", loudness_dir="loudness/")
    """
    single = isinstance(audio_path, (str, Path))
    paths = [Path(audio_path)] if single else [Path(p) for p in audio_path]

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_str: "str | None" = str(out)
    else:
        out_str = None

    results: dict[str, np.ndarray] = {}

    if single or num_workers <= 1 or len(paths) == 1:
        # Sequential path — use module-level cached extractors (avoids creating
        # fresh extractors per thread for trivial single-file callers).
        spec_ext, mel_ext = _get_extractors(device)
        dev = _extractor_device
        for path in paths:
            audiodata, _ = librosa.load(str(path), sr=_SAMPLE_RATE, mono=True)
            x = torch.from_numpy(audiodata).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                spectrogram = spec_ext(x)
                logmel = mel_ext(spectrogram)
            arr = logmel[0, 0].cpu().numpy().astype(np.float32)
            if out_str is not None:
                np.save(os.path.join(out_str, path.stem + ".npy"), arr)
            results[path.stem] = arr
    else:
        # Parallel path — each worker thread uses its own thread-local extractor.
        def _worker(path: Path) -> tuple[str, np.ndarray]:
            arr = _extract_one(str(path), out_str, device)
            return path.stem, arr

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_worker, p): p for p in paths}
            with tqdm(total=len(futures), desc="Extracting mel") as pbar:
                for future in as_completed(futures):
                    stem, arr = future.result()
                    results[stem] = arr
                    pbar.update(1)

    return results[paths[0].stem] if single else results


def extract_mel(
    input_dir: "str | Path",
    output_dir: "str | Path",
    num_workers: int = 4,
) -> None:
    """Extract log-mel spectrograms for every audio file in *input_dir*.

    Saves one ``.npy`` file per clip into *output_dir*.
    Skips files whose output already exists.

    Parameters
    ----------
    input_dir   : directory searched recursively for audio files.
    output_dir  : directory where ``.npy`` outputs are written.
    num_workers : parallel worker threads.  Default 4.
    """
    run_jobs(str(input_dir), str(output_dir), num_workers=num_workers)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI: soundaqnet-extract-mel"""
    parser = argparse.ArgumentParser(description="Extract log-mel spectrograms from audio files.")
    parser.add_argument(
        "--input_dir", required=True, help="Directory of audio files (searched recursively)."
    )
    parser.add_argument(
        "--output_dir", default="Dataset_mel", help="Directory for .npy output files."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of parallel worker threads (default 4)."
    )
    args = parser.parse_args()

    run_jobs(args.input_dir, args.output_dir, num_workers=args.num_workers)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)

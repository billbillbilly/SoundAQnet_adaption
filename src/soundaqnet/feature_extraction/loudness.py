"""
soundaqnet.feature_extraction.loudness
=======================================
ISO 532-1 (Zwicker method) loudness extraction — cross-platform.

Platform behaviour
------------------
Windows
    Calls the bundled ``ISO_532-1.exe`` binary via threads
    (ThreadPoolExecutor).  Fast and production-ready.

macOS / Linux
    Uses **mosqito** (``pip install mosqito>=1.2.0``), a pure-Python
    implementation of the ISO 532-1 Zwicker method.  Results are
    numerically comparable to the .exe for most practical inputs.
    mosqito's heavy work is numpy/scipy FFT-based and releases the GIL,
    so ``ThreadPoolExecutor`` delivers real parallel speedup.

Public API
----------
    extract_loudness(input_dir, output_dir, num_workers=4, ...)
                                             — batch: directory → .npy files
    extract_loudness_from_file(audio_path, num_workers=4)
                                             — single/multi file → numpy array(s)

CLI
---
    soundaqnet-extract-loudness --input_dir <audio_dir> --output_dir <out_dir> \\
                                [--num_workers N]
    soundaqnet-extract-loudness-serial  (single-threaded)
    soundaqnet-extract-loudness-para    (process-based parallel)

Parallelism
-----------
    Both batch and multi-file APIs parallelise over files with
    ``ThreadPoolExecutor``.  The Pa calibration scale is computed once
    per process (cached at module level) and reused by all threads.
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess as sbp
import warnings
import shutil
import tempfile
import threading
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files as _pkg_files

import numpy as np
import librosa
import soundfile
from tqdm import tqdm

# Audio formats supported as input.  librosa (via audioread/soundfile) handles
# all of these transparently; non-WAV files are converted in-memory before
# being written as PCM_16 WAV for the ISO_532-1.exe.
SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".opus")

warnings.filterwarnings("ignore")


# ── bundled-asset helpers ─────────────────────────────────────────────────────

def _iso532_exe() -> Path:
    """Return the path to the bundled ISO_532-1.exe (Windows only)."""
    if sys.platform != "win32":
        raise RuntimeError(
            "ISO_532-1.exe is a Windows-only binary and cannot run on "
            f"{sys.platform!r}.  On macOS/Linux, mosqito is used instead.\n"
            "  pip install 'mosqito>=1.2.0'"
        )
    exe = Path(str(_pkg_files("soundaqnet.feature_extraction") / "ISO_532_bin" / "ISO_532-1.exe"))
    if not exe.exists():
        raise FileNotFoundError(
            f"Bundled binary not found: {exe}\n"
            "Re-installing the soundaqnet package should restore it."
        )
    return exe


def _calibration_wav() -> Path:
    """Return the path to the bundled calibration WAV (1 kHz / 60 dB SPL)."""
    wav = Path(str(_pkg_files("soundaqnet.feature_extraction")
                   / "calibration_audio_file"
                   / "calibration_signal_sine_1kHz_60dB.wav"))
    if not wav.exists():
        raise FileNotFoundError(
            f"Bundled calibration WAV not found: {wav}\n"
            "Re-installing the soundaqnet package should restore it."
        )
    return wav


# ── Per-thread state ──────────────────────────────────────────────────────────
_thread_local = threading.local()
_worker_dirs: list[str] = []
_worker_dirs_lock = threading.Lock()


def _get_worker_dir(tmp_root: str) -> str:
    d = getattr(_thread_local, "dir", None)
    if d is None:
        d = tempfile.mkdtemp(prefix="iso_w_", dir=tmp_root)
        _thread_local.dir = d
        with _worker_dirs_lock:
            _worker_dirs.append(d)
    return d


# ── Utilities ─────────────────────────────────────────────────────────────────

def createDirs(dname: str) -> None:
    os.makedirs(dname, exist_ok=True)


def listFnames(dirName: str) -> list[str]:
    """Return all supported audio files under *dirName* (recursive)."""
    fnames = []
    for rootDir, _, filesList in os.walk(dirName):
        fnames += [
            os.path.join(rootDir, f) for f in filesList
            if Path(f).suffix.lower() in SUPPORTED_EXTS
        ]
    return fnames


def audio_to_output_path(audioFile: str, output_dir: str) -> str:
    return os.path.join(output_dir, Path(audioFile).stem + ".npy")


# ── Conformance check ──────────────────────────────────────────────────────────

ALLOWED_SRS = {32000, 44100, 48000}


def is_iso_conformant(src_path: str, target_sr: int) -> bool:
    """Return True only for WAV files already in the exact format the exe needs."""
    if Path(src_path).suffix.lower() != ".wav":
        return False   # non-WAV always needs conversion
    try:
        info = soundfile.info(str(src_path))
    except Exception:
        return False
    return (
        info.channels == 1
        and info.subtype == "PCM_16"
        and info.samplerate == target_sr
        and info.samplerate in ALLOWED_SRS
    )


def prepare_iso_input(src_path: str, out_dir: str, target_sr: int = 48000,
                      reuse_name: str = "input_iso.wav") -> str:
    """Convert any supported audio format to mono PCM_16 WAV at *target_sr*.

    Uses librosa so that mp3, flac, ogg, m4a, aiff, etc. are all handled
    transparently in addition to WAV.
    """
    if target_sr not in ALLOWED_SRS:
        raise ValueError(f"target_sr must be one of {sorted(ALLOWED_SRS)}")

    out_path = Path(out_dir).resolve() / reuse_name

    # librosa.load returns mono float32 at the requested sr
    x, _ = librosa.load(str(src_path), sr=target_sr, mono=True)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(x, -1.0, 1.0, out=x)

    soundfile.write(str(out_path), x, target_sr, subtype="PCM_16")
    return str(out_path)


def runProcess(loudnessExe: str, method: str, soundField: str, audioFile: str,
               refFile: str, refLevel: float, work_dir: str) -> str:
    comList = [loudnessExe, method, soundField, audioFile, refFile, str(int(refLevel))]
    proc = sbp.run(
        comList,
        stdout=sbp.PIPE, stderr=sbp.PIPE, text=True,
        encoding="utf-8", errors="replace",
        cwd=work_dir, shell=False,
    )
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    lowered = output.lower()
    if proc.returncode != 0 or "error" in lowered or "nan" in lowered:
        raise RuntimeError(output.strip() or f"ISO_532-1 failed with exit code {proc.returncode}")
    return output


def getData(csv_path: str) -> np.ndarray:
    arr = np.loadtxt(csv_path, delimiter=";", skiprows=6)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 1:]  # drop time/index column


# ── Per-file worker ───────────────────────────────────────────────────────────

def process_one(task: tuple) -> dict:
    (
        audioFile, output_dir, tmp_root, exe, meth, sf, rfile, rlev,
        target_sr, debug_paths,
    ) = task

    out_npy = audio_to_output_path(audioFile, output_dir)
    worker_tmp = _get_worker_dir(tmp_root)

    try:
        if is_iso_conformant(audioFile, target_sr):
            audio_iso = os.path.abspath(audioFile)
        else:
            audio_iso = prepare_iso_input(audioFile, out_dir=worker_tmp, target_sr=target_sr)

        if debug_paths:
            for p, label in [(audioFile, "audioFile"), (audio_iso, "audio_iso"),
                             (exe, "exe"), (rfile, "rfile"), (worker_tmp, "worker_tmp")]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{label} not found: {p}")

        runProcess(exe, meth, sf, audio_iso, rfile, rlev, worker_tmp)

        loudness_csv = os.path.join(worker_tmp, "Loudness.csv")
        if not os.path.exists(loudness_csv):
            raise RuntimeError(f"Loudness.csv not found for {audioFile}")

        loud = getData(loudness_csv)
        np.save(out_npy, loud)

        return {"status": "ok", "audioFile": audioFile, "output": out_npy, "error": ""}

    except Exception as e:
        return {
            "status": "error", "audioFile": audioFile, "output": out_npy,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# ── mosqito back-end (macOS / Linux) ─────────────────────────────────────────

_TARGET_SR_POSIX = 48000   # mosqito requires exactly 48 kHz
_CAL_DB          = 60.0    # calibration signal level in dB SPL
_P_REF_PA        = 20e-6   # ISO 1683 reference sound pressure (20 µPa)

# Module-level Pa scale cache — computed once, reused by all threads/calls.
_PA_SCALE:      float | None = None
_PA_SCALE_LOCK: threading.Lock = threading.Lock()


def _digital_to_pa_scale() -> float:
    """
    Return the Pa-per-digital-unit scale factor derived from the bundled
    calibration WAV (1 kHz sine at 60 dB SPL).

    ISO_532-1.exe uses this calibration internally.  mosqito requires input
    in Pascal units, so we must apply the same conversion before calling it.
    The result is cached at module level so that parallel workers don't all
    re-read the calibration WAV.
    """
    global _PA_SCALE
    if _PA_SCALE is not None:
        return _PA_SCALE
    with _PA_SCALE_LOCK:
        if _PA_SCALE is not None:          # re-check after acquiring lock
            return _PA_SCALE
        cal_path = _calibration_wav()
        cal, cal_sr = soundfile.read(str(cal_path), always_2d=False)
        cal = cal.astype(np.float64)
        if cal.ndim > 1:
            cal = cal.mean(axis=1)
        # Resample to 48 kHz if needed
        if cal_sr != _TARGET_SR_POSIX:
            dur   = len(cal) / cal_sr
            old_t = np.linspace(0, dur, len(cal),                           endpoint=False)
            new_t = np.linspace(0, dur, int(round(dur * _TARGET_SR_POSIX)), endpoint=False)
            cal   = np.interp(new_t, old_t, cal)
        cal_rms_digital = float(np.sqrt(np.mean(cal ** 2)))
        cal_rms_pa      = _P_REF_PA * 10 ** (_CAL_DB / 20.0)
        _PA_SCALE = cal_rms_pa / cal_rms_digital
    return _PA_SCALE


def _load_as_pa(audio_path: str, scale: float) -> tuple[np.ndarray, int]:
    """Load any supported audio format as mono 48 kHz, scaled to Pascal.

    Uses librosa so mp3/flac/ogg/m4a/aiff are all handled transparently.
    """
    x, _ = librosa.load(str(audio_path), sr=_TARGET_SR_POSIX, mono=True)
    return x.astype(np.float64) * scale, _TARGET_SR_POSIX


def _posix_worker_one(args: tuple) -> dict:
    """Thread worker: extract mosqito loudness for one file.

    Parameters are packed in a tuple for easy use with ``map``/``submit``.
    Returns a status dict matching the Windows ``process_one`` contract.
    """
    audio_path_str, out_npy_str, scale = args
    try:
        from mosqito.sq_metrics import loudness_zwtv
    except ImportError as exc:
        raise ImportError(
            "mosqito is required for loudness extraction on macOS/Linux.\n"
            "  pip install 'mosqito>=1.2.0'"
        ) from exc

    try:
        signal_pa, fs = _load_as_pa(audio_path_str, scale)
        N, _N_spec, _bark, _time = loudness_zwtv(signal_pa, fs)
        arr = np.asarray(N, dtype=np.float32)[:, None]   # (T, 1)
        np.save(out_npy_str, arr)
        return {"status": "ok", "audioFile": audio_path_str, "output": out_npy_str, "error": ""}
    except Exception as exc:
        import traceback
        return {
            "status": "error", "audioFile": audio_path_str, "output": out_npy_str,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _extract_loudness_posix(
    input_dir:   Path,
    output_dir:  Path,
    num_workers: int = 4,
    chunk_size:  int = 5000,
) -> None:
    """ISO 532-1 loudness extraction via mosqito (macOS / Linux).

    Applies the same Pa calibration as ISO_532-1.exe so that the output
    loudness values are numerically equivalent across platforms.
    Saves each file as shape ``(T, 1)`` — total loudness in sone over time —
    matching the Windows exe output format used during model training.

    mosqito's FFT/numpy operations release the GIL so multiple threads can
    run mosqito concurrently for real parallel speedup.

    Parameters
    ----------
    input_dir   : directory of audio files.
    output_dir  : directory for ``.npy`` output files.
    num_workers : parallel worker threads (default 4).
    chunk_size  : files submitted to the thread pool per chunk (caps memory).
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute calibration scale once (cached for subsequent calls).
    scale = _digital_to_pa_scale()
    print(f"[mosqito] Pa/digital scale = {scale:.5f}  "
          f"(from {_calibration_wav().name} at {_CAL_DB} dB SPL)")

    audio_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTS
    )

    # Skip already-completed outputs.
    todo = [f for f in audio_files
            if not (output_dir / (f.stem + ".npy")).exists()]
    print(f"[mosqito] {len(audio_files)} files found; "
          f"{len(audio_files) - len(todo)} already done; "
          f"{len(todo)} remaining.")

    if not todo:
        return

    ok_count = err_count = 0
    error_log = output_dir / "errors.log"

    if num_workers <= 1:
        # Sequential fallback.
        for audio_file in tqdm(todo, desc="Extracting loudness"):
            out_npy = str(output_dir / (audio_file.stem + ".npy"))
            result  = _posix_worker_one((str(audio_file), out_npy, scale))
            if result["status"] == "ok":
                ok_count += 1
            else:
                err_count += 1
                with open(str(error_log), "a") as fh:
                    fh.write(f"FILE: {result['audioFile']}\n{result['error']}\n{'='*80}\n")
    else:
        print(f"[mosqito] Using {num_workers} worker threads.")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(todo), desc="Extracting loudness") as pbar:
                for chunk_start in range(0, len(todo), chunk_size):
                    chunk   = todo[chunk_start: chunk_start + chunk_size]
                    tasks   = [
                        (str(f), str(output_dir / (f.stem + ".npy")), scale)
                        for f in chunk
                    ]
                    futures = [executor.submit(_posix_worker_one, t) for t in tasks]
                    for future in as_completed(futures):
                        result = future.result()
                        if result["status"] == "ok":
                            ok_count += 1
                        else:
                            err_count += 1
                            with open(str(error_log), "a") as fh:
                                fh.write(
                                    f"FILE: {result['audioFile']}\n"
                                    f"{result['error']}\n{'='*80}\n"
                                )
                        pbar.update(1)

    print(f"[mosqito] Done.  OK: {ok_count}  Errors: {err_count}"
          + (f"  Log: {error_log}" if err_count else ""))


# ── public API ────────────────────────────────────────────────────────────────

def _extract_loudness_single_posix(audio_path: Path) -> np.ndarray:
    """Single-file loudness extraction using mosqito (macOS / Linux)."""
    try:
        from mosqito.sq_metrics import loudness_zwtv
    except ImportError as exc:
        raise ImportError(
            "mosqito is required for loudness extraction on macOS/Linux.\n"
            "  pip install 'mosqito>=1.2.0'"
        ) from exc

    scale = _digital_to_pa_scale()
    signal_pa, fs = _load_as_pa(str(audio_path), scale)
    N, _N_spec, _bark, _time = loudness_zwtv(signal_pa, fs)
    return np.asarray(N, dtype=np.float32)[:, None]   # (T, 1)


def extract_loudness_from_file(
    audio_path:  str | Path | list[str | Path],
    output_dir:  str | Path | None = None,
    target_sr:   int = 48000,
    method:      str = "Varying",
    sound_field: str = "Free",
    num_workers: int = 4,
) -> np.ndarray | dict[str, np.ndarray]:
    """Extract ISO 532-1 Zwicker loudness from one or more audio files.

    Dispatches to the bundled ISO_532-1.exe on Windows, or mosqito on
    macOS / Linux.  When *audio_path* is a list and *num_workers* > 1,
    files are processed in parallel using ``ThreadPoolExecutor``.

    Parameters
    ----------
    audio_path  : a single file path **or** a list of file paths.
                  Accepts any format supported by librosa
                  (.wav, .mp3, .flac, .ogg, .aiff, .m4a, .opus, …).
    output_dir  : if provided, each result is saved as ``<stem>.npy`` in
                  this directory (created automatically if it does not exist).
                  Useful for producing inputs that :meth:`SoundAQnet.predict`
                  can read via its *loudness_dir* argument.
    target_sr   : sample rate for the ISO computation (32000 / 44100 / 48000).
                  Defaults to 48000 Hz (recommended).
    method      : ``"Varying"`` (time-varying, default) or ``"Stationary"``.
                  Windows only; mosqito always uses the time-varying method.
    sound_field : ``"Free"`` (default) or ``"Diffuse"``.
                  Windows only; mosqito always uses free-field.
    num_workers : number of parallel worker threads when *audio_path* is a
                  list.  Pass ``1`` for sequential processing.  Ignored for
                  a single file.  Default ``4``.

    Returns
    -------
    * **Single path** → ``np.ndarray`` of shape ``(T, 1)``, dtype float32.
    * **List of paths** → ``dict[str, np.ndarray]`` mapping each file stem to
      its ``(T, 1)`` array.

    Examples
    --------
    Single file::

        loud = extract_loudness_from_file("clip.wav")
        # shape: (T, 1)

    Multiple files, parallel, with export::

        louds = extract_loudness_from_file(
            ["clip1.wav", "clip2.wav", "clip3.wav"],
            output_dir="loudness_features/",
            num_workers=4,
        )
        # louds == {"clip1": array(...), "clip2": array(...), ...}
        # loudness_features/clip1.npy, clip2.npy, clip3.npy written

        # The output_dir can then be passed directly to SoundAQnet.predict():
        # df = model.predict(mel_dir="mel/", loudness_dir="loudness_features/")
    """
    single = isinstance(audio_path, (str, Path))
    paths  = [Path(audio_path)] if single else [Path(p) for p in audio_path]

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_str: str | None = str(out)
    else:
        out_str = None

    def _one(path: Path) -> np.ndarray:
        """Extract loudness for a single file and optionally save."""
        if sys.platform != "win32":
            arr = _extract_loudness_single_posix(path)
        else:
            arr = _extract_loudness_single_windows(path, target_sr, method, sound_field)
        if out_str is not None:
            np.save(os.path.join(out_str, path.stem + ".npy"), arr)
        return arr

    results: dict[str, np.ndarray] = {}

    if single or num_workers <= 1 or len(paths) == 1:
        # Sequential path.
        for path in tqdm(paths, desc="Extracting loudness", disable=single):
            results[path.stem] = _one(path)
    else:
        # Parallel path — threads overlap I/O and mosqito/exe computation.
        # mosqito's numpy/scipy FFT calls release the GIL so parallel speedup
        # is real.  Windows workers each spawn a subprocess per file (already
        # parallel-safe via per-thread temp dirs).
        def _worker(path: Path) -> tuple[str, np.ndarray]:
            return path.stem, _one(path)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_worker, p): p for p in paths}
            with tqdm(total=len(futures), desc="Extracting loudness") as pbar:
                for future in as_completed(futures):
                    stem, arr = future.result()
                    results[stem] = arr
                    pbar.update(1)

    return results[paths[0].stem] if single else results


def _extract_loudness_single_windows(
    audio_path: Path,
    target_sr: int = 48000,
    method: str = "Varying",
    sound_field: str = "Free",
) -> np.ndarray:
    """Single-file loudness extraction using ISO_532-1.exe (Windows)."""
    import tempfile, shutil

    METHOD_DICT = {"Varying": "Time_varying", "Stationary": "Stationary"}
    SF_DICT = {"Free": "F", "Diffuse": "D"}
    exe   = str(_iso532_exe())
    rfile = str(_calibration_wav())
    meth  = METHOD_DICT[method]
    sf    = SF_DICT[sound_field]

    tmp_dir = tempfile.mkdtemp(prefix="soundaqnet_loud_")
    try:
        if is_iso_conformant(str(audio_path), target_sr):
            audio_iso = str(audio_path.resolve())
        else:
            audio_iso = prepare_iso_input(str(audio_path), out_dir=tmp_dir, target_sr=target_sr)

        runProcess(exe, meth, sf, audio_iso, rfile, 60, tmp_dir)

        loudness_csv = os.path.join(tmp_dir, "Loudness.csv")
        if not os.path.exists(loudness_csv):
            raise RuntimeError(f"ISO_532-1.exe did not produce Loudness.csv for {audio_path}")

        loud = getData(loudness_csv)
        return loud.astype(np.float32)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_loudness(
    input_dir: str | Path,
    output_dir: str | Path,
    tmp_dir: str | Path | None = None,
    num_workers: int = 4,
    chunk_size: int = 5000,
    target_sr: int = 48000,
    method: str = "Varying",
    sound_field: str = "Free",
    overwrite: bool = False,
) -> None:
    """
    Extract ISO 532-1 Zwicker loudness features from all ``.wav`` files.

    Dispatches to ISO_532-1.exe (Windows) or mosqito (macOS/Linux).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform != "win32":
        if tmp_dir is not None:
            warnings.warn("tmp_dir is ignored on macOS/Linux (mosqito back-end).", stacklevel=2)
        _extract_loudness_posix(input_dir, output_dir, num_workers, chunk_size)
        return

    # Windows path ─────────────────────────────────────────────────────────────
    exe = str(_iso532_exe())
    rfile = str(_calibration_wav())
    rlev = 60

    METHOD_DICT = {"Varying": "Time_varying", "Stationary": "Stationary"}
    SF_DICT = {"Free": "F", "Diffuse": "D"}
    meth = METHOD_DICT[method]
    sf = SF_DICT[sound_field]

    if tmp_dir is None:
        tmp_dir = output_dir / "_tmp"
    tmp_root = str(Path(tmp_dir).resolve())
    createDirs(tmp_root)

    audioFiles = sorted(listFnames(str(input_dir)))
    print(f"Found {len(audioFiles)} audio files ({', '.join(SUPPORTED_EXTS)})")

    if not overwrite:
        existing = {Path(e.name).stem for e in os.scandir(str(output_dir))
                    if e.is_file() and e.name.endswith(".npy")}
        before = len(audioFiles)
        audioFiles = [f for f in audioFiles if Path(f).stem not in existing]
        print(f"Skipping {before - len(audioFiles)} already-complete; {len(audioFiles)} remaining")

    error_log_path = os.path.join(str(output_dir), "errors.log")
    ok_count = err_count = 0

    def make_task(af):
        return (af, str(output_dir), tmp_root, exe, meth, sf, rfile, rlev, target_sr, False)

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(audioFiles), desc="Processing") as pbar:
                for chunk_start in range(0, len(audioFiles), chunk_size):
                    chunk = audioFiles[chunk_start: chunk_start + chunk_size]
                    futures = [executor.submit(process_one, make_task(f)) for f in chunk]
                    for future in as_completed(futures):
                        result = future.result()
                        if result["status"] == "ok":
                            ok_count += 1
                        else:
                            err_count += 1
                            with open(error_log_path, "a", encoding="utf-8") as fh:
                                fh.write(f"FILE: {result['audioFile']}\n{result['error']}\n{'='*80}\n")
                        pbar.update(1)
    finally:
        with _worker_dirs_lock:
            for d in _worker_dirs:
                shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nDone.  OK: {ok_count}  Errors: {err_count}  Log: {error_log_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> int:
    """CLI: soundaqnet-extract-loudness"""
    METHOD_DICT = {"Varying": "Time_varying", "Stationary": "Stationary"}
    SF_DICT = {"Free": "F", "Diffuse": "D"}

    parser = argparse.ArgumentParser(
        description="Extract ISO 532-1 loudness features (Windows: .exe; macOS/Linux: mosqito).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir",   required=True,
                        help="Directory of audio files (.wav, .mp3, .flac, .ogg, .aiff, .m4a, .opus). Searched recursively.")
    parser.add_argument("--output_dir",  default="Dataset_wav_loudness",
                        help="Output directory for .npy loudness files.")
    parser.add_argument("--tmp_dir",     default=None,
                        help="Temp dir for worker threads (Windows only).")
    parser.add_argument("--target_sr",   type=int, default=48000, choices=[32000, 44100, 48000])
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--chunk_size",  type=int, default=5000)
    parser.add_argument("--start_idx",   type=int, default=0)
    parser.add_argument("--end_idx",     type=int, default=None)
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--method",      default="Varying", choices=["Varying", "Stationary"])
    parser.add_argument("--sound_field", default="Free",    choices=["Free", "Diffuse"])
    parser.add_argument("--debug_paths", action="store_true")
    args = parser.parse_args()

    extract_loudness(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tmp_dir=args.tmp_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        target_sr=args.target_sr,
        method=args.method,
        sound_field=args.sound_field,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

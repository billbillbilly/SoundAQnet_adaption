"""
soundaqnet.feature_extraction.loudness_parallel
================================================
Process-based parallel ISO 532-1 loudness extraction (Windows-only via ISO_532-1.exe).

Uses ``ProcessPoolExecutor`` — each worker process gets its own temp directory,
so there is no file-collision between workers.  Ideal for multi-core machines.

For thread-based parallelism (lower overhead, shared memory), use
``soundaqnet-extract-loudness`` instead.

CLI
---
    soundaqnet-extract-loudness-para --input_dir <wav_dir> --output_dir <out>
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import subprocess as sbp
import sys
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files as _pkg_files
from pathlib import Path

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".opus")

warnings.filterwarnings("ignore")


# ── bundled-asset helpers ─────────────────────────────────────────────────────


def _iso532_exe() -> Path:
    if sys.platform != "win32":
        raise RuntimeError(
            "ISO_532-1.exe is Windows-only.  "
            "On macOS/Linux use: soundaqnet-extract-loudness (mosqito back-end)."
        )
    exe = Path(str(_pkg_files("soundaqnet.feature_extraction") / "ISO_532_bin" / "ISO_532-1.exe"))
    if not exe.exists():
        raise FileNotFoundError(f"Bundled binary not found: {exe}")
    return exe


def _calibration_wav() -> Path:
    wav = Path(
        str(
            _pkg_files("soundaqnet.feature_extraction")
            / "calibration_audio_file"
            / "calibration_signal_sine_1kHz_60dB.wav"
        )
    )
    if not wav.exists():
        raise FileNotFoundError(f"Bundled calibration WAV not found: {wav}")
    return wav


# ── helpers ───────────────────────────────────────────────────────────────────


def createDirs(dname: str) -> None:
    if not os.path.isdir(dname):
        os.makedirs(dname, exist_ok=True)


def listFnames(dirName: str) -> list[str]:
    """Return all supported audio files under *dirName* (recursive)."""
    fnames = []
    for rootDir, _, filesList in os.walk(dirName):
        fnames += [
            os.path.join(rootDir, f) for f in filesList if Path(f).suffix.lower() in SUPPORTED_EXTS
        ]
    return fnames


def prepare_iso_input(src_path: str, out_dir: str, target_sr: int = 48000) -> str:
    """Convert any supported audio format to mono PCM_16 WAV at *target_sr*."""
    allowed_srs = {32000, 44100, 48000}
    if target_sr not in allowed_srs:
        raise ValueError(f"target_sr must be one of {sorted(allowed_srs)}")

    src_path = Path(src_path).resolve()
    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    x, _ = librosa.load(str(src_path), sr=target_sr, mono=True)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(x, -1.0, 1.0, out=x)

    out_path = out_dir_p / f"{src_path.stem}_iso.wav"
    soundfile.write(str(out_path), x, target_sr, subtype="PCM_16")
    return str(out_path.resolve())


def runProcess(
    loudnessExe: str,
    method: str,
    soundField: str,
    audioFile: str,
    refFile: str,
    refLevel: float,
    work_dir: str,
) -> str:
    loudnessExe = os.path.abspath(loudnessExe)
    audioFile = os.path.abspath(audioFile)
    refFile = os.path.abspath(refFile)
    work_dir = os.path.abspath(work_dir)

    comList = [loudnessExe, method, soundField, audioFile, refFile, str(int(refLevel))]
    proc = sbp.run(
        comList,
        stdout=sbp.PIPE,
        stderr=sbp.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=work_dir,
        shell=False,
    )
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    lowered = output.lower()
    if proc.returncode != 0 or "error" in lowered or "nan" in lowered:
        raise RuntimeError(output.strip() or f"ISO_532-1 failed with exit code {proc.returncode}")
    return output


def getData(csv_path: str) -> np.ndarray:
    import pandas as pd

    csv_path = os.path.abspath(csv_path)
    dfLoudness = pd.read_csv(csv_path, sep=";", skiprows=5, index_col=0)
    return dfLoudness.to_numpy()


def audio_to_output_path(audioFile: str, output_dir: str) -> str:
    return os.path.join(output_dir, Path(audioFile).stem + ".npy")


# ── Top-level worker (must be module-level for Windows multiprocessing) ───────


def process_one(task: tuple) -> dict:
    (
        audioFile,
        output_dir,
        tmp_root,
        exe,
        meth,
        sf,
        rfile,
        rlev,
        target_sr,
        overwrite,
        debug_paths,
    ) = task

    audioFile = os.path.abspath(audioFile)
    output_dir = os.path.abspath(output_dir)
    tmp_root = os.path.abspath(tmp_root)
    exe = os.path.abspath(exe)
    rfile = os.path.abspath(rfile)

    out_npy = audio_to_output_path(audioFile, output_dir)

    if (not overwrite) and os.path.exists(out_npy):
        return {"status": "skipped", "audioFile": audioFile, "output": out_npy, "error": ""}

    worker_tmp = tempfile.mkdtemp(prefix="iso_", dir=tmp_root)

    try:
        audio_iso = prepare_iso_input(audioFile, out_dir=worker_tmp, target_sr=target_sr)

        if debug_paths:
            for p, label in [
                (audioFile, "audioFile"),
                (audio_iso, "audio_iso"),
                (exe, "exe"),
                (rfile, "rfile"),
                (worker_tmp, "worker_tmp"),
            ]:
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
            "status": "error",
            "audioFile": audioFile,
            "output": out_npy,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }

    finally:
        shutil.rmtree(worker_tmp, ignore_errors=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI: soundaqnet-extract-loudness-para"""
    METHOD_DICT = {"Varying": "Time_varying", "Stationary": "Stationary"}
    SF_DICT = {"Free": "F", "Diffuse": "D"}

    parser = argparse.ArgumentParser(
        description="Process-parallel ISO 532-1 loudness extraction (Windows only)."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="Dataset_wav_loudness")
    parser.add_argument("--target_sr", type=int, default=48000, choices=[32000, 44100, 48000])
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method", default="Varying", choices=["Varying", "Stationary"])
    parser.add_argument("--sound_field", default="Free", choices=["Free", "Diffuse"])
    parser.add_argument("--debug_paths", action="store_true")
    args = parser.parse_args()

    if sys.platform != "win32":
        print("ERROR: soundaqnet-extract-loudness-para requires Windows (ISO_532-1.exe).")
        print("On macOS/Linux, use:  soundaqnet-extract-loudness  (mosqito back-end)")
        return 1

    exe = str(_iso532_exe())
    rfile = str(_calibration_wav())
    rlev = 60
    meth = METHOD_DICT[args.method]
    sf = SF_DICT[args.sound_field]

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    tmp_root = os.path.abspath(os.path.join(output_dir, "_tmp"))

    createDirs(output_dir)
    createDirs(tmp_root)

    audioFiles = sorted(listFnames(input_dir))
    total_found = len(audioFiles)
    print(f"Found {total_found} wav files in: {input_dir}")

    start_idx = max(0, args.start_idx)
    end_idx = args.end_idx if args.end_idx is not None else total_found
    end_idx = min(end_idx, total_found)

    if start_idx >= end_idx:
        print(f"Nothing to do: start_idx={start_idx}, end_idx={end_idx}")
        return 0

    audioFiles = audioFiles[start_idx:end_idx]

    if not args.overwrite:
        pending = [f for f in audioFiles if not os.path.exists(audio_to_output_path(f, output_dir))]
        skip_count = len(audioFiles) - len(pending)
        audioFiles = pending
        print(f"Skipping {skip_count} already-complete; {len(audioFiles)} remaining")

    print(f"Output dir: {output_dir}  Workers: {args.num_workers}  SR: {args.target_sr}")

    tasks = [
        (
            af,
            output_dir,
            tmp_root,
            exe,
            meth,
            sf,
            rfile,
            rlev,
            args.target_sr,
            args.overwrite,
            args.debug_paths,
        )
        for af in audioFiles
    ]

    ok_count = err_count = skip_count_run = 0
    error_log_path = os.path.join(output_dir, "errors.log")

    with open(error_log_path, "w", encoding="utf-8") as fh:
        fh.write(f"ISO loudness parallel run\ninput_dir={input_dir}\n\n")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result["status"] == "ok":
                ok_count += 1
            elif result["status"] == "skipped":
                skip_count_run += 1
            else:
                err_count += 1
                with open(error_log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"FILE: {result['audioFile']}\n{result['error']}\n{'='*80}\n")

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\nDone.  OK: {ok_count}  Skipped: {skip_count_run}  Errors: {err_count}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())

"""
soundaqnet.feature_extraction.loudness_serial
==============================================
Single-threaded ISO 532-1 loudness extraction (Windows-only via ISO_532-1.exe).

For batch jobs, prefer ``soundaqnet-extract-loudness`` (threaded) or
``soundaqnet-extract-loudness-para`` (process-based).

CLI
---
    soundaqnet-extract-loudness-serial --input_dir <wav_dir> --output_dir <out>
"""

from __future__ import annotations

import argparse
import os
import subprocess as sbp
import sys
import warnings
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
            "ISO_532-1.exe is a Windows-only binary.  "
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
        os.makedirs(dname)


def listFnames(dirName: str) -> list[str]:
    """Return all supported audio files under *dirName* (recursive)."""
    fnames = []
    for rootDir, _, filesList in os.walk(dirName):
        fnames += [
            os.path.join(rootDir, f) for f in filesList if Path(f).suffix.lower() in SUPPORTED_EXTS
        ]
    return fnames


def prepare_iso_input(src_path: str, out_dir: str = "pcm16", target_sr: int = 48000) -> str:
    """Convert any supported audio format to mono PCM_16 WAV at *target_sr*."""
    allowed_srs = {32000, 44100, 48000}
    if target_sr not in allowed_srs:
        raise ValueError(f"target_sr must be one of {sorted(allowed_srs)}")

    src_path = Path(src_path)
    out_dir_p = src_path.parent if out_dir is None else Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    x, _ = librosa.load(str(src_path), sr=target_sr, mono=True)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(x, -1.0, 1.0, out=x)

    out_path = out_dir_p / f"{src_path.stem}_iso.wav"
    soundfile.write(str(out_path), x, target_sr, subtype="PCM_16")
    return str(out_path)


def runProcess(
    loudnessExe: str, method: str, soundField: str, audioFile: str, refFile: str, refLevel: float
) -> str:
    comList = [loudnessExe, method, soundField, audioFile, refFile, str(int(refLevel))]
    proc = sbp.run(
        comList,
        stdout=sbp.PIPE,
        stderr=sbp.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    lowered = output.lower()
    if proc.returncode != 0 or "error" in lowered or "nan" in lowered:
        raise RuntimeError(output.strip() or f"ISO_532-1 failed with exit code {proc.returncode}")
    return output


def getData() -> np.ndarray:
    import pandas as pd

    dfLoudness = pd.read_csv("Loudness.csv", sep=";", skiprows=5, index_col=0)
    return dfLoudness.to_numpy()


def removeTemp() -> None:
    for fname in ("Loudness.csv", "SpecLoudness.csv"):
        if os.path.exists(fname):
            os.remove(fname)


# ── main / CLI ────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI: soundaqnet-extract-loudness-serial"""
    METHOD_DICT = {"Varying": "Time_varying", "Stationary": "Stationary"}
    SF_DICT = {"Free": "F", "Diffuse": "D"}

    parser = argparse.ArgumentParser(
        description="Single-threaded ISO 532-1 loudness extraction (Windows only)."
    )
    parser.add_argument("--input_dir", required=True, help="Directory of .wav files.")
    parser.add_argument(
        "--output_dir",
        default="Dataset_wav_loudness",
        help="Output directory for .npy loudness files.",
    )
    parser.add_argument("--target_sr", type=int, default=48000, choices=[32000, 44100, 48000])
    args = parser.parse_args()

    if sys.platform != "win32":
        print("ERROR: soundaqnet-extract-loudness-serial requires Windows (ISO_532-1.exe).")
        print("On macOS/Linux, use:  soundaqnet-extract-loudness  (mosqito back-end)")
        return 1

    exe = str(_iso532_exe())
    rfile = str(_calibration_wav())
    rlev = 60
    meth = METHOD_DICT["Varying"]
    sf = SF_DICT["Free"]

    input_dir = args.input_dir
    output_dir = args.output_dir
    target_sr = args.target_sr

    createDirs(output_dir)
    createDirs("pcm16")

    audioFiles = listFnames(input_dir)
    print(f"Found {len(audioFiles)} audio files")

    for audioFile in tqdm(audioFiles, total=len(audioFiles)):
        fileDir = os.path.join(output_dir, Path(audioFile).stem + ".npy")

        try:
            audio_iso = prepare_iso_input(audioFile, out_dir="pcm16", target_sr=target_sr)
            runProcess(exe, meth, sf, audio_iso, rfile, rlev)
            loud = getData()
            np.save(fileDir, loud)
            removeTemp()
        except Exception as e:
            print(f"Skip {audioFile} due to error: {e}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())

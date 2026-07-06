"""
soundaqnet.feature_extraction.emosoundscape_features
====================================================
Reproducible audio features for Emo-Soundscape valence/arousal models.

The original Emo-Soundscapes papers used YAAFE/MIRToolbox/Essentia-derived
features. Those exact extractors are not bundled with the dataset, so this
module provides a package-native feature set. Models trained with this module
can be used for inference from arbitrary audio through the same code path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import get_window, resample_poly
from tqdm import tqdm

SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".opus")
SAMPLE_RATE = 44_100
FRAME_SIZE = 4096
HOP_SIZE = 2048
N_MELS = 28
N_MFCC = 13
N_CHROMA = 12

BASE_FEATURES = [
    "rms",
    "energy",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff85",
    "spectral_flatness",
    "spectral_flux",
]
EMOSOUNDSCAPE_NATIVE_FEATURE_NAMES = (
    [f"{name}_{stat}" for name in BASE_FEATURES for stat in ("mean", "std")]
    + [f"mfcc_{i}_{stat}" for i in range(1, N_MFCC + 1) for stat in ("mean", "std")]
    + [f"chroma_{i}_{stat}" for i in range(1, N_CHROMA + 1) for stat in ("mean", "std")]
    + [f"mel_{i}_{stat}" for i in range(1, N_MELS + 1) for stat in ("mean", "std")]
)


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 50.0) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(sr / 2), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)

    for i in range(n_mels):
        left, center, right = hz_points[i : i + 3]
        left_slope = (freqs - left) / max(center - left, 1e-12)
        right_slope = (right - freqs) / max(right - center, 1e-12)
        fb[i] = np.maximum(0.0, np.minimum(left_slope, right_slope))
    return fb


def _load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, file_sr = sf.read(str(path), always_2d=True)
    audio = audio.astype(np.float32).mean(axis=1)
    if file_sr != sr:
        gcd = np.gcd(file_sr, sr)
        audio = resample_poly(audio, sr // gcd, file_sr // gcd).astype(np.float32)
    if not np.any(np.isfinite(audio)):
        raise ValueError(f"Audio contains no finite samples: {path}")
    return np.nan_to_num(audio)


def _frame_audio(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    n_frames = 1 + int(np.ceil((len(audio) - frame_size) / hop_size))
    padded_len = frame_size + (n_frames - 1) * hop_size
    if len(audio) < padded_len:
        audio = np.pad(audio, (0, padded_len - len(audio)))
    shape = (n_frames, frame_size)
    strides = (audio.strides[0] * hop_size, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)), float(np.std(values))


def extract_emosoundscape_features_from_array(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE,
) -> np.ndarray:
    """Extract the 122-D package-native Emo-Soundscape feature vector from audio."""
    if sr != SAMPLE_RATE:
        gcd = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio.astype(np.float32), SAMPLE_RATE // gcd, sr // gcd)
        sr = SAMPLE_RATE

    frames = _frame_audio(np.asarray(audio, dtype=np.float32), frame_size, hop_size)
    window = get_window("hann", frame_size, fftbins=True).astype(np.float32)
    windowed = frames * window[None, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    mag = np.abs(spectrum).astype(np.float64)
    power = np.maximum(mag**2, 1e-20)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)

    rms = np.sqrt(np.mean(frames**2, axis=1))
    energy = np.mean(frames**2, axis=1)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    mag_sum = np.maximum(np.sum(mag, axis=1), 1e-20)
    centroid = np.sum(mag * freqs[None, :], axis=1) / mag_sum
    bandwidth = np.sqrt(np.sum(mag * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / mag_sum)
    cumulative = np.cumsum(mag, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85 * cumulative[:, [-1]], axis=1)
    rolloff = freqs[rolloff_idx]
    flatness = np.exp(np.mean(np.log(np.maximum(mag, 1e-20)), axis=1)) / np.maximum(
        np.mean(mag, axis=1), 1e-20
    )
    norm_mag = mag / mag_sum[:, None]
    flux = np.sqrt(np.sum(np.diff(norm_mag, axis=0, prepend=norm_mag[[0]]) ** 2, axis=1))

    fb = _mel_filterbank(sr, frame_size, N_MELS)
    mel_energy = np.maximum(power @ fb.T, 1e-20)
    log_mel = np.log(mel_energy)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]

    chroma = np.zeros((len(frames), N_CHROMA), dtype=np.float64)
    positive = freqs > 0
    midi = np.round(12.0 * np.log2(freqs[positive] / 440.0) + 69.0).astype(int)
    pitch_classes = np.mod(midi, N_CHROMA)
    for pc in range(N_CHROMA):
        chroma[:, pc] = np.sum(power[:, positive][:, pitch_classes == pc], axis=1)
    chroma = chroma / np.maximum(chroma.sum(axis=1, keepdims=True), 1e-20)

    features: list[float] = []
    for values in (rms, energy, zcr, centroid, bandwidth, rolloff, flatness, flux):
        features.extend(_mean_std(values))
    for i in range(N_MFCC):
        features.extend(_mean_std(mfcc[:, i]))
    for i in range(N_CHROMA):
        features.extend(_mean_std(chroma[:, i]))
    for i in range(N_MELS):
        features.extend(_mean_std(log_mel[:, i]))
    return np.asarray(features, dtype=np.float32)


def extract_emosoundscape_features_from_file(audio_path: str | Path) -> np.ndarray:
    """Extract features from one audio file."""
    return extract_emosoundscape_features_from_array(_load_audio(audio_path), sr=SAMPLE_RATE)


def extract_emosoundscape_features(
    audio_dir: str | Path | None = None,
    audio_files: list[str | Path] | None = None,
    output_csv: str | Path | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract package-native Emo-Soundscape features for audio files."""
    if audio_files is None:
        if audio_dir is None:
            raise ValueError("Specify either audio_dir or audio_files")
        audio_files = sorted(
            str(p)
            for p in Path(audio_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("._")
        )
    else:
        audio_files = [str(p) for p in audio_files]

    rows = []
    for path in tqdm(audio_files, desc="Emo features", disable=not show_progress):
        vec = extract_emosoundscape_features_from_file(path)
        rows.append({"fileName": Path(path).name, **dict(zip(EMOSOUNDSCAPE_NATIVE_FEATURE_NAMES, vec))})
    df = pd.DataFrame(rows)
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract package-native Emo-Soundscape features.")
    parser.add_argument("--audio_dir", type=str, help="Directory of audio files.")
    parser.add_argument("--audio_file", action="append", help="Audio file; can be repeated.")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    extract_emosoundscape_features(
        audio_dir=args.audio_dir,
        audio_files=args.audio_file,
        output_csv=args.output_csv,
    )
    print(f"Saved features -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

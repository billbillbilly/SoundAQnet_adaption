"""
soundaqnet.feature_extraction.mel_spectrogram
==============================================
Log-mel spectrogram extraction from mono audio clips.

Produces one ``.npy`` file per ``.wav`` file.
Shape: (frames, mel_bins) = (T, 64) at 16 kHz / hop_size=160.

CLI
---
    soundaqnet-extract-mel --input_dir <wav_dir> --output_dir <out_dir>
"""

from __future__ import annotations

import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import librosa
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
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
            os.path.join(rootDir, f) for f in filesList
            if Path(f).suffix.lower() in SUPPORTED_EXTS
        ]
    return fnames


def pad_or_truncate(x: np.ndarray, audio_length: int) -> np.ndarray:
    """Pad all audio to a specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0:audio_length]


def move_data_to_device(x: np.ndarray, device: torch.device) -> torch.Tensor:
    if 'float' in str(x.dtype):
        x_t = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x_t = torch.LongTensor(x)
    else:
        return x  # type: ignore[return-value]
    return x_t.to(device)


# ── extraction ────────────────────────────────────────────────────────────────

def run_jobs(input_dir: str, output_dir: str) -> None:
    """Extract log-mel spectrograms for every .wav in *input_dir*."""
    create_folder(output_dir)

    mel_bins = 64
    sample_rate = 16000
    fmax = int(sample_rate / 2)
    fmin = 50
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    window_size = 512
    hop_size = 160

    spectrogram_extractor = Spectrogram(
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        center=center,
        pad_mode=pad_mode,
        freeze_parameters=True,
    )

    logmel_extractor = LogmelFilterBank(
        sr=sample_rate,
        n_fft=window_size,
        n_mels=mel_bins,
        fmin=fmin,
        fmax=fmax,
        ref=ref,
        amin=amin,
        top_db=top_db,
        freeze_parameters=True,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    spectrogram_extractor.to(device)
    logmel_extractor.to(device)

    audio_files = listFnames(input_dir)
    print(f"Found {len(audio_files)} audio files ({', '.join(SUPPORTED_EXTS)})")

    # Skip already-completed outputs
    audio_files = [f for f in audio_files
                   if not os.path.exists(
                       os.path.join(output_dir, Path(f).stem + ".npy")
                   )]
    print(f"Skipping already-complete files; {len(audio_files)} remaining")

    for n, audio_path in tqdm(enumerate(audio_files)):
        output_feature = os.path.join(output_dir, Path(audio_path).stem + ".npy")

        audiodata, fs = librosa.load(audio_path, sr=sample_rate, mono=True)

        spectrogram = spectrogram_extractor(
            move_data_to_device(audiodata[None, :], device)
        )

        logmel = logmel_extractor(spectrogram)
        logmel = logmel[0, 0].data.cpu().numpy()

        np.save(output_feature, logmel)


# Programmatic API
def extract_mel(input_dir: str, output_dir: str) -> None:
    """Extract log-mel spectrograms for every .wav in *input_dir*."""
    run_jobs(input_dir, output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    """CLI: soundaqnet-extract-mel"""
    parser = argparse.ArgumentParser(
        description="Extract log-mel spectrograms from .wav files."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory of mono .wav files.")
    parser.add_argument("--output_dir", default="Dataset_mel",
                        help="Directory for .npy output files.")
    args = parser.parse_args()

    run_jobs(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)

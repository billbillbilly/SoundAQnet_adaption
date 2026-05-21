"""
soundaqnet.feature_extraction
==============================
Utilities for extracting ISO 532-1 loudness and log-mel spectrograms
from audio clips.

Single-file API (returns numpy arrays directly)
-----------------------------------------------
    extract_mel_from_file(audio_path)        → np.ndarray (T, 64)
    extract_loudness_from_file(audio_path)   → np.ndarray (T, 1)

Batch API (reads a directory, writes .npy files)
-------------------------------------------------
    extract_mel(input_dir, output_dir)
    extract_loudness(input_dir, output_dir, ...)
"""

from soundaqnet.feature_extraction.loudness import (
    extract_loudness,
    extract_loudness_from_file,
)
from soundaqnet.feature_extraction.mel_spectrogram import (
    extract_mel,
    extract_mel_from_file,
)

__all__ = [
    "extract_loudness",
    "extract_loudness_from_file",
    "extract_mel",
    "extract_mel_from_file",
]

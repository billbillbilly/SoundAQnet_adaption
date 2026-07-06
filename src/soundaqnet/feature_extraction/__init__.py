"""
soundaqnet.feature_extraction
==============================
Utilities for extracting ISO 532-1 loudness, log-mel spectrograms, and
Emo-Soundscape tabular features from audio clips.

Imports are lazy so lightweight feature extractors do not require every audio
dependency to be installed at import time.
"""

__all__ = [
    "EMOSOUNDSCAPE_NATIVE_FEATURE_NAMES",
    "extract_emosoundscape_features",
    "extract_emosoundscape_features_from_array",
    "extract_emosoundscape_features_from_file",
    "extract_loudness",
    "extract_loudness_from_file",
    "extract_mel",
    "extract_mel_from_file",
]


def __getattr__(name: str):
    if name in {
        "EMOSOUNDSCAPE_NATIVE_FEATURE_NAMES",
        "extract_emosoundscape_features",
        "extract_emosoundscape_features_from_array",
        "extract_emosoundscape_features_from_file",
    }:
        from soundaqnet.feature_extraction import emosoundscape_features as mod

        return getattr(mod, name)
    if name in {"extract_loudness", "extract_loudness_from_file"}:
        from soundaqnet.feature_extraction import loudness as mod

        return getattr(mod, name)
    if name in {"extract_mel", "extract_mel_from_file"}:
        from soundaqnet.feature_extraction import mel_spectrogram as mod

        return getattr(mod, name)
    raise AttributeError(f"module 'soundaqnet.feature_extraction' has no attribute {name!r}")

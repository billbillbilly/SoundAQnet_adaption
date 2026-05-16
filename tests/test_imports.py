"""Smoke tests — verify the package structure imports cleanly."""

import importlib
import pytest


def test_package_version():
    """soundaqnet exposes a version string."""
    import soundaqnet
    assert hasattr(soundaqnet, "__version__")
    assert isinstance(soundaqnet.__version__, str)


def test_soundaqnet_class():
    """SoundAQnet model class is importable from the top-level package."""
    from soundaqnet import SoundAQnet
    assert SoundAQnet is not None


def test_config():
    """config module loads and exposes expected attributes."""
    from soundaqnet.framework import config
    assert hasattr(config, "event_labels")
    assert hasattr(config, "scene_labels")
    assert len(config.event_labels) == 15
    assert len(config.scene_labels) == 3


def test_feature_extraction_submodules():
    """All feature_extraction submodules are importable."""
    from soundaqnet.feature_extraction import loudness, mel_spectrogram
    from soundaqnet.feature_extraction import loudness_serial, loudness_parallel


def test_framework_submodules():
    """All framework submodules are importable."""
    mods = [
        "soundaqnet.framework.config",
        "soundaqnet.framework.utilities",
        "soundaqnet.framework.earlystop",
        "soundaqnet.framework.AutomaticWeightedLoss",
        "soundaqnet.framework.pytorch_utils",
        "soundaqnet.framework.gated_gcn_layer",
        "soundaqnet.framework.models_pytorch",
        "soundaqnet.framework.processing",
    ]
    for mod in mods:
        importlib.import_module(mod)


def test_inference_module():
    """soundaqnet.inference is importable and exposes main()."""
    from soundaqnet import inference
    assert callable(inference.main)


def test_prediction_module():
    """soundaqnet.prediction is importable and exposes main()."""
    from soundaqnet import prediction
    assert callable(prediction.main)
    assert callable(prediction.predictions_to_dataframe)

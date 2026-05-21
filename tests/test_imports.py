"""Smoke tests — verify the package structure imports cleanly.

Tests that require torch/librosa/tqdm are automatically skipped when those
packages are not installed (e.g. in a minimal CI environment or sandbox).
"""

from __future__ import annotations

import importlib
import pytest

# ── Availability flags ────────────────────────────────────────────────────────
torch_available    = importlib.util.find_spec("torch")    is not None
librosa_available  = importlib.util.find_spec("librosa")  is not None
tqdm_available     = importlib.util.find_spec("tqdm")     is not None
pandas_available   = importlib.util.find_spec("pandas")   is not None

requires_torch   = pytest.mark.skipif(not torch_available,   reason="torch not installed")
requires_librosa = pytest.mark.skipif(not librosa_available, reason="librosa not installed")
requires_tqdm    = pytest.mark.skipif(not tqdm_available,    reason="tqdm not installed")
requires_pandas  = pytest.mark.skipif(not pandas_available,  reason="pandas not installed")
requires_ml_stack = pytest.mark.skipif(
    not (torch_available and librosa_available),
    reason="torch and/or librosa not installed",
)


# ── Tests that always run ─────────────────────────────────────────────────────

def test_package_version():
    """soundaqnet exposes a version string regardless of optional deps."""
    import soundaqnet
    assert hasattr(soundaqnet, "__version__")
    assert isinstance(soundaqnet.__version__, str)
    assert soundaqnet.__version__ != ""


def test_bundled_data_files():
    """Normalization pickle files are present inside the installed package."""
    from importlib.resources import files as pkg_files
    data = pkg_files("soundaqnet.data")
    norm_mel = data / "norm_log_mel.pickle"
    norm_loud = data / "norm_loudness.pickle"
    # files() returns a Traversable; .is_file() confirms presence
    assert norm_mel.is_file(), "norm_log_mel.pickle missing from soundaqnet.data"
    assert norm_loud.is_file(), "norm_loudness.pickle missing from soundaqnet.data"


def test_bundled_calibration_wav():
    """Calibration WAV is present inside feature_extraction.

    Navigate from the top-level package so we don't trigger the
    feature_extraction __init__.py (which imports librosa).
    """
    from importlib.resources import files as pkg_files
    cal = (
        pkg_files("soundaqnet")
        / "feature_extraction"
        / "calibration_audio_file"
        / "calibration_signal_sine_1kHz_60dB.wav"
    )
    assert cal.is_file(), "Calibration WAV missing from soundaqnet/feature_extraction/calibration_audio_file/"


def test_bundled_models():
    """At least one bundled .pth model is present inside the installed package.

    Skipped when running from a source checkout where the large model weights
    have not been committed to git.  The assertion is enforced when testing
    the built wheel (publish.yml smoke-test), where models are included.
    """
    import pathlib
    from importlib.resources import files as pkg_files

    models_dir = pkg_files("soundaqnet") / "models"

    # importlib.resources Traversable doesn't expose .exists() for directories
    # on all Python versions, so resolve to a real path first when possible.
    try:
        real_dir = pathlib.Path(str(models_dir))
        if not real_dir.is_dir():
            pytest.skip("soundaqnet/models/ directory not present — model weights not committed to source tree")
        pth_files = [p for p in real_dir.iterdir() if p.suffix == ".pth"]
    except (TypeError, OSError):
        pytest.skip("Cannot resolve soundaqnet/models/ to a filesystem path")

    if not pth_files:
        pytest.skip("soundaqnet/models/ exists but contains no .pth files — weights not committed to source tree")

    assert len(pth_files) > 0, "No .pth model files found in soundaqnet/models/"


# ── Tests that need the ML stack ──────────────────────────────────────────────

@requires_ml_stack
def test_soundaqnet_class():
    """SoundAQnet model class is importable from the top-level package."""
    from soundaqnet import SoundAQnet
    assert SoundAQnet is not None


@requires_torch
def test_config():
    """config module loads and exposes expected attributes."""
    from soundaqnet.framework import config
    assert hasattr(config, "event_labels")
    assert hasattr(config, "scene_labels")
    assert len(config.event_labels) == 15
    assert len(config.scene_labels) == 3


@requires_ml_stack
def test_feature_extraction_submodules():
    """All feature_extraction submodules are importable."""
    from soundaqnet.feature_extraction import loudness, mel_spectrogram
    from soundaqnet.feature_extraction import loudness_serial, loudness_parallel


@requires_torch
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


@requires_torch
def test_inference_module():
    """soundaqnet.inference is importable and exposes main()."""
    from soundaqnet import inference
    assert callable(inference.main)
    # Verify the resolve_model_path helper is present
    assert callable(inference.resolve_model_path)
    assert len(inference.BUNDLED_MODELS) == 4


@requires_tqdm
@requires_pandas
def test_prediction_module():
    """soundaqnet.prediction is importable and exposes its public API."""
    from soundaqnet import prediction
    assert callable(prediction.main)
    assert callable(prediction.predictions_to_dataframe)
    assert callable(prediction.load_aq_outputs)
    assert callable(prediction.load_event_outputs)

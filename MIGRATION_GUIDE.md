# Migration Guide — Turning the scripts into a pip-installable package

This guide walks you through completing the package after the scaffold has
been generated.

---

## Step 0 — Understand the new layout

```
soundaqnet/                                   ← repo root
├── src/
│   └── soundaqnet/                           ← the importable Python package
│       ├── __init__.py                       ← version + compatibility checks
│       ├── _compat.py                        ← runtime version-checker (new)
│       ├── inference.py                      ← was application/Inference.py
│       ├── prediction.py                     ← was application/prediction_to_df.py
│       ├── feature_extraction/
│       │   ├── __init__.py
│       │   ├── loudness.py                   ← was ISO_loudness_fast.py (cross-platform)
│       │   ├── mel_spectrogram.py            ← was log_mel_spectrogram.py
│       │   ├── ISO_532_bin/
│       │   │   └── ISO_532-1.exe             ← copy from feature_extraction/ISO_532_bin/
│       │   └── calibration_audio_file/
│       │       └── calibration_signal_sine_1kHz_60dB.wav  ← copy from feature_extraction/calibration_audio_file/
│       └── framework/
│           ├── __init__.py
│           └── *.py                          ← all files from framework/ go here
├── tests/
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

---

## Step 1 — Copy your source files

### Python files

| Original | Destination |
|----------|-------------|
| `application/Inference.py` | `src/soundaqnet/inference.py` |
| `application/prediction_to_df.py` | `src/soundaqnet/prediction.py` |
| `feature_extraction/ISO_loudness_fast.py` | `src/soundaqnet/feature_extraction/loudness.py` |
| `feature_extraction/log_mel_spectrogram.py` | `src/soundaqnet/feature_extraction/mel_spectrogram.py` |
| `framework/*.py` | `src/soundaqnet/framework/` (copy all files) |

Each destination file already contains a `HOW TO MIGRATE` comment block
telling you exactly where to paste the code.

### Binary / audio assets

| Original | Destination | Used on |
|----------|-------------|---------|
| `feature_extraction/ISO_532_bin/ISO_532-1.exe` | `src/soundaqnet/feature_extraction/ISO_532_bin/ISO_532-1.exe` | Windows |
| `feature_extraction/calibration_audio_file/calibration_signal_sine_1kHz_60dB.wav` | `src/soundaqnet/feature_extraction/calibration_audio_file/calibration_signal_sine_1kHz_60dB.wav` | Windows |

Both are already declared in `pyproject.toml` as `package-data` so they get
included in the wheel automatically.  They are resolved at runtime using
`importlib.resources` — no path hard-coding needed.

---

## Step 2 — Fix cross-module imports

Replace all script-style path hacks with package imports:

| Old (script-style) | New (package-style) |
|--------------------|---------------------|
| `sys.path.insert(0, '../framework')` | *(delete)* |
| `from framework.xxx import yyy` | `from soundaqnet.framework.xxx import yyy` |
| `sys.path.insert(0, '../feature_extraction')` | *(delete)* |
| `from feature_extraction.xxx import yyy` | `from soundaqnet.feature_extraction.xxx import yyy` |
| Hard-coded `"ISO_532_bin/ISO_532-1.exe"` | `from soundaqnet.feature_extraction.loudness import _iso532_exe; exe = _iso532_exe()` |
| Hard-coded `"calibration_audio_file/..."` | `from soundaqnet.feature_extraction.loudness import _calibration_wav; wav = _calibration_wav()` |

You can also use relative imports within the package:
```python
# inside src/soundaqnet/inference.py
from .framework import SomeModelClass   # relative — fine for internal use
```

---

## Step 3 — Implement the cross-platform loudness back-ends

`loudness.py` has two back-end stubs that need to be filled in:

### `_run_windows()` — paste from `ISO_loudness_fast.py`

The original script calls `ISO_532-1.exe` as a subprocess.  Replace the
hard-coded binary path with `_iso532_exe()` and the calibration WAV path
with `_calibration_wav()`:

```python
import subprocess

def _run_windows(input_dir, output_dir, tmp_dir, num_workers, chunk_size):
    exe = str(_iso532_exe())
    cal = str(_calibration_wav())
    # … rest of the original subprocess call, adapted from ISO_loudness_fast.py
    subprocess.run([exe, cal, ...], check=True)
```

### `_run_posix()` — implement using mosqito

```python
from mosqito.sound_level_meter.comp_loudness import comp_loudness
import soundfile as sf
import numpy as np
import h5py

def _run_posix(input_dir, output_dir, num_workers, chunk_size):
    for wav_file in sorted(Path(input_dir).glob("*.wav")):
        signal, fs = sf.read(wav_file)
        # mosqito requires mono float32; resample to 48 kHz if needed
        N, N_spec, bark_axis, time_axis = comp_loudness(
            signal, fs=fs, field_type="free"
        )
        # save results to .h5 in the same format produced by ISO_532-1.exe
        out_path = Path(output_dir) / (wav_file.stem + "_loudness.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("loudness_sones", data=N)
            f.create_dataset("loudness_specific", data=N_spec)
            f.create_dataset("bark_axis", data=bark_axis)
            f.create_dataset("time_axis", data=time_axis)
```

> **Important:** make sure the `.h5` structure written by `_run_posix` exactly
> matches the structure written by the Windows `.exe`, because `Inference.py`
> reads both interchangeably.  Check what keys `ISO_loudness_fast.py` uses
> and mirror them.

---

## Step 4 — Handle model weights

The `.pth` files in `application/system/model/` are large and should not be
distributed via PyPI.  Two options:

### Option A — local path (simplest)
Users supply the path manually:
```python
model = SoundAQnet(model_path="path/to/SoundAQnet_ASC96_AEC94_PAQ1027.pth")
```

### Option B — download on first use (recommended)
Add a downloader in `inference.py`:
```python
import urllib.request
from pathlib import Path

_MODELS_DIR = Path.home() / ".soundaqnet" / "models"
_MODEL_URLS = {
    "SoundAQnet_ASC96_AEC94_PAQ1027": "https://github.com/.../releases/download/v0.1.0/SoundAQnet_ASC96_AEC94_PAQ1027.pth",
    # ... other models
}

def fetch_model(name: str) -> Path:
    dest = _MODELS_DIR / f"{name}.pth"
    if not dest.exists():
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URLS[name], dest)
    return dest
```
Upload the `.pth` files as [GitHub Release assets](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) first.

---

## Step 5 — Install and test

```bash
# from the repo root
pip install -e ".[dev]"

# verify imports and compatibility checks
python -c "import soundaqnet; print(soundaqnet.__version__)"

# run smoke tests
pytest tests/

# test CLI scripts
soundaqnet-extract-loudness --help
soundaqnet-extract-mel      --help
soundaqnet-infer            --help
soundaqnet-to-df            --help
```

---

## Step 6 — Publish to PyPI

```bash
# build the wheel and sdist
python -m build

# test upload to TestPyPI
twine upload --repository testpypi dist/*

# final upload to PyPI
twine upload dist/*
```

Store your API token in `~/.pypirc` or export `TWINE_PASSWORD` before running
`twine upload`.

---

## Dependency installation quick-reference

### CUDA 12.1 (Windows / Linux)
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install dgl --index-url https://data.dgl.ai/wheels/cu121/repo.html
pip install soundaqnet
```

### CUDA 11.8 (Windows / Linux)
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
pip install dgl --index-url https://data.dgl.ai/wheels/cu118/repo.html
pip install soundaqnet
```

### macOS (Apple Silicon — MPS)
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install dgl
pip install soundaqnet   # mosqito installed automatically
```

### CPU-only (any OS)
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install dgl
pip install soundaqnet
```

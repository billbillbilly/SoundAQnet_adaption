# soundaqnet

**Soundscape affective quality prediction via deep learning.**

[![PyPI version](https://img.shields.io/pypi/v/soundaqnet.svg)](https://pypi.org/project/soundaqnet/)
[![Python](https://img.shields.io/pypi/pyversions/soundaqnet.svg)](https://pypi.org/project/soundaqnet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/billbillbilly/SoundAQnet_adaption/actions/workflows/publish.yml/badge.svg?branch=package)](https://github.com/billbillbilly/SoundAQnet_adaption/actions)

`soundaqnet` is a pip-installable Python package that predicts soundscape perceptual
quality from audio clips using the **SoundAQnet** multi-task deep learning model (Hou et al., 2026).
Given a mono audio clip it outputs:

- **Acoustic scene** — `urban`, `suburban`, or `park`
- **Audio events** — probabilities for 15 event classes
- **ISO Pleasantness / Eventfulness** [−1,1]
- **PAQ 8-D affective quality** [1,5] — pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous

Works on **Windows**, **macOS**, and **Linux**.

> Adapted from [SoundSCaper](https://github.com/Yuanbo2020/SoundSCaper) — credits: Hou et al. (2026), IEEE Transactions on Multimedia.

---

## Installation

Install PyTorch **first** so the GPU/CPU variant is resolved correctly, then install `soundaqnet`.

### CPU (any platform)/ Apple Silicon (MPS)
```bash
pip install torch torchvision torchaudio
pip install soundaqnet
```

### GPU — CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install soundaqnet
```

> **macOS / Linux**: loudness extraction uses [mosqito](https://github.com/Eomys/MoSQITo) (installed automatically).  
> **Windows**: uses the bundled `ISO_532-1.exe` binary — no extra install required.

---

## Quick start

### Python API

```python
from soundaqnet import SoundAQnet
from soundaqnet.feature_extraction import (
    extract_mel_from_file,
    extract_loudness_from_file,
)

# Single file
mel  = extract_mel_from_file("my_clip.wav")       # → (T, 64) float32
loud = extract_loudness_from_file("my_clip.wav")  # → (T, 1)  float32

model  = SoundAQnet()                  # loads default bundled weights
result = model.predict_sample(mel, loud)

print(result["scene"])        # e.g. "park"
print(result["isop"])         # e.g.  0.62
print(result["top_events"])   # e.g. ["Bird", "Wind", "Natural sounds", ...]

# Batch from a directory
df = model.predict_from_audio(
    audio_dir="audio/",
    batch_size=32,
    num_workers=4,
)
print(df[["clip_id", "scene", "isop", "isoe"]].head())
```

### CLI

```bash
# Step 1 — extract features (parallel)
soundaqnet-extract-mel      --input_dir audio/ --output_dir mel/      --num_workers 4
soundaqnet-extract-loudness --input_dir audio/ --output_dir loudness/ --num_workers 4

# Step 2 — run inference
soundaqnet-infer --dataset_mel mel/ --dataset_wav_loudness loudness/

# Step 3 — convert to CSV
soundaqnet-to-df \
    --results_dir SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs \
    --events_dir  SoundAQnet_event_probability \
    --output      predictions.csv
```

---

## Feature extraction API

Both functions accept a **single path** or a **list of paths**, and save `.npy` files when `output_dir` is given.

```python
from soundaqnet.feature_extraction import (
    extract_mel_from_file, extract_loudness_from_file,
    extract_mel, extract_loudness,
)

# Single file → numpy array
mel  = extract_mel_from_file("clip.wav")                # (T, 64)
loud = extract_loudness_from_file("clip.wav")           # (T, 1)

# Multiple files → dict {stem: array}, processed in parallel
mels  = extract_mel_from_file(
    ["a.wav", "b.wav"], output_dir="mel/", num_workers=4)
louds = extract_loudness_from_file(
    ["a.wav", "b.wav"], output_dir="loudness/", num_workers=4)

# Batch directory — resumes automatically (skips completed files)
extract_mel("audio/",      "mel/",      num_workers=4)
extract_loudness("audio/", "loudness/", num_workers=4)
```

---

## Inference API

| Method | Input | Returns |
|---|---|---|
| `predict_sample(mel, loudness)` | numpy arrays `(T, 64)` and `(T, 1)` | `dict` |
| `predict(mel_dir, loudness_dir, batch_size)` | directories of `.npy` files | `pd.DataFrame` |
| `predict_from_audio(audio_dir, batch_size, num_workers)` | audio directory | `pd.DataFrame` |

### Output columns

| Column | Type | Description |
|---|---|---|
| `clip_id` | str | File stem (no extension) |
| `scene` | str | `"urban"` / `"suburban"` / `"park"` |
| `isop` | float | ISO Pleasantness −1 … +1 |
| `isoe` | float | ISO Eventfulness −1 … +1 |
| `pleasant` … `monotonous` | float | PAQ 8-D affective quality scores |
| `top_events` | list\[str\] | Top-5 audio event labels by probability |
| `event_probs` | dict | `{label: probability}` for all 15 event classes |

---

## Bundled models

Four pre-trained checkpoints are included.  Pass the short name to `SoundAQnet()`:

| Model name | ASC | AEC | PAQ F1 |
|---|---|---|---|
| `SoundAQnet_ASC96_AEC94_PAQ1027` **(default)** | 96 % | 94 % | 10.27 |
| `SoundAQnet_ASC96_AEC94_PAQ1039` | 96 % | 94 % | 10.39 |
| `SoundAQnet_ASC96_AEC94_PAQ1041` | 96 % | 94 % | 10.41 |
| `SoundAQnet_ASC96_AEC95_PAQ1052` | 96 % | 95 % | 10.52 |

```python
model = SoundAQnet("SoundAQnet_ASC96_AEC95_PAQ1052")
# or load your own fine-tuned checkpoint:
model = SoundAQnet("/path/to/my_finetuned.pth")
```

---

## Tutorials

Interactive Jupyter notebooks live in [`tutorials/`](tutorials/):

| Notebook | Contents |
|---|---|
| [`01_python_api.ipynb`](tutorials/01_python_api.ipynb) | Feature extraction, single-sample and batch inference, end-to-end API, circumplex plots |
| [`02_cli.ipynb`](tutorials/02_cli.ipynb) | Full CLI pipeline, output inspection, ready-to-run shell script |

---

## Platform notes

| Platform | Loudness backend | GPU |
|---|---|---|
| Windows | Bundled `ISO_532-1.exe` (MATLAB-compiled ISO 532-1) | CUDA |
| macOS Intel / Apple Silicon | mosqito (pure Python) | MPS |
| Linux | mosqito (pure Python) | CUDA |

Both backends produce numerically equivalent output using the same 1 kHz / 60 dB SPL calibration reference and the ISO 532-1 Zwicker time-varying method at 48 kHz.

---

## Development

```bash
git clone https://github.com/billbillbilly/SoundAQnet_adaption.git -b package
cd SoundAQnet_adaption
pip install -e ".[dev]"
pytest tests/
```

---

## Citation

```bibtex
@article{hou2026soundscape,
  title   = {Soundscape captioning using sound affective quality network
             and large language model},
  author  = {Hou, Yuanbo and Ren, Qiaoqiao and Mitchell, Andrew and
             Wang, Wenwu and Kang, Jian and Belpaeme, Tony and
             Botteldooren, Dick},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026}
}
```

---

## License

[MIT](LICENSE)

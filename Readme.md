# soundaqnet

**Soundscape affective quality prediction via deep learning.**

[![PyPI version](https://img.shields.io/pypi/v/soundaqnet.svg)](https://pypi.org/project/soundaqnet/)
[![Python](https://img.shields.io/pypi/pyversions/soundaqnet.svg)](https://pypi.org/project/soundaqnet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/billbillbilly/SoundAQnet_adaption/actions/workflows/publish.yml/badge.svg?branch=package)](https://github.com/billbillbilly/SoundAQnet_adaption/actions)

`soundaqnet` is a pip-installable Python package that predicts soundscape perceptual
quality from audio clips using machine learning models (Hou et al., 2026; Kranabetter et al., 2022).
It accepts common audio formats (`.wav`, `.mp3`, `.flac`, `.ogg`, `.aiff`, `.aif`,
`.m4a`, `.opus`) and outputs:

- **Acoustic scene** — `urban`, `suburban`, or `park`
- **Audio events** — probabilities for 15 event classes
- **ISO Pleasantness / Eventfulness** [−1,1]
- **Valence / Arousal** [−1,1]
- **PAQ 8-D affective quality** [1,5] — pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous

Works on **Windows**, **macOS**, and **Linux**.

> Adapted from [SoundSCaper](https://github.com/Yuanbo2020/SoundSCaper) — credits: Hou et al. (2026), IEEE Transactions on Multimedia.

---

## 1 Installation

Requires **Python 3.10+**. `soundaqnet` installs the audio, data, and
Emo-Soundscape dependencies it needs, including `librosa`, `soundfile`,
`scikit-learn`, `torchlibrosa`, and `mosqito` on macOS/Linux.

For GPU acceleration, install the matching PyTorch build **before**
`soundaqnet`; otherwise pip will install the default CPU-compatible PyTorch
packages from PyPI.

### CPU / Apple Silicon (MPS)
```bash
pip install torch torchvision torchaudio
pip install soundaqnet
```

### NVIDIA GPU — CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install soundaqnet
```

### From source
```bash
git clone https://github.com/billbillbilly/SoundAQnet_adaption.git
cd SoundAQnet_adaption
pip install -e ".[dev]"
```

Notes:

- **Windows** loudness extraction uses the bundled `ISO_532-1.exe`.
- **macOS/Linux** loudness extraction uses `mosqito`, installed automatically.
- **NumPy 2.x is not supported** by the current audio stack, so the package pins `numpy<2`.

---

## 2 Quick start

### 2.1 SoundAQnet

Predict acoustic scene, audio-event probabilities, ISO Pleasantness/Eventfulness,
and PAQ 8-D affective quality.

| Column | Type | Description |
|---|---|---|
| `clip_id` | str | File stem (no extension) |
| `scene` | str | `"urban"` / `"suburban"` / `"park"` |
| `isop` | float | ISO Pleasantness −1 … +1 |
| `isoe` | float | ISO Eventfulness −1 … +1 |
| `pleasant` … `monotonous` | float | PAQ 8-D affective quality scores |
| `top_events` | list\[str\] | Top-5 audio event labels by probability |
| `event_probs` | dict | `{label: probability}` for all 15 event classes |

#### Python API

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

#### CLI

```bash
# Step 1 — extract features (parallel)
soundaqnet-extract-mel      --input_dir audio/ --output_dir mel/      --num_workers 4
soundaqnet-extract-loudness --input_dir audio/ --output_dir loudness/ --num_workers 4

# Step 2 — run inference
soundaqnet-infer \
    --dataset_mel mel/ \
    --dataset_wav_loudness loudness/ \
    --batch_size 32

# Step 3 — convert to CSV
soundaqnet-to-df \
    --paq_dir SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs \
    --event_dir SoundAQnet_event_probability \
    --output_prefix soundAQ
```

The converter writes `soundAQ.csv` (including `scene`, ISO, and PAQ columns),
`soundAQ_stats.csv`, and `soundAQEventRank.csv` depending on `--export`.

### 2.2 EmoSoundscape

Predict valence and arousal from audio using the bundled
`EmoS_gradient_boosting` model.

The 122-feature vector summarizes RMS/energy, zero-crossing rate, spectral
shape, MFCCs, chroma, and log-mel bands using frame-level mean and standard
deviation. The extractor uses 44.1 kHz mono audio, 4096-sample frames, and a
2048-sample hop.

| Column | Type | Description |
|---|---|---|
| `clip_id` | str | File stem (no extension) |
| `valence` | float | valence −1 … +1 |
| `arousal` | float | arousal −1 … +1 |

#### Python API

```python
from soundaqnet import EmoSoundscape

model = EmoSoundscape()

# Single file or explicit list
df = model.predict_from_audio(audio_files=["clip.wav"])
print(df[["clip_id", "valence", "arousal"]])

# Batch from a directory
df = model.predict_from_audio(audio_dir="audio/")

# Feature extraction
from soundaqnet.feature_extraction import (
    extract_emosoundscape_features,
    extract_emosoundscape_features_from_file,
)
# Single file -> numpy array, shape (122,)
features = extract_emosoundscape_features_from_file("clip.wav")
result = model.predict_sample(features)
# Batch directory -> DataFrame, optionally saved to CSV
feature_df = extract_emosoundscape_features(
    audio_dir="audio/",
    output_csv="emosoundscape_features.csv",
)
```

#### CLI

```bash
soundaqnet-emosoundscape \
    --audio_file clip.wav \
    --output_csv emosoundscape_predictions.csv
```

---

## 3 Tutorials

Interactive Jupyter notebooks live in [`tutorials/`](tutorials/):

| Notebook | Contents |
|---|---|
| [`01_python_api.ipynb`](tutorials/01_python_api.ipynb) | SoundAQnet feature extraction/inference, EmoSoundscape valence/arousal, bundled models, circumplex plots |
| [`02_cli.ipynb`](tutorials/02_cli.ipynb) | SoundAQnet CLI pipeline, EmoSoundscape CLI inference, output inspection, ready-to-run shell script |

---

## 4 Bundled models

Bundled model names can be passed directly to `SoundAQnet(...)` or
`EmoSoundscape(...)`.

### 4.1 SoundAQnet checkpoints

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

### 4.2 EmoSoundscape checkpoint

| Model name | Input | Intended use | 10-fold CV R2 |
|---|---|---|---|
| `EmoS_gradient_boosting` **(default)** | 122 package-native features from audio | New audio inference | mean `0.796`, valence `0.696`, arousal `0.895` |

```python
from soundaqnet import EmoSoundscape

model = EmoSoundscape("EmoS_gradient_boosting")
# or load your own model trained on package-native features:
model = EmoSoundscape("/path/to/my_model.pkl")
```

---


## 5 Method note

### Clip length

The Emo-Soundscapes dataset contains 6-second excerpts. The extractor can
summarize audio of any length, but predictions are most trustworthy for clips
near that duration and with a consistent soundscape. For longer recordings,
split into 6-second windows, predict each window, then average the predictions
or keep the window-level valence/arousal time series.

### Default Emo-Soundscape model

Use `EmoS_gradient_boosting` for real audio files. You can also load your own
compatible model with `EmoSoundscape("/path/to/model.pkl")`.

---


## 6 Platform notes

| Platform | Loudness backend | GPU |
|---|---|---|
| Windows | Bundled `ISO_532-1.exe` (MATLAB-compiled ISO 532-1) | CUDA / CPU |
| macOS Intel / Apple Silicon | mosqito (pure Python) | MPS / CPU |
| Linux | mosqito (pure Python) | CUDA / CPU |

Both backends produce numerically equivalent output using the same 1 kHz / 60 dB SPL calibration reference and the ISO 532-1 Zwicker time-varying method at 48 kHz.

---

## Development

```bash
git clone https://github.com/billbillbilly/SoundAQnet_adaption.git -b package
cd SoundAQnet_adaption
pip install -e ".[package]"
pytest tests/
```

---

## Citation

If you use the SoundAQnet model, cite Hou et al. If you use the
Emo-Soundscape valence/arousal module or trained Emo-Soundscapes models, also
cite the Emo-Soundscapes dataset, soundscape emotion recognition, and
Audio Metaphor 2.0 papers.
The bundled Emo-Soundscape reference PDFs are in `docs/emosoundscape/`.

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

@inproceedings{fan2017emosoundscapes,
  title     = {Emo-Soundscapes: A Dataset for Soundscape Emotion Recognition},
  author    = {Fan, Jianyu and Thorogood, Miles and Pasquier, Philippe},
  booktitle = {Proceedings of the International Conference on Affective
               Computing and Intelligent Interaction},
  year      = {2017}
}

@inproceedings{fan2018soundscape,
  title  = {SOUNDSCAPE EMOTION RECOGNITION VIA DEEP LEARNING},
  author = {Fan, Jianyu and Tung, Fred and Li, William and Pasquier, Philippe},
  year   = {2018}
}

@misc{kranabetter2022audiometaphor,
  title  = {Audio Metaphor 2.0: An Improved Classification and Segmentation
            Pipeline for Generative Sound Design Systems},
  author = {Kranabetter, Joshua and Carpenter, Craig and Tchemeube, Renaud
            Bougueng and Pasquier, Philippe and Thorogood, Miles},
  year   = {2022},
  note   = {Bundled reference PDF: docs/emosoundscape/AudioMetaphor2.0.pdf}
}
```

---

## License

[MIT](LICENSE)

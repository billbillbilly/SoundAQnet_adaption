# Inferring affective quality from soundscape clips (SoundAQnet)

SoundAQnet aims to convert the soundscape audio clips (**only files containing one channel are allowed**) into the predicted audio event probabilities, the acoustic scene labels, and the ISOP, ISOE, and PAQ 8D AQ values.

**Note**: 
- This repo is copied and adapted from a part of [SoundSCaper](https://github.com/Yuanbo2020/SoundSCaper). The credits goes to Hou et. al (2026) 

- Since the data processing needs to run an external windows application (ISO_532-1.exe - for loundness extraction), please use this repo on Windows computers

## Installation
Before starting installation, please install conda and setup an environment:
```sh
conda create -n soundqa python=3.10
conda activate soundaq
```

After installing conda:
```sh
cd SoundAQnet
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl --index-url https://data.dgl.ai/wheels/cu121/repo.html
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```
## Usage
### 1. Data preparation
Extrat log mel features from clips using: 
```sh
cd feature_extraction
python ISO_loudness_fast.py --input_dir $input_dir --output_dir $output_dir --tmp_dir $temp --num_workers $num --chunk_size $size
python log_mel_spectrogram.py --input_dir $input_dir --output_dir $output_dir
cd ..
```

The log Mel features and ISO 532-1 loudness features file will be used as input in the inference model.
 
### 2. Run the inference to get soundscape predictions

```sh
cd application
python Inference.py --dataset_mel $mel_spectrogram --dataset_wav_loudness $ISO_loudness --model system/model/SoundAQnet_ASC96_AEC94_PAQ1027.pth
```

- There are four slightly different SoundAQnet models in the `system/model` directory:
	- SoundAQnet_ASC96_AEC94_PAQ1027.pth
	- SoundAQnet_ASC96_AEC94_PAQ1039.pth
	- SoundAQnet_ASC96_AEC94_PAQ1041.pth
	- SoundAQnet_ASC96_AEC95_PAQ1052.pth

The inference results including sound event, scene, and affective quality will be in:
- application/SoundAQnet_event_probability
- application/SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs

### 3. Convert prediction to dataframe
```sh
python prediction_to_df.py
cd ..
```

## Reference
```
Hou, Y., Ren, Q., Mitchell, A., Wang, W., Kang, J., Belpaeme, T., & Botteldooren, D. (2026). Soundscape captioning using sound affective quality network and large language model. IEEE Transactions on Multimedia.
```
import pickle
import sys, os, h5py, time, librosa, torch
import numpy as np
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import argparse
from tqdm import tqdm


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd, exist_ok=True)


def listFnames(dirName, ext=''):
    fnames = []
    for (rootDir, dirList, filesList) in os.walk(dirName):
        fnames += [os.path.join(rootDir, f) for f in filesList if f.endswith(ext)]
    return fnames


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0:audio_length]


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def save_pickle(file, dict):
    with open(file, 'wb') as f:
        pickle.dump(dict, f)


def run_jobs(input_dir, output_dir):
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
        freeze_parameters=True
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
        freeze_parameters=True
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    spectrogram_extractor.to(device)
    logmel_extractor.to(device)

    audio_files = listFnames(input_dir, '.wav')
    print(f"Found {len(audio_files)} audio files")

    audio_files = [f for f in audio_files
                   if not os.path.exists(
                       os.path.join(output_dir, os.path.basename(f).replace('.wav', '.npy'))
                   )]
    print(f"Skipping already-complete files; {len(audio_files)} remaining")

    for n, audio_path in tqdm(enumerate(audio_files)):
        audioname = os.path.basename(audio_path)
        feature_filename = audioname.replace('.wav', '.npy')
        output_feature = os.path.join(output_dir, feature_filename)

        audiodata, fs = librosa.core.load(audio_path, sr=sample_rate, mono=True)

        spectrogram = spectrogram_extractor(
            move_data_to_device(audiodata[None, :], device)
        )

        logmel = logmel_extractor(spectrogram)
        logmel = logmel[0, 0].data.cpu().numpy()

        # print(n, feature_filename, logmel.shape)

        np.save(output_feature, logmel)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="The directory of the wav files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Dataset_mel",
        help="The directory of the output mel feature files."
    )
    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir)
    output_dir = args.output_dir

    run_jobs(input_dir, output_dir)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
import os, sys
import time as tm
import subprocess as sbp
import pandas as pd
import numpy as np
import warnings
import wave
import soundfile
from pathlib import Path
import argparse
from tqdm import tqdm

warnings.filterwarnings("ignore")


def cleanScreen():
    p = sys.platform
    if "win" in p:
        os.system("cls")
    elif "linux" in p:
        os.system("clear")


def createDirs(dname):
    if not os.path.isdir(dname):
        os.makedirs(dname)


def listFnames(dirName, ext=""):
    fnames = []
    for (rootDir, dirList, filesList) in os.walk(dirName):
        fnames += [os.path.join(rootDir, f) for f in filesList if f.endswith(ext)]
    return fnames


def is_mono_wav(filename):
    with wave.open(filename, "rb") as wf:
        return wf.getnchannels() == 1


def prepare_iso_input(src_path, out_dir="pcm16", target_sr=48000):
    """
    Create a WAV file suitable for ISO_532-1.exe:
    - mono
    - PCM_16
    - sample rate in {32000, 44100, 48000}
    """
    allowed_srs = {32000, 44100, 48000}
    if target_sr not in allowed_srs:
        raise ValueError(f"target_sr must be one of {sorted(allowed_srs)}")

    src_path = Path(src_path)

    if out_dir is None:
        out_dir = src_path.parent
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Read audio
    x, sr = soundfile.read(src_path, always_2d=True)

    # x shape: (samples, channels)
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to mono by averaging channels
    if x.shape[1] > 1:
        # print("Convert the audio to mono")
        x = x.mean(axis=1)
    else:
        x = x[:, 0]

    # Resample if needed
    if sr != target_sr:
        # print(f"Resample audio from {sr} Hz to {target_sr} Hz")
        duration = len(x) / sr
        old_t = np.linspace(0, duration, num=len(x), endpoint=False)
        new_len = int(round(duration * target_sr))
        new_t = np.linspace(0, duration, num=new_len, endpoint=False)
        x = np.interp(new_t, old_t, x).astype(np.float32)
        sr = target_sr

    # Clip before int16 writing
    x = np.clip(x, -1.0, 1.0)

    out_path = out_dir / f"{src_path.stem}_iso.wav"
    soundfile.write(out_path, x, sr, subtype="PCM_16")
    return str(out_path)


def runProcess(loudnessExe, method, soundField, audioFile, refFile, refLevel, module="subprocess"):
    # print(" |- running ISO_532-1 loudness calculation")

    if module == "os":
        command = f'"{loudnessExe}" {method} {soundField} "{audioFile}" "{refFile}" {int(refLevel)}'
        # print(" |- CMD:", command)
        ret = os.system(command)
        if ret != 0:
            raise RuntimeError(f"ISO_532-1 command failed with exit code {ret}: {command}")
        return ""

    elif module == "subprocess":
        comList = [loudnessExe, method, soundField, audioFile, refFile, str(int(refLevel))]
        # print(" |- CMD:", " ".join([f'"{c}"' if " " in c else c for c in comList]))

        proc = sbp.run(
            comList,
            stdout=sbp.PIPE,
            stderr=sbp.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")

        lowered = output.lower()
        if proc.returncode != 0 or "error" in lowered or "nan" in lowered:
            print(f" |- ERROR (exit {proc.returncode}):\n{output}")
            raise RuntimeError(output.strip() or f"ISO_532-1 failed with exit code {proc.returncode}")

        return output

    else:
        raise NotImplementedError(f"Unknown module={module!r}")


def getData():
    # print(" |- getting loudness features")
    dfLoudness = pd.read_csv("Loudness.csv", sep=";", skiprows=5, index_col=0)
    loud = dfLoudness.to_numpy()
    return loud


def saveData(fileDir, loud):
    # print(" |- saving loudness features")
    np.save(fileDir, loud)


def removeTemp():
    # print(" |- removing temporary csv files")
    if os.path.exists("Loudness.csv"):
        os.remove("Loudness.csv")
        # print("    |- removing Loudness.csv")
    if os.path.exists("SpecLoudness.csv"):
        os.remove("SpecLoudness.csv")
        # print("    |- removing SpecLoudness.csv")


if __name__ == "__main__":
    cleanScreen()

    METHOD_DICT = {"Varying": "Time_varying", "Stationary": "Stationary"}
    SF_DICT = {"Free": "F", "Diffuse": "D"}

    LOUDNESS_BINARY = os.path.join(os.getcwd(), "ISO_532_bin", "ISO_532-1.exe")

    METHOD = "Varying"
    SOUND_FIELD = "Free"

    CAL_FILE = os.path.join(
        os.getcwd(),
        "calibration_audio_file",
        "calibration_signal_sine_1kHz_60dB.wav",
    )
    RMS_DB = 60

    exe = LOUDNESS_BINARY
    meth = METHOD_DICT[METHOD]
    sf = SF_DICT[SOUND_FIELD]
    rfile = CAL_FILE
    rlev = RMS_DB

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="The directory of the wav files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Dataset_wav_loudness",
        help="The directory of the output loudness files.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=48000,
        choices=[32000, 44100, 48000],
        help="Target sample rate for ISO_532-1 input wav.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    target_sr = args.target_sr

    createDirs(output_dir)
    createDirs("pcm16")

    audioFiles = listFnames(input_dir, "wav")
    print(f"Find {len(audioFiles)} audio files")

    for audioFile in tqdm(audioFiles, total=len(audioFiles)):
        audioName = os.path.basename(audioFile).split(".w")[0]
        fileDir = os.path.join(output_dir, audioName + ".npy")

        # print(f'Processing the audio clip: "{audioFile}"')
        st = tm.perf_counter()

        try:
            audio_iso = prepare_iso_input(audioFile, out_dir="pcm16", target_sr=target_sr)
            runProcess(exe, meth, sf, audio_iso, rfile, rlev)
            loud = getData()
            saveData(fileDir, loud)
            removeTemp()
        except Exception as e:
            print(f"Skip {audioFile} due to error: {e}")
            continue

        et = tm.perf_counter()
        # print("Loudness extraction is done in {:.3f} seconds.".format(et - st))
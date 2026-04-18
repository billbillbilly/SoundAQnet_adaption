import os
import sys
import argparse
import subprocess as sbp
import warnings
import shutil
import tempfile
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
import soundfile
from tqdm import tqdm

warnings.filterwarnings("ignore")


def createDirs(dname):
    if not os.path.isdir(dname):
        os.makedirs(dname, exist_ok=True)


def listFnames(dirName, ext=""):
    fnames = []
    for rootDir, dirList, filesList in os.walk(dirName):
        fnames += [os.path.join(rootDir, f) for f in filesList if f.endswith(ext)]
    return fnames


def prepare_iso_input(src_path, out_dir, target_sr=48000):
    """
    Create a WAV file suitable for ISO_532-1.exe:
    - mono
    - PCM_16
    - sample rate in {32000, 44100, 48000}
    """
    allowed_srs = {32000, 44100, 48000}
    if target_sr not in allowed_srs:
        raise ValueError(f"target_sr must be one of {sorted(allowed_srs)}")

    src_path = Path(src_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x, sr = soundfile.read(str(src_path), always_2d=True)

    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to mono
    if x.shape[1] > 1:
        x = x.mean(axis=1)
    else:
        x = x[:, 0]

    # Resample if needed
    if sr != target_sr:
        duration = len(x) / sr
        old_t = np.linspace(0, duration, num=len(x), endpoint=False)
        new_len = int(round(duration * target_sr))
        new_t = np.linspace(0, duration, num=new_len, endpoint=False)
        x = np.interp(new_t, old_t, x).astype(np.float32)
        sr = target_sr

    x = np.clip(x, -1.0, 1.0)

    out_path = out_dir / f"{src_path.stem}_iso.wav"
    soundfile.write(str(out_path), x, sr, subtype="PCM_16")
    return str(out_path.resolve())


def runProcess(loudnessExe, method, soundField, audioFile, refFile, refLevel, work_dir):
    """
    Run ISO_532-1.exe inside a worker-specific temp directory so that
    Loudness.csv / SpecLoudness.csv do not collide across workers.
    """
    loudnessExe = os.path.abspath(loudnessExe)
    audioFile = os.path.abspath(audioFile)
    refFile = os.path.abspath(refFile)
    work_dir = os.path.abspath(work_dir)

    comList = [loudnessExe, method, soundField, audioFile, refFile, str(int(refLevel))]

    proc = sbp.run(
        comList,
        stdout=sbp.PIPE,
        stderr=sbp.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=work_dir,
        shell=False,
    )

    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    lowered = output.lower()

    if proc.returncode != 0 or "error" in lowered or "nan" in lowered:
        raise RuntimeError(output.strip() or f"ISO_532-1 failed with exit code {proc.returncode}")

    return output


def getData(csv_path):
    csv_path = os.path.abspath(csv_path)
    dfLoudness = pd.read_csv(csv_path, sep=";", skiprows=5, index_col=0)
    loud = dfLoudness.to_numpy()
    return loud


def audio_to_output_path(audioFile, output_dir):
    audioName = Path(audioFile).stem
    return os.path.join(output_dir, audioName + ".npy")


def process_one(task):
    """
    Worker-safe processing of a single audio clip.
    Must be top-level for Windows multiprocessing.
    """
    (
        audioFile,
        output_dir,
        tmp_root,
        exe,
        meth,
        sf,
        rfile,
        rlev,
        target_sr,
        overwrite,
        debug_paths,
    ) = task

    audioFile = os.path.abspath(audioFile)
    output_dir = os.path.abspath(output_dir)
    tmp_root = os.path.abspath(tmp_root)
    exe = os.path.abspath(exe)
    rfile = os.path.abspath(rfile)

    out_npy = audio_to_output_path(audioFile, output_dir)

    if (not overwrite) and os.path.exists(out_npy):
        return {
            "status": "skipped",
            "audioFile": audioFile,
            "output": out_npy,
            "error": "",
        }

    worker_tmp = tempfile.mkdtemp(prefix="iso_", dir=tmp_root)

    try:
        audio_iso = prepare_iso_input(audioFile, out_dir=worker_tmp, target_sr=target_sr)

        if debug_paths:
            for p, label in [
                (audioFile, "audioFile"),
                (audio_iso, "audio_iso"),
                (exe, "exe"),
                (rfile, "rfile"),
                (worker_tmp, "worker_tmp"),
            ]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{label} not found: {p}")

        runProcess(exe, meth, sf, audio_iso, rfile, rlev, worker_tmp)

        loudness_csv = os.path.join(worker_tmp, "Loudness.csv")
        if not os.path.exists(loudness_csv):
            raise RuntimeError(f"Loudness.csv not found for {audioFile}")

        loud = getData(loudness_csv)
        np.save(out_npy, loud)

        return {
            "status": "ok",
            "audioFile": audioFile,
            "output": out_npy,
            "error": "",
        }

    except Exception as e:
        return {
            "status": "error",
            "audioFile": audioFile,
            "output": out_npy,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }

    finally:
        shutil.rmtree(worker_tmp, ignore_errors=True)


def main():
    METHOD_DICT = {"Varying": "Time_varying", "Stationary": "Stationary"}
    SF_DICT = {"Free": "F", "Diffuse": "D"}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input wav files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Dataset_wav_loudness",
        help="Directory for output loudness .npy files.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=48000,
        choices=[32000, 44100, 48000],
        help="Target sample rate for ISO_532-1 input wav.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index of files to process after sorting.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index (exclusive) of files to process after sorting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if .npy already exists.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="Varying",
        choices=["Varying", "Stationary"],
        help="Loudness calculation type.",
    )
    parser.add_argument(
        "--sound_field",
        type=str,
        default="Free",
        choices=["Free", "Diffuse"],
        help="Sound field type.",
    )
    parser.add_argument(
        "--debug_paths",
        action="store_true",
        help="Check that exe, calibration file, temp wav, and worker tmp directory exist before running the exe.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    exe = os.path.abspath(os.path.join(script_dir, "ISO_532_bin", "ISO_532-1.exe"))
    rfile = os.path.abspath(
        os.path.join(
            script_dir,
            "calibration_audio_file",
            "calibration_signal_sine_1kHz_60dB.wav",
        )
    )
    rlev = 60
    meth = METHOD_DICT[args.method]
    sf = SF_DICT[args.sound_field]

    if not os.path.exists(exe):
        raise FileNotFoundError(f"ISO executable not found: {exe}")
    if not os.path.exists(rfile):
        raise FileNotFoundError(f"Calibration file not found: {rfile}")

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    tmp_root = os.path.abspath(os.path.join(output_dir, "_tmp"))

    createDirs(output_dir)
    createDirs(tmp_root)

    audioFiles = sorted(listFnames(input_dir, ".wav"))
    total_found = len(audioFiles)
    print(f"Found {total_found} wav files in: {input_dir}")

    start_idx = max(0, args.start_idx)
    end_idx = args.end_idx if args.end_idx is not None else total_found
    end_idx = min(end_idx, total_found)

    if start_idx >= end_idx:
        print(f"Nothing to do: start_idx={start_idx}, end_idx={end_idx}, total_found={total_found}")
        return 0

    audioFiles = audioFiles[start_idx:end_idx]
    print(f"Processing slice [{start_idx}:{end_idx}] -> {len(audioFiles)} files")
    print(f"Output dir: {output_dir}")
    print(f"Temp root: {tmp_root}")
    print(f"Workers: {args.num_workers}")
    print(f"Target sample rate: {args.target_sr}")

    tasks = [
        (
            audioFile,
            output_dir,
            tmp_root,
            exe,
            meth,
            sf,
            rfile,
            rlev,
            args.target_sr,
            args.overwrite,
            args.debug_paths,
        )
        for audioFile in audioFiles
    ]

    ok_count = 0
    skip_count = 0
    err_count = 0
    error_log_path = os.path.join(output_dir, "errors.log")

    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write(
            f"ISO loudness parallel run\n"
            f"input_dir={input_dir}\n"
            f"output_dir={output_dir}\n"
            f"start_idx={start_idx}\n"
            f"end_idx={end_idx}\n"
            f"num_workers={args.num_workers}\n\n"
        )

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()

            if result["status"] == "ok":
                ok_count += 1
            elif result["status"] == "skipped":
                skip_count += 1
            else:
                err_count += 1
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"FILE: {result['audioFile']}\n")
                    f.write(result["error"])
                    f.write("\n" + "=" * 80 + "\n")

    shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nDone.")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {err_count}")
    print(f"Error log: {error_log_path}")

    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
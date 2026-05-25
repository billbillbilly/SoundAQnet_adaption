import os
import sys
import argparse
import subprocess as sbp
import warnings
import shutil
import tempfile
import threading
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Per-thread state: each worker thread gets its own persistent tmp directory
# (item 2). The ISO EXE always writes Loudness.csv / SpecLoudness.csv into its
# cwd, so as long as no two threads share a cwd, they can't clobber each other.
# ──────────────────────────────────────────────────────────────────────────────
_thread_local = threading.local()
_worker_dirs = []           # for cleanup at the end
_worker_dirs_lock = threading.Lock()


def _get_worker_dir(tmp_root):
    d = getattr(_thread_local, "dir", None)
    if d is None:
        d = tempfile.mkdtemp(prefix="iso_w_", dir=tmp_root)
        _thread_local.dir = d
        with _worker_dirs_lock:
            _worker_dirs.append(d)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def createDirs(dname):
    os.makedirs(dname, exist_ok=True)


def listFnames(dirName, ext=""):
    fnames = []
    for rootDir, _, filesList in os.walk(dirName):
        fnames += [os.path.join(rootDir, f) for f in filesList if f.endswith(ext)]
    return fnames


def audio_to_output_path(audioFile, output_dir):
    return os.path.join(output_dir, Path(audioFile).stem + ".npy")


# ──────────────────────────────────────────────────────────────────────────────
# Conformance check (item 3): if the source is already mono / PCM_16 / allowed
# sample rate, skip the rewrite entirely and feed the EXE the original path.
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_SRS = {32000, 44100, 48000}


def is_iso_conformant(src_path, target_sr):
    try:
        info = soundfile.info(str(src_path))
    except Exception:
        return False
    return (
        info.channels == 1
        and info.subtype == "PCM_16"
        and info.samplerate == target_sr
        and info.samplerate in ALLOWED_SRS
    )


def prepare_iso_input(src_path, out_dir, target_sr=48000, reuse_name="input_iso.wav"):
    """
    Write a mono / PCM_16 / target_sr WAV into `out_dir` using a fixed filename
    so each worker thread reuses the same path on every call (no mkdtemp churn).
    """
    if target_sr not in ALLOWED_SRS:
        raise ValueError(f"target_sr must be one of {sorted(ALLOWED_SRS)}")

    src_path = Path(src_path).resolve()
    out_dir = Path(out_dir).resolve()

    x, sr = soundfile.read(str(src_path), always_2d=True)
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.shape[1] > 1:
        x = x.mean(axis=1)
    else:
        x = x[:, 0]

    if sr != target_sr:
        duration = len(x) / sr
        old_t = np.linspace(0, duration, num=len(x), endpoint=False)
        new_len = int(round(duration * target_sr))
        new_t = np.linspace(0, duration, num=new_len, endpoint=False)
        x = np.interp(new_t, old_t, x).astype(np.float32)
        sr = target_sr

    np.clip(x, -1.0, 1.0, out=x)

    out_path = out_dir / reuse_name
    soundfile.write(str(out_path), x, sr, subtype="PCM_16")
    return str(out_path)


def runProcess(loudnessExe, method, soundField, audioFile, refFile, refLevel, work_dir):
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


# ──────────────────────────────────────────────────────────────────────────────
# Fast CSV reader (item 6). Loudness.csv has 5 header rows + 1 column-name row;
# original code used skiprows=5 with index_col=0 (drop col 0). We mirror that.
# ──────────────────────────────────────────────────────────────────────────────
def getData(csv_path):
    # skiprows=6 here = the same 5 metadata lines + 1 column-header line that
    # pandas was consuming via skiprows=5 + header inference.
    arr = np.loadtxt(csv_path, delimiter=";", skiprows=6)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 1:]  # drop the time/index column, matches index_col=0


# ──────────────────────────────────────────────────────────────────────────────
# Per-file work
# ──────────────────────────────────────────────────────────────────────────────
def process_one(task):
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
        debug_paths,
    ) = task

    out_npy = audio_to_output_path(audioFile, output_dir)
    worker_tmp = _get_worker_dir(tmp_root)

    try:
        # Item 3: skip rewrite when source is already conformant.
        if is_iso_conformant(audioFile, target_sr):
            audio_iso = os.path.abspath(audioFile)
        else:
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

        return {"status": "ok", "audioFile": audioFile, "output": out_npy, "error": ""}

    except Exception as e:
        return {
            "status": "error",
            "audioFile": audioFile,
            "output": out_npy,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    METHOD_DICT = {"Varying": "Time_varying", "Stationary": "Stationary"}
    SF_DICT = {"Free": "F", "Diffuse": "D"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Dataset_wav_loudness")
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="Where worker temp dirs live. Point this at a RAM disk (e.g. R:\\iso_tmp) "
             "for a large speedup. Defaults to <output_dir>/_tmp.",
    )
    parser.add_argument("--target_sr", type=int, default=48000, choices=[32000, 44100, 48000])
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel worker threads. Try cpu_count to 1.5x cpu_count.",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method", type=str, default="Varying", choices=["Varying", "Stationary"])
    parser.add_argument("--sound_field", type=str, default="Free", choices=["Free", "Diffuse"])
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Submit work in chunks of this size to bound memory (item 7).",
    )
    parser.add_argument("--debug_paths", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    exe = os.path.abspath(os.path.join(script_dir, "ISO_532_bin", "ISO_532-1.exe"))
    rfile = os.path.abspath(
        os.path.join(script_dir, "calibration_audio_file", "calibration_signal_sine_1kHz_60dB.wav")
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
    tmp_root = os.path.abspath(args.tmp_dir) if args.tmp_dir else os.path.abspath(
        os.path.join(output_dir, "_tmp")
    )

    createDirs(output_dir)
    createDirs(tmp_root)

    audioFiles = sorted(listFnames(input_dir, ".wav"))
    total_found = len(audioFiles)
    print(f"Found {total_found} wav files in: {input_dir}")

    start_idx = max(0, args.start_idx)
    end_idx = args.end_idx if args.end_idx is not None else total_found
    end_idx = min(end_idx, total_found)
    if start_idx >= end_idx:
        print(f"Nothing to do: start_idx={start_idx}, end_idx={end_idx}")
        return 0

    audioFiles = audioFiles[start_idx:end_idx]
    print(f"Processing slice [{start_idx}:{end_idx}] -> {len(audioFiles)} files")

    # Item 8: one scandir instead of 800k stat calls.
    skip_count = 0
    if not args.overwrite:
        existing = {
            Path(e.name).stem
            for e in os.scandir(output_dir)
            if e.is_file() and e.name.endswith(".npy")
        }
        before = len(audioFiles)
        audioFiles = [f for f in audioFiles if Path(f).stem not in existing]
        skip_count = before - len(audioFiles)
        print(f"Skipping {skip_count} already-complete file(s); {len(audioFiles)} remaining")

    print(f"Output dir: {output_dir}")
    print(f"Temp root:  {tmp_root}  (RAM disk recommended)")
    print(f"Workers:    {args.num_workers} (threads)")
    print(f"Target SR:  {args.target_sr}")

    error_log_path = os.path.join(output_dir, "errors.log")
    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write(
            f"ISO loudness run\n"
            f"input_dir={input_dir}\noutput_dir={output_dir}\ntmp_root={tmp_root}\n"
            f"start_idx={start_idx}\nend_idx={end_idx}\nworkers={args.num_workers}\n\n"
        )

    ok_count = 0
    err_count = 0

    def make_task(audioFile):
        return (
            audioFile, output_dir, tmp_root, exe, meth, sf, rfile, rlev,
            args.target_sr, args.debug_paths,
        )

    # Item 7: chunked submission so we never hold 800k futures in memory.
    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            with tqdm(total=len(audioFiles), desc="Processing") as pbar:
                for chunk_start in range(0, len(audioFiles), args.chunk_size):
                    chunk = audioFiles[chunk_start:chunk_start + args.chunk_size]
                    futures = [executor.submit(process_one, make_task(f)) for f in chunk]

                    for future in as_completed(futures):
                        result = future.result()
                        if result["status"] == "ok":
                            ok_count += 1
                        else:
                            err_count += 1
                            with open(error_log_path, "a", encoding="utf-8") as f:
                                f.write(f"FILE: {result['audioFile']}\n")
                                f.write(result["error"])
                                f.write("\n" + "=" * 80 + "\n")
                        pbar.update(1)
    finally:
        # Clean up persistent worker dirs.
        with _worker_dirs_lock:
            for d in _worker_dirs:
                shutil.rmtree(d, ignore_errors=True)
        # Only remove tmp_root if we created it under output_dir; leave a
        # user-supplied RAM disk path alone.
        if not args.tmp_dir:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nDone.")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {err_count}")
    print(f"Error log: {error_log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
soundaqnet.prediction
=====================
Convert raw SoundAQnet output folders into tidy pandas DataFrames / CSV files.

CLI
---
    soundaqnet-to-df [--export both|aq|event]
                     [--paq_dir  SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs]
                     [--event_dir SoundAQnet_event_probability]
                     [--output_prefix soundAQ]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── AQ (scene / ISO / PAQ) parsing ───────────────────────────────────────────


def load_aq_outputs(paq_dir: str | Path) -> pd.DataFrame:
    """
    Read every ``*_scene_PAQ.txt`` file in *paq_dir* and return a DataFrame
    with one row per soundscape clip.
    """
    paq_dir = Path(paq_dir)
    files = [p for p in paq_dir.iterdir() if p.is_file()]

    dic: dict[str, list] = {
        "id": [],
        "scene": [],
        "ISOEvs": [],
        "ISOPls": [],
        "pleasant": [],
        "eventful": [],
        "chaotic": [],
        "vibrant": [],
        "uneventful": [],
        "calm": [],
        "annoying": [],
        "monotonous": [],
    }

    for p in tqdm(files, total=len(files)):
        clip_id = p.name.split("_scene_PAQ.txt")[0]
        lines = [
            line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if len(lines) < 2:
            raise ValueError(f"Expected at least ISO and PAQ lines in {p}")

        # Current soundaqnet-infer output is:
        #   scene
        #   isop<TAB>isoe
        #   pleasant<TAB>eventful<TAB>...
        # Older files may omit the scene line, so keep them readable.
        if len(lines) >= 3:
            scene = lines[0]
            iso_line = lines[1]
            paq_line = lines[2]
        else:
            scene = ""
            iso_line = lines[0]
            paq_line = lines[1]

        ISOPls, ISOEvs = iso_line.split("\t")
        pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = (
            paq_line.split("\t")
        )

        dic["id"].append(clip_id)
        dic["scene"].append(scene)
        dic["ISOPls"].append(float(ISOPls))
        dic["ISOEvs"].append(float(ISOEvs))
        dic["pleasant"].append(float(pleasant))
        dic["eventful"].append(float(eventful))
        dic["chaotic"].append(float(chaotic))
        dic["vibrant"].append(float(vibrant))
        dic["uneventful"].append(float(uneventful))
        dic["calm"].append(float(calm))
        dic["annoying"].append(float(annoying))
        dic["monotonous"].append(float(monotonous))

    return pd.DataFrame(dic)


def aggregate_aq(df: pd.DataFrame) -> pd.DataFrame:
    """Group by clip id and compute min/max/std/mean for every AQ attribute."""
    agg_cols = [
        "ISOPls",
        "ISOEvs",
        "pleasant",
        "eventful",
        "chaotic",
        "vibrant",
        "uneventful",
        "calm",
        "annoying",
        "monotonous",
    ]
    agg_kwargs = {}
    for col in agg_cols:
        agg_kwargs[f"min_{col}"] = (col, "min")
        agg_kwargs[f"max_{col}"] = (col, "max")
        agg_kwargs[f"std_{col}"] = (col, "std")
        agg_kwargs[f"avg_{col}"] = (col, "mean")

    return df.groupby("id", as_index=True, group_keys=True).agg(**agg_kwargs)


# ── Event probability parsing ─────────────────────────────────────────────────

EVENT_LABELS = [
    "Silence",
    "Human_sounds",
    "Wind",
    "Water",
    "Natural_sounds",
    "Traffic",
    "Sounds_of_things",
    "Vehicle",
    "Bird",
    "Outside_rural_or_natural",
    "Environment_and_background",
    "Speech",
    "Music",
    "Noise",
    "Animal",
]


def load_event_outputs(event_dir: str | Path) -> pd.DataFrame:
    """
    Read every ``*_event.txt`` file in *event_dir* and return a DataFrame
    with event labels ranked by probability (highest first).
    """
    event_dir = Path(event_dir)
    files = [p for p in event_dir.iterdir() if p.is_file()]

    dic: dict[str, list] = {"id": [], "event_rank": []}

    for p in tqdm(files, total=len(files)):
        clip_id = p.name.split("_event.txt")[0]
        df = pd.read_csv(p, header=None, names=["event"])
        prob = df["event"].tolist()
        reranked = sorted(zip(EVENT_LABELS, prob), key=lambda x: x[1], reverse=True)
        dic["id"].append(clip_id)
        dic["event_rank"].append([label for label, score in reranked])

    return pd.DataFrame(dic)


# ── Public API ────────────────────────────────────────────────────────────────


def predictions_to_dataframe(
    paq_dir: str | Path | None = None,
    event_dir: str | Path | None = None,
    export: str = "both",
) -> dict[str, pd.DataFrame]:
    """
    Load SoundAQnet output directories and return DataFrames.

    Parameters
    ----------
    paq_dir:    Path to SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs/
    event_dir:  Path to SoundAQnet_event_probability/
    export:     'both' | 'aq' | 'event'

    Returns
    -------
    dict with keys 'aq', 'aq_stats', 'events' (whichever were requested).
    """
    result: dict[str, pd.DataFrame] = {}

    if export in ("both", "aq"):
        if paq_dir is None:
            paq_dir = os.path.join(os.getcwd(), "SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs")
        df_aq = load_aq_outputs(paq_dir)
        result["aq"] = df_aq
        result["aq_stats"] = aggregate_aq(df_aq)

    if export in ("both", "event"):
        if event_dir is None:
            event_dir = os.path.join(os.getcwd(), "SoundAQnet_event_probability")
        result["events"] = load_event_outputs(event_dir)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI: soundaqnet-to-df"""
    parser = argparse.ArgumentParser(
        description="Convert SoundAQnet output folders to CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--export",
        default="both",
        choices=["both", "aq", "event"],
        help="Which outputs to process (default: both).",
    )
    parser.add_argument(
        "--paq_dir",
        default=os.path.join(os.getcwd(), "SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs"),
        help="Directory with scene/ISO/PAQ outputs.",
    )
    parser.add_argument(
        "--event_dir",
        default=os.path.join(os.getcwd(), "SoundAQnet_event_probability"),
        help="Directory with event probability outputs.",
    )
    parser.add_argument("--output_prefix", default="soundAQ", help="Prefix for output CSV files.")
    args = parser.parse_args()

    dfs = predictions_to_dataframe(
        paq_dir=args.paq_dir if args.export in ("both", "aq") else None,
        event_dir=args.event_dir if args.export in ("both", "event") else None,
        export=args.export,
    )

    cwd = os.getcwd()

    if "aq" in dfs:
        out_path = os.path.join(cwd, f"{args.output_prefix}.csv")
        dfs["aq"].to_csv(out_path, index=False)
        print(f"Saved {len(dfs['aq'])} rows → {out_path}")

    if "aq_stats" in dfs:
        out_path = os.path.join(cwd, f"{args.output_prefix}_stats.csv")
        dfs["aq_stats"].to_csv(out_path, index=True)
        print(f"Saved stats → {out_path}")

    if "events" in dfs:
        out_path = os.path.join(cwd, f"{args.output_prefix}EventRank.csv")
        dfs["events"].to_csv(out_path, index=False)
        print(f"Saved event rankings → {out_path}")

    return 0


if __name__ == "__main__":
    main()

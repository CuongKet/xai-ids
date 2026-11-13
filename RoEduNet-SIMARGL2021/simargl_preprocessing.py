#!/usr/bin/env python
"""
RoEduNet-SIMARGL2021 leak-free preprocessing.

Replaces the copy-pasted preprocessing of the legacy scripts (now in legacy/), fixing:
- The TEST set was oversampled (8 scripts), inflating every reported metric.
  -> oversampling is optional and applies to the training split only.
- PROTOCOL_MAP one-hot columns were joined to the shuffled DataFrame by index, so they were attached to the
  wrong rows, and were named from pd.factorize order (appearance) while OneHotEncoder orders categories
  alphabetically. -> OneHotEncoder fit on the training split, applied row-aligned, named from its categories.
- Max-scaling was fit on the full dataset before the split. -> MaxAbsScaler fit on the training split only.
- Unstratified, unseeded split with no validation set. -> seeded, stratified train / val / test split.
- The all-feature scripts used identifier / time columns (FLOW_ID, TOTAL_FLOWS_EXP == FLOW_ID, FIRST_SWITCHED,
  LAST_SWITCHED, L4_SRC_PORT, L4_DST_PORT). Each capture file contains a single class recorded at a different
  time, so these columns identify the file (and therefore the class). -> excluded.

Classes: Normal (ALERT empty), Denial of Service, Port Scanning. The 571-row malware capture is excluded by
default, as in the legacy scripts (--include-malware adds it as a 4th class).

Usage:
    python RoEduNet-SIMARGL2021/simargl_preprocessing.py --features top
    python RoEduNet-SIMARGL2021/simargl_preprocessing.py --features all
    python Framework/All_models.py --data-dir RoEduNet-SIMARGL2021/data/top --model all --seeds 42 43 44
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SUITE_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "Framework"))
from prep_utils import split_scale_save, stratified_cap  # noqa: E402

CATEGORICAL = ["PROTOCOL_MAP"]

# Feature subsets of the legacy *_final.py (top) and *_all_final.py (all) scripts
TOP_FEATURES = [
    "PROTOCOL_MAP", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "TCP_WIN_MIN_IN", "TCP_WIN_MIN_OUT", "TCP_WIN_MSS_IN", "TCP_WIN_SCALE_IN", "TCP_WIN_SCALE_OUT",
    "SRC_TOS", "DST_TOS",
]
LEAKY_ID_COLUMNS = ["FLOW_ID", "TOTAL_FLOWS_EXP", "FIRST_SWITCHED", "LAST_SWITCHED", "L4_SRC_PORT", "L4_DST_PORT"]
ALL_FEATURES = [
    "FLOW_DURATION_MILLISECONDS", "TCP_WIN_MSS_IN", "TCP_WIN_MAX_IN", "TCP_WIN_MIN_IN", "TCP_WIN_MIN_OUT",
    "PROTOCOL", "PROTOCOL_MAP", "TCP_WIN_MAX_OUT", "TCP_FLAGS", "TCP_WIN_SCALE_OUT", "TCP_WIN_SCALE_IN",
    "SRC_TOS", "DST_TOS", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "TOTAL_PKTS_EXP", "TOTAL_BYTES_EXP",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
]  # legacy all-feature list minus LEAKY_ID_COLUMNS

FILE_CLASS = {"normal": "Normal", "dos": "Denial of Service", "portscanning": "Port Scanning", "malware": "Malware"}
ALERT_TO_CLASS = {"Denial of Service": "Denial of Service", "Port Scanning": "Port Scanning", "Malware": "Malware"}

RAW_DIR_CANDIDATES = [
    os.path.join(REPO_DIR, "sensor_db"),
    os.path.join(SUITE_DIR, "sensor_db"),
    os.path.join(REPO_DIR, "Framework", "sigma"),
]


def find_raw_dir(explicit):
    for d in ([explicit] if explicit else RAW_DIR_CANDIDATES):
        if d and glob.glob(os.path.join(d, "normal-*.csv")):
            return d
    raise FileNotFoundError(
        "SIMARGL2021 sensor CSVs (normal-*.csv, dos-*.csv, portscanning-*.csv) not found. Pass --raw-dir or "
        "place them in one of: " + ", ".join(RAW_DIR_CANDIDATES)
    )


def load_simargl(raw_dir, features, include_malware):
    numeric = [c for c in features if c not in CATEGORICAL]
    dtypes = {c: np.float32 for c in numeric}
    dtypes.update({c: "category" for c in CATEGORICAL if c in features})
    dtypes["ALERT"] = "category"

    frames = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        kind = os.path.basename(f).split("-")[0]
        if kind not in FILE_CLASS or (kind == "malware" and not include_malware):
            print(f"Skipping {os.path.basename(f)}")
            continue
        n_rows, kept = 0, []
        # Chunked read + per-chunk dedup keeps memory bounded (raw data is ~31M rows / 4.5 GB)
        for chunk in pd.read_csv(f, usecols=features + ["ALERT"], dtype=dtypes, chunksize=2_000_000):
            n_rows += len(chunk)
            kept.append(chunk.drop_duplicates())
        df = pd.concat(kept, ignore_index=True).drop_duplicates()

        # Label from the ALERT column (empty = normal traffic); check it agrees with the file name
        label = df["ALERT"].astype(object).map(ALERT_TO_CLASS).fillna("Normal")
        mismatched = int((label != FILE_CLASS[kind]).sum())
        if mismatched:
            print(f"[WARN] {os.path.basename(f)}: {mismatched:,} rows whose ALERT disagrees with the file name")
        df = df.drop(columns=["ALERT"]).assign(Label=label.to_numpy())
        print(f"Loaded {os.path.basename(f)}: {n_rows:,} rows -> {len(df):,} unique")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype(str)
    return df


def main():
    p = argparse.ArgumentParser(description="Leak-free RoEduNet-SIMARGL2021 preprocessing.")
    p.add_argument("--features", choices=["top", "all"], default="top")
    p.add_argument("--raw-dir", default=None, help="Directory with the sensor CSV files.")
    p.add_argument("--output-dir", default=None, help="Default: RoEduNet-SIMARGL2021/data/<features>")
    p.add_argument("--include-malware", action="store_true", help="Add the 571-row malware capture as a 4th class.")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--oversample", choices=["none", "balance"], default="none",
                   help="Training-split oversampling (default none: models use balanced class weights).")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Stratified subsample of the deduplicated data before splitting (for slow models / quick runs).")
    p.add_argument("--scaler", choices=["maxabs", "minmax", "standard"], default="maxabs",
                   help="Scaler fit on the training split (default maxabs = the legacy divide-by-max scaling).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    features = TOP_FEATURES if args.features == "top" else ALL_FEATURES
    assert not set(features) & set(LEAKY_ID_COLUMNS)
    class_names = ["Normal", "Denial of Service", "Port Scanning"] + (["Malware"] if args.include_malware else [])
    output_dir = args.output_dir or os.path.join(SUITE_DIR, "data", args.features)
    raw_dir = find_raw_dir(args.raw_dir)

    df = load_simargl(raw_dir, features, args.include_malware)
    n_before = len(df)
    df = df.drop_duplicates()
    n_cross_dupes = n_before - len(df)
    numeric = [c for c in features if c not in CATEGORICAL]
    finite = np.isfinite(df[numeric].to_numpy(dtype=np.float64)).all(axis=1)
    n_nonfinite = int((~finite).sum())
    df = df[finite]
    conflicting = int(df.duplicated(subset=features, keep=False).sum())
    print(
        f"Rows after per-file dedup: {n_before:,} | cross-file duplicates dropped: {n_cross_dupes:,} | "
        f"non-finite dropped: {n_nonfinite:,} | remaining: {len(df):,} "
        f"({conflicting:,} rows share a feature vector with a different label)"
    )

    y = df["Label"].map({c: i for i, c in enumerate(class_names)}).to_numpy()
    X = df[features].reset_index(drop=True)
    X, y = stratified_cap(X, y, args.max_rows, args.seed)

    split_scale_save(
        X, y, class_names, output_dir,
        categorical_cols=CATEGORICAL,
        test_size=args.test_size, val_size=args.val_size,
        scaler=args.scaler, oversample=args.oversample, seed=args.seed,
        meta_extra={"dataset": "RoEduNet-SIMARGL2021", "feature_set": args.features, "raw_dir": raw_dir,
                    "excluded_id_columns": LEAKY_ID_COLUMNS, "include_malware": args.include_malware,
                    "max_rows": args.max_rows},
    )


if __name__ == "__main__":
    main()

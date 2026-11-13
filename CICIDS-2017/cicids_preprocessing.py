#!/usr/bin/env python
"""
CICIDS-2017 (MachineLearningCVE) leak-free preprocessing.

Replaces the copy-pasted preprocessing of the legacy scripts (now in legacy/), fixing:
- Train and test labels were factorized separately (pd.factorize), so the same integer could mean a
  different class in each split; AUC and per-class metrics were computed on mismatched labels.
  -> one fixed label -> id mapping (CLASS_NAMES) for every split.
- Max-scaling was fit on the full dataset before the split. -> MaxAbsScaler fit on the training split only.
- inf / NaN in flow-rate features were turned into 0 after scaling. -> rows with non-finite features are dropped
  (and counted) before splitting.
- Some legacy scripts oversampled the TEST set. -> oversampling is optional and applies to the training split only.
- Unstratified, unseeded split with no validation set. -> seeded, stratified train / val / test split.

Usage:
    python CICIDS-2017/cicids_preprocessing.py --features top   # 15 features of the *_final.py scripts
    python CICIDS-2017/cicids_preprocessing.py --features all   # 77 features of the *_all_final.py scripts
    python Framework/All_models.py --data-dir CICIDS-2017/data/top --model all --seeds 42 43 44
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

CLASS_NAMES = ["BENIGN", "DoS/DDoS", "PortScan", "Brute Force", "Web Attack", "Bot", "Infiltration"]

# Feature subsets used by the legacy *_final.py (top) and *_all_final.py (all) scripts
TOP_FEATURES = [
    "Packet Length Std", "Total Length of Bwd Packets", "Subflow Bwd Bytes", "Destination Port",
    "Packet Length Variance", "Bwd Packet Length Mean", "Avg Bwd Segment Size", "Bwd Packet Length Max",
    "Init_Win_bytes_backward", "Total Length of Fwd Packets", "Subflow Fwd Bytes", "Init_Win_bytes_forward",
    "Average Packet Size", "Packet Length Mean", "Max Packet Length",
]
ALL_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min",
]

RAW_DIR_CANDIDATES = [
    os.path.join(REPO_DIR, "cicids_db"),
    os.path.join(SUITE_DIR, "cicids_db"),
    os.path.join(REPO_DIR, "Framework", "cicids-2017"),
]


def map_label(raw: str):
    s = str(raw).strip()
    if s == "BENIGN":
        return "BENIGN"
    if s.startswith("DoS") or s in ("DDoS", "Heartbleed"):
        return "DoS/DDoS"
    if s == "PortScan":
        return "PortScan"
    if s in ("FTP-Patator", "SSH-Patator"):
        return "Brute Force"
    if s.startswith("Web Attack"):  # raw labels contain a mis-encoded dash ("Web Attack \x96 XSS")
        return "Web Attack"
    if s == "Bot":
        return "Bot"
    if s == "Infiltration":
        return "Infiltration"
    return None


def find_raw_dir(explicit):
    for d in ([explicit] if explicit else RAW_DIR_CANDIDATES):
        if d and glob.glob(os.path.join(d, "*.pcap_ISCX.csv")):
            return d
    raise FileNotFoundError(
        "CICIDS-2017 CSVs (*.pcap_ISCX.csv) not found. Pass --raw-dir or place them in one of: "
        + ", ".join(RAW_DIR_CANDIDATES)
    )


def load_cicids(raw_dir: str, features):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.pcap_ISCX.csv")))
    if len(files) != 8:
        print(f"[WARN] Expected 8 CICIDS-2017 CSV files, found {len(files)} in {raw_dir}")
    wanted = set(features) | {"Label"}
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False,
                         usecols=lambda c: c.strip() in wanted)
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        missing = wanted - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(f)} is missing columns: {sorted(missing)}")
        frames.append(df)
        print(f"Loaded {os.path.basename(f)}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description="Leak-free CICIDS-2017 preprocessing.")
    p.add_argument("--features", choices=["top", "all"], default="top")
    p.add_argument("--raw-dir", default=None, help="Directory with the 8 *.pcap_ISCX.csv files.")
    p.add_argument("--output-dir", default=None, help="Default: CICIDS-2017/data/<features>")
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
    output_dir = args.output_dir or os.path.join(SUITE_DIR, "data", args.features)
    raw_dir = find_raw_dir(args.raw_dir)

    df = load_cicids(raw_dir, features)
    n0 = len(df)
    df["Label"] = df["Label"].map(map_label)
    n_unlabelled = int(df["Label"].isna().sum())
    df = df.dropna(subset=["Label"])

    X = df[features].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    finite = np.isfinite(X.to_numpy()).all(axis=1)
    n_nonfinite = int((~finite).sum())
    X, labels = X[finite], df["Label"][finite]

    data = X.assign(Label=labels.to_numpy()).drop_duplicates()
    n_dupes = len(X) - len(data)
    conflicting = int(data.duplicated(subset=features, keep=False).sum())
    print(
        f"Rows: {n0:,} raw | {n_unlabelled:,} unlabelled/blank dropped | {n_nonfinite:,} with inf/NaN features dropped | "
        f"{n_dupes:,} exact duplicates dropped | {len(data):,} remaining "
        f"({conflicting:,} rows share a feature vector with a different label)"
    )

    y = data["Label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()
    X = data[features].reset_index(drop=True)
    X, y = stratified_cap(X, y, args.max_rows, args.seed)

    split_scale_save(
        X, y, CLASS_NAMES, output_dir,
        test_size=args.test_size, val_size=args.val_size,
        scaler=args.scaler, oversample=args.oversample, seed=args.seed,
        meta_extra={"dataset": "CICIDS-2017", "feature_set": args.features, "raw_dir": raw_dir,
                    "max_rows": args.max_rows},
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
"""
NSL-KDD Preprocessing Pipeline (Leak-Free & Reproducible)

Fixes applied from audit:
- B1: Uses KDDTrain+.txt for training and KDDTest+.txt for testing (no train-on-test leakage).
- B3: Uses script-relative paths and configurable output directory.
- B4/B5: Fits OneHotEncoder(handle_unknown='ignore') and StandardScaler on X_train ONLY,
         then transforms X_test with guaranteed identical feature column order.
- B6: Does not standardize or re-encode the integer target labels.
- B7: Replaces extreme 1295x U2R duplication with a controlled, capped oversampling strategy
      (or optional --no-oversample to rely purely on model class_weight).
- B12: Saves both integer class labels (Y_train.pkl, Y_test.pkl) and one-hot matrices
       (Y_train_bin.pkl, Y_test_bin.pkl) to Framework/ directory.
- V1: Carves a stratified validation split out of KDDTrain+ BEFORE fitting encoders/scaler
      and BEFORE oversampling (X_val.pkl, Y_val.pkl), so early stopping / model selection never
      touches KDDTest+ and never sees oversampled duplicates of training rows.
"""

import argparse
import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

SEED = 42

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

ATTACK_MAP = {
    "normal": 0,
    # DoS (1)
    "neptune": 1, "back": 1, "land": 1, "pod": 1, "smurf": 1, "teardrop": 1,
    "mailbomb": 1, "apache2": 1, "processtable": 1, "udpstorm": 1, "worm": 1,
    # Probe (2)
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
    # R2L (3)
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "sendmail": 3, "named": 3,
    "snmpgetattack": 3, "snmpguess": 3, "xlock": 3, "xsnoop": 3, "httptunnel": 3,
    # U2R (4)
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4, "ps": 4,
    "sqlattack": 4, "xterm": 4,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocess_nsl_kdd(
    train_path: str,
    test_path: str,
    output_dir: str,
    oversample_mode: str = "capped",
    val_size: float = 0.1,
    seed: int = SEED,
):
    set_seed(seed)

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found at {train_path}. "
            "Please place KDDTrain+.txt in the Framework/ directory."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Testing file not found at {test_path}.")

    df_train = pd.read_csv(train_path, names=FEATURE_NAMES)
    df_test = pd.read_csv(test_path, names=FEATURE_NAMES)

    print("Dimensions of the Training set:", df_train.shape)
    print("Dimensions of the Test set:", df_test.shape)

    # Drop difficulty score column
    df_train.drop(["difficulty"], axis=1, inplace=True)
    df_test.drop(["difficulty"], axis=1, inplace=True)

    # Map attack strings to 5-class integer IDs (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)
    y_train = df_train["label"].map(ATTACK_MAP)
    y_test = df_test["label"].map(ATTACK_MAP)
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Unmapped attack labels encountered in dataset.")

    y_train = y_train.astype(np.int64).to_numpy()
    y_test = y_test.astype(np.int64).to_numpy()

    X_train_raw = df_train.drop(["label"], axis=1)
    X_test_raw = df_test.drop(["label"], axis=1)

    # Stratified validation split from KDDTrain+ (before any fitting or oversampling)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_val_raw = X_val_raw.reset_index(drop=True)

    numeric_columns = [c for c in X_train_raw.columns if c not in CATEGORICAL_COLUMNS]

    # Fit OneHotEncoder on training categorical columns ONLY (handle_unknown='ignore' prevents test crash/leakage)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)

    X_train_cat = ohe.fit_transform(X_train_raw[CATEGORICAL_COLUMNS])
    X_val_cat = ohe.transform(X_val_raw[CATEGORICAL_COLUMNS])
    X_test_cat = ohe.transform(X_test_raw[CATEGORICAL_COLUMNS])

    cat_feature_names = []
    for col, cats in zip(CATEGORICAL_COLUMNS, ohe.categories_):
        prefix = "Protocol_type_" if col == "protocol_type" else f"{col}_"
        cat_feature_names.extend([f"{prefix}{c}" for c in cats])

    # Fit StandardScaler on training numeric columns ONLY
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_raw[numeric_columns].astype(np.float64))
    X_val_num = scaler.transform(X_val_raw[numeric_columns].astype(np.float64))
    X_test_num = scaler.transform(X_test_raw[numeric_columns].astype(np.float64))

    all_columns = numeric_columns + cat_feature_names
    X_train = pd.DataFrame(
        np.hstack([X_train_num, X_train_cat]),
        columns=all_columns,
    )
    X_val = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=all_columns)
    X_test = pd.DataFrame(
        np.hstack([X_test_num, X_test_cat]),
        columns=all_columns,
    )

    print("Processed X_train shape:", X_train.shape)
    print("Processed X_val shape:", X_val.shape)
    print("Processed X_test shape:", X_test.shape)
    print("Raw training class distribution:", dict(sorted(Counter(y_train).items())))
    print("Validation class distribution:", dict(sorted(Counter(y_val).items())))
    print("Test class distribution:", dict(sorted(Counter(y_test).items())))

    # Handle class imbalance on training split only
    if oversample_mode == "capped":
        # Moderate cap for rare R2L (class 3) and U2R (class 4) without 1,295x identical duplication
        counts = Counter(y_train)
        strategy = {
            cls: max(cnt, 5000 if cls == 3 else (2000 if cls == 4 else cnt))
            for cls, cnt in counts.items()
        }
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        print("Training class distribution (after capped oversampling):", dict(sorted(Counter(y_train).items())))
    elif oversample_mode == "minority":
        ros = RandomOverSampler(sampling_strategy="minority", random_state=seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        print("Training class distribution (after minority oversampling):", dict(sorted(Counter(y_train).items())))
    else:
        print("Oversampling disabled; keeping natural training distribution.")

    lb = LabelBinarizer()
    Y_train_bin = lb.fit_transform(y_train)
    Y_test_bin = lb.transform(y_test)

    os.makedirs(output_dir, exist_ok=True)
    artifacts = {
        "X_train.pkl": X_train,
        "X_val.pkl": X_val,
        "X_test.pkl": X_test,
        "Y_train.pkl": y_train,
        "Y_val.pkl": y_val,
        "Y_test.pkl": y_test,
        "Y_train_bin.pkl": Y_train_bin,
        "Y_test_bin.pkl": Y_test_bin,
    }
    for fname, obj in artifacts.items():
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "wb") as fp:
            pickle.dump(obj, fp)
    print(f"Saved preprocessed datasets to {output_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Preprocess NSL-KDD dataset without leakage.")
    parser.add_argument("--train-file", default=os.path.join(base_dir, "KDDTrain+.txt"))
    parser.add_argument("--test-file", default=os.path.join(base_dir, "KDDTest+.txt"))
    parser.add_argument("--output-dir", default=base_dir)
    parser.add_argument(
        "--oversample",
        choices=["capped", "minority", "none"],
        default="capped",
        help="Training oversampling mode (default: capped).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of KDDTrain+ held out (stratified) as a validation set (default: 0.1).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    preprocess_nsl_kdd(
        train_path=args.train_file,
        test_path=args.test_file,
        output_dir=args.output_dir,
        oversample_mode=args.oversample,
        val_size=args.val_size,
        seed=args.seed,
    )

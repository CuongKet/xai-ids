"""
Shared leak-free split / scale / save helpers for tabular NIDS datasets.

Used by CICIDS-2017/cicids_preprocessing.py and RoEduNet-SIMARGL2021/simargl_preprocessing.py.
Produces the same artifacts as Framework/NSL_preprossing.py (X/Y_{train,val,test}.pkl plus
one-hot Y_*_bin.pkl) and a meta.json with the class names, so Framework/All_models.py can
train and explain any of the datasets with `--data-dir <dir>`.

Order of operations (each step only ever fits on the training split):
  1. stratified train / val / test split (seeded)
  2. one-hot encoder (handle_unknown='ignore') and scaler fit on train, applied to val and test
  3. optional oversampling of the TRAINING split only
"""

import json
import os
import pickle
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MaxAbsScaler, MinMaxScaler, OneHotEncoder, StandardScaler

SCALERS = {"maxabs": MaxAbsScaler, "minmax": MinMaxScaler, "standard": StandardScaler}


def stratified_cap(X: pd.DataFrame, y: np.ndarray, max_rows: Optional[int], seed: int, min_per_class: int = 20):
    """
    Stratified subsample to about `max_rows` rows. Each class keeps its share of `max_rows`, but never fewer than
    `min_per_class` rows (or all its rows if it has fewer), so rare classes survive the later stratified
    train / val / test split.
    """
    if not max_rows or len(X) <= max_rows:
        return X, y
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    frac = max_rows / len(X)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        n = min(len(idx), max(min_per_class, int(round(len(idx) * frac))))
        keep.append(rng.choice(idx, size=n, replace=False))
    keep = np.sort(np.concatenate(keep))
    print(f"Stratified subsample: {len(X):,} -> {len(keep):,} rows (>= {min_per_class} per class)")
    return X.iloc[keep].reset_index(drop=True), y[keep]


def split_scale_save(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: List[str],
    output_dir: str,
    categorical_cols: Optional[List[str]] = None,
    test_size: float = 0.3,
    val_size: float = 0.1,
    scaler: str = "maxabs",
    oversample: str = "none",
    seed: int = 42,
    meta_extra: Optional[dict] = None,
):
    """
    Args:
        X: raw features (numeric + categorical columns); y: integer labels 0..n_classes-1.
        test_size: fraction of all rows held out as the test set.
        val_size: fraction of all rows held out as the validation set (taken from the non-test part).
        oversample: 'none' or 'balance' (RandomOverSampler on the training split only).
    """
    categorical_cols = list(categorical_cols or [])
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    y = np.asarray(y, dtype=np.int64)

    # 1. Stratified, seeded train / val / test split
    X_tmp, X_test_raw, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1.0 - test_size), stratify=y_tmp, random_state=seed
    )

    # 2. Fit transforms on train only
    sc = SCALERS[scaler]()
    parts = {}
    num_train = sc.fit_transform(X_train_raw[numeric_cols].to_numpy(dtype=np.float64))
    parts["train"] = [num_train]
    parts["val"] = [sc.transform(X_val_raw[numeric_cols].to_numpy(dtype=np.float64))]
    parts["test"] = [sc.transform(X_test_raw[numeric_cols].to_numpy(dtype=np.float64))]
    feature_names = list(numeric_cols)

    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)
        parts["train"].append(ohe.fit_transform(X_train_raw[categorical_cols].astype(str)))
        parts["val"].append(ohe.transform(X_val_raw[categorical_cols].astype(str)))
        parts["test"].append(ohe.transform(X_test_raw[categorical_cols].astype(str)))
        for col, cats in zip(categorical_cols, ohe.categories_):
            feature_names.extend(f"{col}_{c}" for c in cats)

    X_train, X_val, X_test = (
        pd.DataFrame(np.hstack(parts[s]).astype(np.float32), columns=feature_names) for s in ("train", "val", "test")
    )

    def dist(v):
        return {class_names[k]: n for k, n in sorted(Counter(v.tolist()).items())}

    print(f"Train {X_train.shape}: {dist(y_train)}")
    print(f"Val   {X_val.shape}: {dist(y_val)}")
    print(f"Test  {X_test.shape}: {dist(y_test)}")

    # 3. Optional oversampling, training split only
    if oversample == "balance":
        from imblearn.over_sampling import RandomOverSampler

        ros = RandomOverSampler(sampling_strategy="not majority", random_state=seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        print(f"Train after balancing oversampling {X_train.shape}: {dist(y_train)}")
    elif oversample != "none":
        raise ValueError(f"Unknown oversample mode {oversample!r}")

    lb = LabelBinarizer().fit(np.arange(len(class_names)))
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {
        "X_train.pkl": X_train,
        "X_val.pkl": X_val,
        "X_test.pkl": X_test,
        "Y_train.pkl": np.asarray(y_train, dtype=np.int64),
        "Y_val.pkl": y_val,
        "Y_test.pkl": y_test,
        "Y_train_bin.pkl": lb.transform(y_train),
        "Y_val_bin.pkl": lb.transform(y_val),
        "Y_test_bin.pkl": lb.transform(y_test),
    }
    for fname, obj in artifacts.items():
        with open(os.path.join(output_dir, fname), "wb") as fp:
            pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "class_names": list(class_names),
        "feature_names": feature_names,
        "categorical_columns": categorical_cols,
        "scaler": scaler,
        "oversample": oversample,
        "test_size": test_size,
        "val_size": val_size,
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    meta.update(meta_extra or {})
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved preprocessed datasets and meta.json to {output_dir}")
    return meta

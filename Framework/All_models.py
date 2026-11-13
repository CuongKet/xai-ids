#!/usr/bin/env python
# coding: utf-8
"""
NSL-KDD Multi-Class Intrusion Detection & Explainable AI (SHAP/LIME) Runner

Fixes applied from audit:
- C1: Replaces blocking `input()` with `--model` CLI argument (falls back to interactive prompt only if omitted in a TTY).
- C2: Sets deterministic global seeds across Python, NumPy, and TensorFlow (incl. TF op determinism).
- C3: Sets AdaBoost `n_estimators=100` (was 1 decision stump).
- C4: Uses native multi-class classifiers (no `MultiOutputClassifier` wrapper that silently maps all-zero rows to Normal).
- C5/C6: Computes multi-class AUC-ROC from continuous probability scores (`y_proba`) and uses division-safe metrics (`eps`).
- C8/D7: Removes dead/commented code, wires `--explain` to `Shap_Lime.py` with single-pass SHAP caching.
- R1: LightGBM never silently falls back to another algorithm; fallback requires --allow-lgbm-fallback and is
      recorded under a different model name ("hgb_fallback").
- R2: Early stopping / model selection uses the held-out validation split from KDDTrain+ (X_val.pkl), never
      KDDTest+ and never oversampled duplicates (MLP, DNN and LightGBM).
- R3: "svm" is a real linear SVM (LinearSVC + sigmoid calibration for probabilities); the former
      SGD log-loss model is kept as "logreg" (it is logistic regression).
- R5: Dataset-agnostic: class names come from <data-dir>/meta.json, so the same runner trains and explains
      NSL-KDD, CICIDS-2017 and RoEduNet-SIMARGL2021 (see each suite's *_preprocessing.py).
- R4: Every run saves the trained model, validation + test metrics (JSON), test probabilities, and appends to
      outputs/results.csv; --seeds runs several seeds and writes a mean ± std summary.
"""

import argparse
import copy
import datetime as dt
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

SEED = 42
# Default (NSL-KDD) class names; replaced at runtime by `class_names` from <data-dir>/meta.json if present.
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

MODEL_REGISTRY = {
    "1": "ada",
    "ada": "ada",
    "2": "knn",
    "knn": "knn",
    "3": "mlp",
    "mlp": "mlp",
    "4": "rf",
    "rf": "rf",
    "5": "dnn",
    "dnn": "dnn",
    "6": "lgbm",
    "lgbm": "lgbm",
    "7": "svm",
    "svm": "svm",
    "8": "logreg",
    "logreg": "logreg",
}
# The seven classifiers of the benchmark; `logreg` is available separately.
PAPER_MODELS = ["ada", "knn", "mlp", "rf", "dnn", "lgbm", "svm"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def ACC(TP, TN, FP, FN):
    return (TP + TN) / max(TP + FP + FN + TN, 1e-7)


def PRECISION(TP, FP):
    return TP / (TP + FP + 1e-7)


def RECALL(TP, FN):
    return TP / (TP + FN + 1e-7)


def F1(Recall, Precision):
    return 2 * Recall * Precision / (Recall + Precision + 1e-7)


def BACC(TP, TN, FP, FN):
    return (TP / (TP + FN + 1e-7) + TN / (TN + FP + 1e-7)) * 0.5


def MCC(TP, TN, FP, FN):
    eps = 1e-7
    return (TN * TP - FN * FP) / (((TP + FP + eps) * (TP + FN + eps) * (TN + FP + eps) * (TN + FN + eps)) ** 0.5)


def load_class_names(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Dataset: {meta.get('dataset', '?')} | feature set: {meta.get('feature_set', '-')} | classes: {meta['class_names']}")
        return list(meta["class_names"])
    print("No meta.json found; assuming NSL-KDD classes.")
    return ["Normal", "DoS", "Probe", "R2L", "U2R"]


def load_artifacts(data_dir: str):
    def load(filename):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing preprocessed file {path}. Run the dataset's preprocessing script first "
                "(Framework/NSL_preprossing.py, CICIDS-2017/cicids_preprocessing.py or "
                "RoEduNet-SIMARGL2021/simargl_preprocessing.py) "
                "(re-run it if X_val.pkl / Y_val.pkl are missing: they were added with the validation split)."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    X_train, X_val, X_test = load("X_train.pkl"), load("X_val.pkl"), load("X_test.pkl")
    ys = []
    for name in ("Y_train.pkl", "Y_val.pkl", "Y_test.pkl"):
        y = np.asarray(load(name))
        ys.append(np.argmax(y, axis=1) if y.ndim == 2 else y)
    y_train, y_val, y_test = ys

    # Guarantee identical feature order across splits
    feature_names = X_train.columns.tolist()
    X_val = X_val.reindex(columns=feature_names, fill_value=0.0)
    X_test = X_test.reindex(columns=feature_names, fill_value=0.0)
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_top_k_anova_features(X_train: pd.DataFrame, y_train: np.ndarray, k: int = 15):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SelectKBest(score_func=f_classif, k="all")
        fit = selector.fit(X_train, y_train)
    scores = pd.Series(np.nan_to_num(fit.scores_, nan=0.0), index=X_train.columns)
    top_k = scores.nlargest(k).index.tolist()
    print(f"Top {k} ANOVA F-score features:", top_k)
    return top_k


def _train_mlp_with_val_early_stopping(X_train, y_train, X_val, y_val, seed, max_epochs, patience):
    """sklearn's built-in early_stopping splits the (oversampled) training data; use the real val split instead."""
    from sklearn.neural_network import MLPClassifier

    classes = np.arange(len(CLASS_NAMES))
    model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=seed, shuffle=True)
    best_loss, best_model, best_epoch, wait = np.inf, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.partial_fit(X_train, y_train, classes=classes)
        val_loss = log_loss(y_val, model.predict_proba(X_val), labels=classes)
        if val_loss < best_loss - 1e-4:
            best_loss, best_model, best_epoch, wait = val_loss, copy.deepcopy(model), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    print(f"[MLP] Early stopping on validation log-loss: best epoch {best_epoch} (val_loss={best_loss:.4f})")
    return best_model, {"best_epoch": best_epoch, "best_val_loss": float(best_loss), "epochs_run": epoch}


def build_and_train_model(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int = SEED,
    max_epochs: int = 200,
    patience: int = 10,
    allow_lgbm_fallback: bool = False,
):
    """Train one model. Returns (model, run_name, training_info, train_time)."""
    n_classes = len(CLASS_NAMES)
    run_name = model_key
    info = {}
    start_train = time.time()

    if model_key == "ada":
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed),
            n_estimators=100,
            learning_rate=0.5,
            random_state=seed,
        )
        model.fit(X_train, y_train)

    elif model_key == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        model.fit(X_train, y_train)

    elif model_key == "mlp":
        model, info = _train_mlp_with_val_early_stopping(
            X_train, y_train, X_val, y_val, seed, max_epochs=max_epochs, patience=patience
        )

    elif model_key == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    elif model_key == "dnn":
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.utils import to_categorical
        from sklearn.utils.class_weight import compute_class_weight

        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        y_val_cat = to_categorical(y_val, num_classes=n_classes)
        weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
        class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(n_classes, activation="softmax"),
        ])
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        history = model.fit(
            X_train.to_numpy(dtype=np.float32),
            y_train_cat,
            validation_data=(X_val.to_numpy(dtype=np.float32), y_val_cat),
            epochs=max_epochs,
            batch_size=128,
            class_weight=class_weight_dict,
            callbacks=[es],
            shuffle=True,
            verbose=0,
        )
        val_losses = history.history["val_loss"]
        best_epoch = int(np.argmin(val_losses)) + 1
        info = {"best_epoch": best_epoch, "best_val_loss": float(min(val_losses)), "epochs_run": len(val_losses)}
        print(f"[DNN] Early stopping on validation loss: best epoch {best_epoch} (val_loss={min(val_losses):.4f})")

    elif model_key == "lgbm":
        try:
            import lightgbm
            from lightgbm import LGBMClassifier
        except OSError as exc:
            if not allow_lgbm_fallback:
                raise SystemExit(
                    f"LightGBM native library could not load ({exc}).\n"
                    "Install OpenMP (Apple Silicon: `arch -arm64 brew install libomp`; Intel: `brew install libomp`) "
                    "and re-run. Pass --allow-lgbm-fallback to use HistGradientBoostingClassifier instead "
                    "(results will be saved as 'hgb_fallback', NOT as LightGBM)."
                )
            print(f"[WARN] LightGBM unavailable ({exc}); using HistGradientBoostingClassifier as 'hgb_fallback'.")
            from sklearn.ensemble import HistGradientBoostingClassifier

            run_name = "hgb_fallback"
            model = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed)
            model.fit(X_train, y_train)
        else:
            model = LGBMClassifier(
                n_estimators=1000,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[lightgbm.early_stopping(stopping_rounds=50, verbose=False)],
            )
            info = {"best_iteration": int(model.best_iteration_)}
            print(f"[LGBM] Early stopping on validation log-loss: best iteration {model.best_iteration_}")

    elif model_key == "svm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        # Linear SVM (hinge loss); sigmoid (Platt) calibration provides probabilities for AUC/SHAP/LIME.
        model = CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000, random_state=seed),
            method="sigmoid",
            cv=3,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    elif model_key == "logreg":
        from sklearn.linear_model import SGDClassifier

        # Logistic regression trained with SGD (log-loss). Formerly mislabelled as "svm".
        model = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    train_time = time.time() - start_train
    print(f"[{run_name.upper()}] Time taken for training: {train_time:.2f} seconds")
    return model, run_name, info, train_time


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, split: str = "test", verbose: bool = True):
    """Print the report and return a flat dict of metrics."""
    labels = list(range(len(CLASS_NAMES)))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    lb = LabelBinarizer()
    lb.fit(labels)
    try:
        roc_auc = roc_auc_score(lb.transform(y_true), y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "macro_auc": roc_auc,
    }

    per_class_lines = []
    for i, class_name in enumerate(CLASS_NAMES):
        TP = confusion[i, i]
        FP = confusion[:, i].sum() - TP
        FN = confusion[i, :].sum() - TP
        TN = confusion.sum() - TP - FP - FN
        Precision, Recall = PRECISION(TP, FP), RECALL(TP, FN)
        cls = {
            "acc": ACC(TP, TN, FP, FN),
            "precision": Precision,
            "recall": Recall,
            "f1": F1(Recall, Precision),
            "bacc": BACC(TP, TN, FP, FN),
            "mcc": MCC(TP, TN, FP, FN),
        }
        for k, v in cls.items():
            metrics[f"{class_name}_{k}"] = v
        per_class_lines.append(
            f"Metrics for {class_name:6s} -> "
            f"Acc: {cls['acc']:.4f} | Prec: {cls['precision']:.4f} | Rec: {cls['recall']:.4f} | "
            f"F1: {cls['f1']:.4f} | BACC: {cls['bacc']:.4f} | MCC: {cls['mcc']:.4f}"
        )
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["confusion_matrix"] = confusion.tolist()

    if verbose:
        print(f"\n[{split.upper()}] Classification Report:")
        print(classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES, zero_division=0))
        print("Confusion Matrix:")
        print(confusion)
    print(
        f"\n[{split.upper()}] Macro F1: {metrics['macro_f1']:.4f} | Overall MCC: {metrics['mcc']:.4f} | "
        f"Macro AUC-ROC (prob): {metrics['macro_auc']:.4f}\n"
    )
    if verbose:
        print("\n".join(per_class_lines))
        print("-" * 81)
    return metrics


def save_run(model, run_name, seed, run_dir, record, y_proba_test):
    os.makedirs(run_dir, exist_ok=True)
    if run_name == "dnn":
        model_path = os.path.join(run_dir, "model.keras")
        model.save(model_path)
    else:
        import joblib
        model_path = os.path.join(run_dir, "model.joblib")
        joblib.dump(model, model_path)
    np.save(os.path.join(run_dir, "test_proba.npy"), y_proba_test)
    record = dict(record, model_path=model_path)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(f"[SAVE] Model, metrics.json and test_proba.npy saved to {run_dir}")


def append_results_csv(path, rows):
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def prompt_model_choice() -> str:
    print("Select the model to train and predict:")
    print("1. AdaBoost Classifier (ada)")
    print("2. K-Nearest Neighbors (knn)")
    print("3. Multi-Layer Perceptron (mlp)")
    print("4. Random Forest Classifier (rf)")
    print("5. Deep Neural Network (dnn)")
    print("6. LightGBM (lgbm)")
    print("7. Linear SVM, calibrated (svm)")
    print("8. Logistic Regression via SGD (logreg)")
    choice = input("Enter your choice (1-8 or name): ").strip().lower()
    if choice not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model choice: {choice!r}")
    return MODEL_REGISTRY[choice]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from Shap_Lime import (
        compute_shap_explanations,
        get_probability_fn,
        lime_plot,
        shap_barplots,
        shap_beeswarmplot,
        shap_waterfallplot,
    )

    parser = argparse.ArgumentParser(description="Train & explain NSL-KDD intrusion detection models.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model choice: 1-8 or one of {ada, knn, mlp, rf, dnn, lgbm, svm, logreg, all}. "
        "'all' runs the 7 benchmark models (excludes logreg).",
    )
    parser.add_argument("--data-dir", type=str, default=base_dir, help="Directory with preprocessed .pkl files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Default: <data-dir>/outputs")
    parser.add_argument("--explain", action="store_true", help="Run SHAP and LIME explanations (first seed only).")
    parser.add_argument("--shap-samples", type=int, default=100, help="Number of stratified test samples for SHAP.")
    parser.add_argument("--row-index", type=int, default=0, help="Test row for local Waterfall/LIME plots (same row for both).")
    parser.add_argument("--class-index", type=int, default=1, help="Target class index for beeswarm/waterfall (see meta.json class_names).")
    parser.add_argument("--seed", type=int, default=SEED, help="Single seed (ignored if --seeds is given).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Run each model for several seeds, e.g. --seeds 42 43 44 45 46.")
    parser.add_argument("--max-epochs", type=int, default=200, help="Max epochs for MLP/DNN (early stopping on validation).")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs) for MLP/DNN.")
    parser.add_argument("--allow-lgbm-fallback", action="store_true", help="If LightGBM cannot load, use HistGradientBoosting (saved as 'hgb_fallback').")
    parser.add_argument("--quiet", action="store_true", help="Only print headline metrics.")
    args = parser.parse_args()

    seeds = args.seeds or [args.seed]
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "outputs")

    global CLASS_NAMES
    CLASS_NAMES = load_class_names(args.data_dir)
    if not 0 <= args.class_index < len(CLASS_NAMES):
        raise SystemExit(f"--class-index must be in [0, {len(CLASS_NAMES) - 1}] for this dataset")

    if args.model is None:
        model_keys = [prompt_model_choice()] if sys.stdin.isatty() else ["rf"]
    elif args.model.lower() == "all":
        model_keys = list(PAPER_MODELS)
    else:
        key = args.model.lower()
        if key not in MODEL_REGISTRY:
            raise SystemExit(f"Invalid --model {args.model!r}. Choose from {sorted(set(MODEL_REGISTRY.values()))} or 'all'.")
        model_keys = [MODEL_REGISTRY[key]]

    set_seed(seeds[0])
    X_train, X_val, X_test, y_train, y_val, y_test = load_artifacts(args.data_dir)
    print(f"Loaded splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    if not 0 <= args.row_index < len(X_test):
        raise SystemExit(f"--row-index must be in [0, {len(X_test) - 1}]")
    row_label = X_test.index[args.row_index]
    compute_top_k_anova_features(X_train, y_train, k=15)

    os.makedirs(args.output_dir, exist_ok=True)
    invocation_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_csv = os.path.join(args.output_dir, "results.csv")
    invocation_rows = []

    for mkey in model_keys:
        for seed_i, seed in enumerate(seeds):
            print(f"\n==================== Running Model: {mkey.upper()} | seed={seed} ====================")
            set_seed(seed)
            model, run_name, train_info, train_time = build_and_train_model(
                mkey,
                X_train,
                y_train,
                X_val,
                y_val,
                seed=seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                allow_lgbm_fallback=args.allow_lgbm_fallback,
            )

            proba_fn = get_probability_fn(model)
            y_proba_val = proba_fn(X_val)
            start_pred = time.time()
            y_proba_test = proba_fn(X_test)
            pred_time = time.time() - start_pred
            print(f"[{run_name.upper()}] Time taken for test prediction: {pred_time:.2f} seconds")

            val_metrics = evaluate_predictions(y_val, np.argmax(y_proba_val, axis=1), y_proba_val, split="val", verbose=False)
            test_metrics = evaluate_predictions(y_test, np.argmax(y_proba_test, axis=1), y_proba_test, split="test", verbose=not args.quiet)

            run_dir = os.path.join(args.output_dir, "runs", run_name, f"seed{seed}")
            record = {
                "invocation_id": invocation_id,
                "model": run_name,
                "estimator": model.__class__.__name__,
                "seed": seed,
                "train_time_s": train_time,
                "test_pred_time_s": pred_time,
                "training_info": train_info,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "val": val_metrics,
                "test": test_metrics,
            }
            save_run(model, run_name, seed, run_dir, record, y_proba_test)

            for split, m in (("val", val_metrics), ("test", test_metrics)):
                row = {
                    "invocation_id": invocation_id,
                    "model": run_name,
                    "seed": seed,
                    "split": split,
                    "train_time_s": round(train_time, 3),
                }
                row.update({k: v for k, v in m.items() if k != "confusion_matrix"})
                invocation_rows.append(row)
            append_results_csv(results_csv, invocation_rows[-2:])

            if args.explain and seed_i == 0:
                explain_dir = os.path.join(args.output_dir, "explanations", run_name, f"seed{seed}")
                os.makedirs(explain_dir, exist_ok=True)
                precomputed = compute_shap_explanations(
                    model,
                    X_train,
                    X_test,
                    y_test=y_test,
                    max_samples=args.shap_samples,
                    seed=seed,
                    cache_path=os.path.join(args.output_dir, "shap_cache", f"shap_{run_name}"),
                    include_index=row_label,
                )
                shap_barplots(
                    model, X_train, X_test,
                    class_names=CLASS_NAMES,
                    output_path=os.path.join(explain_dir, "shap_bar.png"),
                    precomputed=precomputed,
                )
                shap_beeswarmplot(
                    model, X_train, X_test,
                    class_index=args.class_index,
                    class_names=CLASS_NAMES,
                    output_path=os.path.join(explain_dir, f"shap_beeswarm_cls{args.class_index}.png"),
                    precomputed=precomputed,
                )
                shap_waterfallplot(
                    model, X_train, X_test,
                    instance_index=row_label,
                    class_index=args.class_index,
                    class_names=CLASS_NAMES,
                    output_path=os.path.join(explain_dir, f"shap_waterfall_row{args.row_index}.png"),
                    precomputed=precomputed,
                )
                lime_plot(
                    model, X_train, X_test,
                    Y_train=y_train,
                    row_index=row_label,
                    class_names=CLASS_NAMES,
                    seed=seed,
                    output_html=os.path.join(explain_dir, f"lime_row{args.row_index}.html"),
                    output_png=os.path.join(explain_dir, f"lime_row{args.row_index}.png"),
                )

    # Mean ± std summary over seeds for this invocation
    df = pd.DataFrame(invocation_rows)
    headline = ["accuracy", "macro_f1", "weighted_f1", "mcc", "macro_auc"] + [f"{c}_recall" for c in CLASS_NAMES]
    summary = df.groupby(["model", "split"])[headline].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_seeds", df.groupby(["model", "split"]).size())
    summary_path = os.path.join(args.output_dir, f"summary_{invocation_id}.csv")
    summary.to_csv(summary_path)
    print("\n==================== Summary (mean ± std over seeds) ====================")
    for (model_name, split), r in summary.iterrows():
        parts = " | ".join(
            f"{m}: {r[f'{m}_mean']:.4f}" + (f" ± {r[f'{m}_std']:.4f}" if r["n_seeds"] > 1 else "")
            for m in ("macro_f1", "mcc", "macro_auc")
        )
        print(f"{model_name:13s} [{split:4s}] n={int(r['n_seeds'])} | {parts}")
    print(f"\nSaved summary to {summary_path}; all runs appended to {results_csv}")


if __name__ == "__main__":
    main()

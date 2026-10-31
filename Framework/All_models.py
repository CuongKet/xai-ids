#!/usr/bin/env python
# coding: utf-8
"""
NSL-KDD Multi-Class Intrusion Detection & Explainable AI (SHAP/LIME) Runner

Fixes applied from audit:
- C1: Replaces blocking `input()` with `--model` CLI argument (falls back to interactive prompt only if omitted in a TTY).
- C2: Sets deterministic global seeds across Python, NumPy, and TensorFlow.
- C3: Sets AdaBoost `n_estimators=100` (was 1 decision stump).
- C4: Uses native multi-class classifiers (no `MultiOutputClassifier` wrapper that silently maps all-zero rows to Normal).
- C5/C6: Computes multi-class AUC-ROC from continuous probability scores (`y_proba`) and uses division-safe metrics (`eps`).
- C8/D7: Removes dead/commented code, wires `--explain` to `Shap_Lime.py` with single-pass SHAP caching.
"""

import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

SEED = 42
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
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
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


def load_artifacts(data_dir: str):
    def get_pkl_path(filename):
        candidate = os.path.join(data_dir, filename)
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(filename):
            return filename
        raise FileNotFoundError(
            f"Missing preprocessed file {filename}. Run `python Framework/NSL_preprossing.py` first."
        )

    with open(get_pkl_path("X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(get_pkl_path("X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(get_pkl_path("Y_train.pkl"), "rb") as f:
        Y_train = pickle.load(f)
    with open(get_pkl_path("Y_test.pkl"), "rb") as f:
        Y_test = pickle.load(f)

    y_train = np.asarray(Y_train)
    y_test = np.asarray(Y_test)
    if y_train.ndim == 2:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)

    # Guarantee identical feature order between train and test
    feature_names = X_train.columns.tolist()
    X_test = X_test.reindex(columns=feature_names, fill_value=0.0)
    return X_train, X_test, y_train, y_test


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


def build_and_train_model(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    seed: int = SEED,
    epochs: int = 25,
):
    from Shap_Lime import get_probability_fn

    n_classes = len(CLASS_NAMES)
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
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=100,
            early_stopping=True,
            random_state=seed,
        )
        model.fit(X_train, y_train)

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
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.utils import to_categorical
        from sklearn.utils.class_weight import compute_class_weight

        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
        class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(n_classes, activation="softmax"),
        ])
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(
            X_train.to_numpy(dtype=np.float32),
            y_train_cat,
            epochs=epochs,
            batch_size=128,
            class_weight=class_weight_dict,
            verbose=0,
        )

    elif model_key == "lgbm":
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
        except OSError as exc:
            print(
                f"[WARN] LightGBM native library could not load ({exc}). "
                "Install OpenMP on macOS via `brew install libomp`, or falling back to "
                "scikit-learn HistGradientBoostingClassifier (LightGBM-equivalent histogram GBDT)."
            )
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(
                max_iter=200,
                class_weight="balanced",
                random_state=seed,
            )
        model.fit(X_train, y_train)

    elif model_key == "svm":
        from sklearn.linear_model import SGDClassifier

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
    print(f"[{model_key.upper()}] Time taken for training: {train_time:.2f} seconds")

    start_pred = time.time()
    proba_fn = get_probability_fn(model)
    y_proba = proba_fn(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    pred_time = time.time() - start_pred
    print(f"[{model_key.upper()}] Time taken for prediction: {pred_time:.2f} seconds")

    return model, y_pred, y_proba


def evaluate_predictions(y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    confusion = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    print("Confusion Matrix:")
    print(confusion)

    lb = LabelBinarizer()
    lb.fit(list(range(len(CLASS_NAMES))))
    y_test_bin = lb.transform(y_test)

    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    overall_mcc = matthews_corrcoef(y_test, y_pred)
    print(f"\nMacro F1: {macro_f1:.4f} | Overall MCC: {overall_mcc:.4f} | Macro AUC-ROC (prob): {roc_auc:.4f}\n")

    for i, class_name in enumerate(CLASS_NAMES):
        TP = confusion[i, i]
        FP = confusion[:, i].sum() - TP
        FN = confusion[i, :].sum() - TP
        TN = confusion.sum() - TP - FP - FN

        Acc = ACC(TP, TN, FP, FN)
        Precision = PRECISION(TP, FP)
        Recall = RECALL(TP, FN)
        F1_val = F1(Recall, Precision)
        Balanced_accuracy = BACC(TP, TN, FP, FN)
        Matthews = MCC(TP, TN, FP, FN)

        print(
            f"Metrics for {class_name:6s} -> "
            f"Acc: {Acc:.4f} | Prec: {Precision:.4f} | Rec: {Recall:.4f} | "
            f"F1: {F1_val:.4f} | BACC: {Balanced_accuracy:.4f} | MCC: {Matthews:.4f}"
        )
    print("-" * 81)


def prompt_model_choice() -> str:
    print("Select the model to train and predict:")
    print("1. AdaBoost Classifier (ada)")
    print("2. K-Nearest Neighbors (knn)")
    print("3. Multi-Layer Perceptron (mlp)")
    print("4. Random Forest Classifier (rf)")
    print("5. Deep Neural Network (dnn)")
    print("6. LightGBM (lgbm)")
    print("7. Linear SVM / SGD (svm)")
    choice = input("Enter your choice (1-7 or name): ").strip().lower()
    if choice not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model choice: {choice!r}")
    return MODEL_REGISTRY[choice]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    from Shap_Lime import (
        compute_shap_explanations,
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
        help="Model choice: 1-7 or one of {ada, knn, mlp, rf, dnn, lgbm, svm, all}.",
    )
    parser.add_argument("--data-dir", type=str, default=base_dir, help="Directory with preprocessed .pkl files.")
    parser.add_argument("--output-dir", type=str, default=os.path.join(base_dir, "outputs"))
    parser.add_argument("--explain", action="store_true", help="Run SHAP and LIME explanations and save plots.")
    parser.add_argument("--shap-samples", type=int, default=100, help="Number of stratified test samples for SHAP.")
    parser.add_argument("--row-index", type=int, default=0, help="Test row index for local Waterfall/LIME plots.")
    parser.add_argument("--class-index", type=int, default=1, help="Target class index (0..4) for beeswarm/waterfall.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    X_train, X_test, y_train, y_test = load_artifacts(args.data_dir)
    compute_top_k_anova_features(X_train, y_train, k=15)

    if args.model is None:
        if sys.stdin.isatty():
            model_keys = [prompt_model_choice()]
        else:
            model_keys = ["rf"]
    elif args.model.lower() == "all":
        model_keys = ["ada", "knn", "mlp", "rf", "dnn", "lgbm", "svm"]
    else:
        key = args.model.lower()
        if key not in MODEL_REGISTRY:
            raise SystemExit(f"Invalid --model {args.model!r}. Choose from {sorted(set(MODEL_REGISTRY.values()))} or 'all'.")
        model_keys = [MODEL_REGISTRY[key]]

    os.makedirs(args.output_dir, exist_ok=True)

    for mkey in model_keys:
        print(f"\n==================== Running Model: {mkey.upper()} ====================")
        model, y_pred, y_proba = build_and_train_model(mkey, X_train, y_train, X_test, seed=args.seed)
        evaluate_predictions(y_test, y_pred, y_proba)

        if args.explain:
            cache_file = os.path.join(args.output_dir, f"shap_cache_{mkey}.npz")
            precomputed = compute_shap_explanations(
                model,
                X_train,
                X_test,
                y_test=y_test,
                max_samples=args.shap_samples,
                seed=args.seed,
                cache_path=cache_file,
            )
            shap_barplots(
                model,
                X_train,
                X_test,
                class_names=CLASS_NAMES,
                output_path=os.path.join(args.output_dir, f"shap_bar_{mkey}.png"),
                precomputed=precomputed,
            )
            shap_beeswarmplot(
                model,
                X_train,
                X_test,
                class_index=args.class_index,
                class_names=CLASS_NAMES,
                output_path=os.path.join(args.output_dir, f"shap_beeswarm_{mkey}_cls{args.class_index}.png"),
                precomputed=precomputed,
            )
            shap_waterfallplot(
                model,
                X_train,
                X_test,
                instance_index=args.row_index,
                class_index=args.class_index,
                class_names=CLASS_NAMES,
                output_path=os.path.join(args.output_dir, f"shap_waterfall_{mkey}_row{args.row_index}.png"),
                precomputed=precomputed,
            )
            lime_plot(
                model,
                X_train,
                X_test,
                Y_train=y_train,
                row_index=args.row_index,
                class_names=CLASS_NAMES,
                seed=args.seed,
                output_html=os.path.join(args.output_dir, f"lime_{mkey}_row{args.row_index}.html"),
                output_png=os.path.join(args.output_dir, f"lime_{mkey}_row{args.row_index}.png"),
            )


if __name__ == "__main__":
    main()

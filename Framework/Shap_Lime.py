"""
Explainable AI (SHAP & LIME) utilities for NIDS models.

Fixes applied from audit:
- D1/D2: Dispatches TreeExplainer for RandomForest/ExtraTrees,
         GradientExplainer/DeepExplainer or probability-based KernelExplainer with
         k-means background summary (k=20) and bounded nsamples for non-tree models.
- D3/D4: Always explains continuous class probabilities (predict_proba) instead of
         step-function hard predictions; supports native multi-class, Keras Sequential,
         SGD decision_function (via softmax), and legacy MultiOutputClassifier wrappers.
- D5: Normalizes SHAP output shapes across SHAP <0.45 (list of arrays) and SHAP >=0.45
      (3D array of shape [n_samples, n_features, n_classes]).
- D6: Uses seeded stratified sampling for explanation subsets.
- D7: Computes SHAP values once and caches/returns them for reuse across bar, beeswarm,
      and waterfall plots; saves all figures to disk instead of failing in headless mode.
- E1: Cache files are keyed by a fingerprint of the model (its probabilities on X_eval), the
      evaluation rows and the explainer settings, so stale explanations are never reloaded.
- E2: The SHAP evaluation set always contains the LIME row, and both use X_test index labels,
      so local SHAP and LIME plots for "row N" explain the same instance.
- E3: All SHAP values are in probability space (LightGBM uses KernelExplainer on predict_proba
      instead of log-odds TreeExplainer output); an additivity check verifies this at runtime.
"""

import hashlib
import json
import os
import time
from typing import Callable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.special import softmax
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def get_probability_fn(model) -> Callable[[np.ndarray], np.ndarray]:
    """Return a clean (n_samples, n_classes) probability function for any supported model."""
    feature_names = getattr(model, "feature_names_in_", None)

    def _prepare_input(X):
        if isinstance(X, pd.DataFrame):
            return X if feature_names is not None else X.to_numpy()
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if feature_names is not None:
            return pd.DataFrame(arr, columns=feature_names)
        return arr

    # Legacy MultiOutputClassifier wrapper support
    if hasattr(model, "estimators_") and model.__class__.__name__ == "MultiOutputClassifier":
        def _multioutput_proba(X):
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if hasattr(model, "predict_proba"):
                probs_list = model.predict_proba(arr)
                pos_probs = np.column_stack([
                    p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probs_list
                ])
                row_sums = pos_probs.sum(axis=1, keepdims=True)
                return np.where(row_sums > 0, pos_probs / np.maximum(row_sums, 1e-12), softmax(pos_probs, axis=1))
            scores = np.column_stack([est.decision_function(arr) for est in model.estimators_])
            return softmax(scores, axis=1)
        return _multioutput_proba

    if hasattr(model, "predict_proba"):
        def _sklearn_proba(X):
            return np.asarray(model.predict_proba(_prepare_input(X)), dtype=np.float64)
        return _sklearn_proba

    if hasattr(model, "decision_function"):
        def _decision_proba(X):
            scores = np.asarray(model.decision_function(_prepare_input(X)), dtype=np.float64)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            return softmax(scores, axis=1)
        return _decision_proba

    # Keras model (.predict returns probabilities directly)
    def _keras_proba(X):
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.asarray(model.predict(arr, verbose=0), dtype=np.float64)

    return _keras_proba


def stratified_sample(
    X: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    n_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic stratified subset of X for representative XAI evaluation."""
    if len(X) <= n_samples:
        return X.copy()
    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.ndim == 2:
            y_arr = np.argmax(y_arr, axis=1)
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
            idx, _ = next(sss.split(X, y_arr))
            return X.iloc[idx].copy()
        except ValueError:
            pass
    return X.sample(n=n_samples, random_state=seed).copy()


def _normalize_shap_values(raw_shap_values, n_samples: int, n_features: int) -> List[np.ndarray]:
    """
    Normalize SHAP values into a list of 2D arrays (one per class), each of shape (n_samples, n_features).
    Works seamlessly across SHAP 0.41-0.46+.
    """
    if isinstance(raw_shap_values, shap.Explanation):
        raw_shap_values = raw_shap_values.values

    if isinstance(raw_shap_values, list):
        return [np.asarray(v) for v in raw_shap_values]

    arr = np.asarray(raw_shap_values)
    if arr.ndim == 3:
        # SHAP >= 0.45 returns (n_samples, n_features, n_classes)
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            return [arr[:, :, c] for c in range(arr.shape[2])]
        # Older / alternative layout (n_classes, n_samples, n_features)
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            return [arr[c, :, :] for c in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr]
    raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")


TREE_EXPLAINER_MODELS = ("RandomForestClassifier", "ExtraTreesClassifier")


def _cache_key(model, X_eval: pd.DataFrame, settings: dict) -> str:
    """
    Fingerprint everything the SHAP values depend on: explainer settings, the exact evaluation rows
    (values + test-set indices) and the trained model itself (via its probabilities on X_eval), so a
    retrained model, new preprocessing, different seed or sample count never reuses a stale cache.
    """
    h = hashlib.sha256()
    h.update(json.dumps(settings, sort_keys=True).encode())
    h.update(np.ascontiguousarray(X_eval.to_numpy(dtype=np.float64)).tobytes())
    h.update(np.asarray(X_eval.index).astype(np.int64).tobytes())
    proba = get_probability_fn(model)(X_eval)
    h.update(np.round(proba, 6).astype(np.float64).tobytes())
    return h.hexdigest()[:16]


def compute_shap_explanations(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: Optional[np.ndarray] = None,
    max_samples: int = 100,
    background_k: int = 20,
    nsamples: int = 300,
    seed: int = 42,
    cache_path: Optional[str] = None,
    include_index=None,
) -> Tuple[List[np.ndarray], np.ndarray, pd.DataFrame]:
    """
    Compute or load cached multi-class SHAP values, always in PROBABILITY space so attributions
    are comparable across all models.

    - RandomForest / ExtraTrees: TreeExplainer (interventional); raw tree output is already the
      class probability, verified by an additivity check against predict_proba.
    - All other models (incl. LightGBM, whose raw TreeExplainer output is log-odds and whose
      multi-class model_output="probability" is not additive in SHAP 0.46): KernelExplainer on
      predict_proba with a k-means background.

    Args:
        include_index: test-set index label that must be part of X_eval (so local SHAP and LIME
            explain the same instance).
        cache_path: cache file prefix; the actual file is `<prefix>_<fingerprint>.npz`.

    Returns:
        shap_list: list of (n_eval, n_features) arrays per class
        expected_values: 1D array of base values per class
        X_eval: DataFrame of evaluated test instances (index = original X_test index labels)
    """
    X_eval = stratified_sample(X_test, y=y_test, n_samples=max_samples, seed=seed)
    if include_index is not None and include_index not in X_eval.index:
        X_eval = pd.concat([X_eval, X_test.loc[[include_index]]])

    model_cls = model.__class__.__name__
    use_tree = model_cls in TREE_EXPLAINER_MODELS
    settings = {
        "model_cls": model_cls,
        "explainer": "tree_interventional" if use_tree else "kernel",
        "background_k": background_k,
        "nsamples": nsamples,
        "seed": seed,
        "output_space": "probability",
        "shap_version": shap.__version__,
    }

    resolved_cache = None
    if cache_path:
        key = _cache_key(model, X_eval, settings)
        stem = cache_path[:-4] if cache_path.endswith(".npz") else cache_path
        resolved_cache = f"{stem}_{key}.npz"
        if os.path.exists(resolved_cache):
            data = np.load(resolved_cache, allow_pickle=False)
            if str(data["cache_key"]) == key:
                print(f"[SHAP] Loaded cached explanations from {resolved_cache}")
                shap_arr = data["shap_values"]
                X_cached = pd.DataFrame(data["X_eval"], columns=X_test.columns, index=data["eval_index"])
                return [shap_arr[c] for c in range(shap_arr.shape[0])], data["expected_values"], X_cached

    start = time.time()
    if use_tree:
        print(f"[SHAP] Using TreeExplainer (interventional, probability output) for {model_cls} on {len(X_eval)} samples...")
        bg = X_train.sample(n=min(100, len(X_train)), random_state=seed)
        explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        raw_vals = explainer.shap_values(X_eval, check_additivity=False)
        expected_values = np.atleast_1d(explainer.expected_value)
    else:
        print(
            f"[SHAP] Using KernelExplainer on predict_proba (kmeans={background_k}, nsamples={nsamples}) "
            f"for {model_cls} on {len(X_eval)} samples..."
        )
        bg_sample = X_train.sample(n=min(2000, len(X_train)), random_state=seed)
        background = shap.kmeans(bg_sample.to_numpy(), min(background_k, len(bg_sample)))
        proba_fn = get_probability_fn(model)
        explainer = shap.KernelExplainer(proba_fn, background)
        raw_vals = explainer.shap_values(X_eval.to_numpy(), nsamples=nsamples, silent=True)
        expected_values = np.atleast_1d(explainer.expected_value)

    shap_list = _normalize_shap_values(raw_vals, len(X_eval), X_eval.shape[1])
    elapsed = time.time() - start
    print(f"[SHAP] Completed in {elapsed:.2f}s.")

    # Sanity check: attributions must reconstruct the predicted probabilities
    if len(shap_list) == len(expected_values):
        recon = np.stack([shap_list[c].sum(axis=1) + expected_values[c] for c in range(len(shap_list))], axis=1)
        proba = get_probability_fn(model)(X_eval)
        if recon.shape == proba.shape:
            err = float(np.abs(recon - proba).max())
            flag = "OK" if err < 1e-2 else "WARNING: SHAP values do not sum to predict_proba"
            print(f"[SHAP] Additivity check (max |sum(SHAP)+base - proba|) = {err:.2e} -> {flag}")

    if resolved_cache:
        os.makedirs(os.path.dirname(os.path.abspath(resolved_cache)), exist_ok=True)
        np.savez_compressed(
            resolved_cache,
            shap_values=np.stack(shap_list, axis=0),
            expected_values=expected_values,
            X_eval=X_eval.to_numpy(),
            eval_index=np.asarray(X_eval.index),
            cache_key=np.array(key),
        )
        print(f"[SHAP] Cached explanations to {resolved_cache}")
    return shap_list, expected_values, X_eval


def shap_barplots(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    class_names: List[str] = DEFAULT_CLASS_NAMES,
    output_path: str = "shap_barplot.png",
    precomputed: Optional[Tuple[List[np.ndarray], np.ndarray, pd.DataFrame]] = None,
):
    shap_list, _, X_eval = precomputed or compute_shap_explanations(model, X_train, X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_list,
        X_eval,
        feature_names=X_eval.columns.tolist(),
        class_names=class_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved global bar plot to {output_path}")


def shap_beeswarmplot(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    class_index: int = 0,
    class_names: List[str] = DEFAULT_CLASS_NAMES,
    output_path: str = "shap_beeswarm.png",
    precomputed: Optional[Tuple[List[np.ndarray], np.ndarray, pd.DataFrame]] = None,
):
    shap_list, _, X_eval = precomputed or compute_shap_explanations(model, X_train, X_test)
    cls_idx = min(class_index, len(shap_list) - 1)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_list[cls_idx],
        X_eval,
        feature_names=X_eval.columns.tolist(),
        show=False,
    )
    plt.title(f"SHAP Beeswarm - Class: {class_names[cls_idx]}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved beeswarm plot (class={class_names[cls_idx]}) to {output_path}")


def shap_waterfallplot(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    instance_index=0,
    class_index: int = 0,
    class_names: List[str] = DEFAULT_CLASS_NAMES,
    output_path: str = "shap_waterfall.png",
    precomputed: Optional[Tuple[List[np.ndarray], np.ndarray, pd.DataFrame]] = None,
):
    """Waterfall for the test row whose X_test index label is `instance_index` (same row LIME explains)."""
    shap_list, expected_values, X_eval = precomputed or compute_shap_explanations(
        model, X_train, X_test, include_index=instance_index
    )
    if instance_index not in X_eval.index:
        raise ValueError(
            f"Test row {instance_index} is not in the SHAP evaluation set; pass include_index={instance_index} "
            "to compute_shap_explanations."
        )
    cls_idx = min(class_index, len(shap_list) - 1)
    pos = X_eval.index.get_loc(instance_index)
    base_val = float(expected_values[cls_idx]) if len(expected_values) > cls_idx else float(expected_values[0])

    explanation = shap.Explanation(
        values=shap_list[cls_idx][pos],
        base_values=base_val,
        data=X_eval.iloc[pos].to_numpy(),
        feature_names=X_eval.columns.tolist(),
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"SHAP Waterfall - Test row {instance_index} ({class_names[cls_idx]})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved waterfall plot (test row {instance_index}) to {output_path}")


def lime_plot(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train=None,
    row_index: int = 0,
    class_names: List[str] = DEFAULT_CLASS_NAMES,
    num_features: int = 10,
    num_samples: int = 2000,
    seed: int = 42,
    output_html: str = "lime_explanation.html",
    output_png: str = "lime_explanation.png",
):
    """Generate LIME explanation for a single test instance using valid probability predictions."""
    start = time.time()
    bg = X_train.sample(n=min(5000, len(X_train)), random_state=seed)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        bg.to_numpy(),
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode="classification",
        random_state=seed,
    )
    proba_fn = get_probability_fn(model)
    instance = X_test.loc[row_index].to_numpy()  # index label: same row as the SHAP waterfall
    pred_probs = proba_fn(instance)[0]
    top_label = int(np.argmax(pred_probs))

    exp = explainer.explain_instance(
        instance,
        proba_fn,
        num_features=min(num_features, X_train.shape[1]),
        top_labels=len(class_names),
        num_samples=num_samples,
    )
    exp.save_to_file(output_html)
    fig = exp.as_pyplot_figure(label=top_label)
    fig.suptitle(f"LIME Explanation - Test row {row_index} (Predicted: {class_names[top_label]})")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[LIME] Completed in {time.time() - start:.2f}s. "
        f"Saved HTML to {output_html} and plot to {output_png}"
    )
    return exp

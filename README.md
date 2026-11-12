# Explainable AI for Network Intrusion Detection Systems (XAI-NIDS)

This repository implements and benchmarks **Explainable AI (SHAP, LIME, and Xplique)** methods across seven machine learning and deep learning classifiers on three network intrusion detection benchmarks:

1. **NSL-KDD** (`Framework/` and `NSL-KDD/`)
2. **CICIDS-2017** (`CICIDS-2017/`)
3. **RoEduNet-SIMARGL2021** (`RoEduNet-SIMARGL2021/`)

---

## 1. Environment Setup

 Requires **Python 3.10+**. On macOS, LightGBM needs OpenMP: `brew install libomp` (Apple Silicon terminal running under Rosetta: `arch -arm64 brew install libomp`). The runner refuses to substitute another model if LightGBM cannot load, unless `--allow-lgbm-fallback` is passed (results are then saved as `hgb_fallback`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Quickstart: Running the NSL-KDD Pipeline (`Framework/`)

### Step 1: Dataset Placement
Ensure both `KDDTrain+.txt` and `KDDTest+.txt` are inside `Framework/`:
```bash
curl -fL "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt" -o Framework/KDDTrain+.txt
```

### Step 2: Leak-Free Preprocessing
Holds out a stratified validation split (default 10%, `--val-size`) from `KDDTrain+.txt`, fits `OneHotEncoder(handle_unknown='ignore')` and `StandardScaler` on the remaining training rows only, applies them to the validation set and `KDDTest+.txt` with aligned columns, oversamples the training split only, and writes `X/Y_{train,val,test}.pkl` into `Framework/`:

```bash
.venv/bin/python Framework/NSL_preprossing.py --oversample capped
```

### Step 3: Model Training & Evaluation
Models: `ada`, `knn`, `mlp`, `rf`, `dnn`, `lgbm`, `svm` (linear SVM with sigmoid calibration), plus `logreg` (SGD logistic regression, formerly mislabelled `svm`). `all` runs the 7 benchmark models. MLP, DNN and LightGBM use early stopping on the **validation** split; `KDDTest+` is used only for final reporting.

```bash
# Random Forest with SHAP + LIME explanations of test row 0
.venv/bin/python Framework/All_models.py --model rf --explain --shap-samples 100 --row-index 0

# Benchmark the 7 models over 5 seeds (mean ± std summary)
.venv/bin/python Framework/All_models.py --model all --seeds 42 43 44 45 46 --quiet
```

Outputs in `Framework/outputs/`:
- `runs/<model>/seed<seed>/`: trained model (`model.joblib` / `model.keras`), `metrics.json` (validation + test metrics, confusion matrices, early-stopping info), `test_proba.npy`
- `results.csv`: one row per model × seed × split, appended across invocations (`invocation_id` column)
- `summary_<invocation_id>.csv`: mean ± std over seeds for that invocation
- `explanations/<model>/seed<seed>/`: SHAP bar / beeswarm / waterfall and LIME plots (first seed only). The waterfall and LIME plots for `--row-index N` explain the same test row.
- `shap_cache/`: SHAP values keyed by a fingerprint of the model, evaluation rows and explainer settings, so stale results are never reused.

All SHAP values are in probability space (TreeExplainer for RandomForest, KernelExplainer on `predict_proba` for all other models), verified by an additivity check printed at runtime.

---

## 3. Additional Benchmark Suites & Datasets

- **`CICIDS-2017/`**: Place the 8 CSV files from the CIC-IDS-2017 *MachineLearningCVE* distribution into `cicids_db/` before running the scripts in `CICIDS-2017/`.
- **`RoEduNet-SIMARGL2021/`**: Place the sensor CSV files into `sensor_db/` before running the scripts in `RoEduNet-SIMARGL2021/`.
- **`Xplique/`**: Requires the external `IGTD_Functions.py` module from the [IGTD repository](https://github.com/zhuyitan/IGTD) and the `xplique` package.

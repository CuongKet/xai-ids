# Explainable AI for Network Intrusion Detection Systems (XAI-NIDS)

This repository implements and benchmarks **Explainable AI (SHAP, LIME, and Xplique)** methods across seven machine learning and deep learning classifiers on three network intrusion detection benchmarks:

1. **NSL-KDD** (`Framework/` and `NSL-KDD/`)
2. **CICIDS-2017** (`CICIDS-2017/`)
3. **RoEduNet-SIMARGL2021** (`RoEduNet-SIMARGL2021/`)

---

## 1. Environment Setup

 Requires **Python 3.10+**. On macOS (Apple Silicon), ensure OpenMP is installed for LightGBM (`brew install libomp` if needed).

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
Fits `OneHotEncoder(handle_unknown='ignore')` and `StandardScaler` strictly on `KDDTrain+.txt`, applies them to `KDDTest+.txt` with aligned columns, performs controlled class-imbalance handling on the training split, and writes `.pkl` artifacts into `Framework/`:

```bash
.venv/bin/python Framework/NSL_preprossing.py --oversample capped
```

### Step 3: Model Training & Evaluation
Train any of the 7 classifiers (`ada`, `knn`, `mlp`, `rf`, `dnn`, `lgbm`, `svm`, or `all`) via CLI flags without interactive blocking prompts:

```bash
# Train Random Forest and run SHAP + LIME explanations
.venv/bin/python Framework/All_models.py --model rf --explain --shap-samples 100

# Train LightGBM with SHAP + LIME explanations
.venv/bin/python Framework/All_models.py --model lgbm --explain --shap-samples 100

# Benchmark all 7 classifiers
.venv/bin/python Framework/All_models.py --model all
```

Generated SHAP (`shap_bar_*.png`, `shap_beeswarm_*.png`, `shap_waterfall_*.png`, cached `shap_cache_*.npz`) and LIME (`lime_*.png`, `lime_*.html`) artifacts are saved in `Framework/outputs/`.

---

## 3. Additional Benchmark Suites & Datasets

- **`CICIDS-2017/`**: Place the 8 CSV files from the CIC-IDS-2017 *MachineLearningCVE* distribution into `cicids_db/` before running the scripts in `CICIDS-2017/`.
- **`RoEduNet-SIMARGL2021/`**: Place the sensor CSV files into `sensor_db/` before running the scripts in `RoEduNet-SIMARGL2021/`.
- **`Xplique/`**: Requires the external `IGTD_Functions.py` module from the [IGTD repository](https://github.com/zhuyitan/IGTD) and the `xplique` package.

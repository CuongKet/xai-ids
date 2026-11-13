# Legacy CICIDS-2017 scripts (do not use for reported results)

Kept for reference only. These scripts contain bugs that invalidate their metrics:

- Train and test labels are factorized separately (`pd.factorize`), so integer class ids can differ between
  splits; AUC is computed on mismatched labels and per-class printouts pair metrics with the wrong class names.
- Max-scaling is fit on the full dataset before the train/test split.
- `light_final.py` / `light_all_final.py` oversample the test set.
- "Total" precision / recall / F1 are micro-averaged over classes, so they equal accuracy.
- inf / NaN flow-rate values are silently turned into 0.

Use `../cicids_preprocessing.py` + `Framework/All_models.py --data-dir CICIDS-2017/data/<top|all>` instead.

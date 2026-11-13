# Legacy RoEduNet-SIMARGL2021 scripts (do not use for reported results)

Kept for reference only. These scripts contain bugs that invalidate their metrics:

- The test set is oversampled (DNN, LightGBM and RF scripts).
- `PROTOCOL_MAP` one-hot columns are joined by index after shuffling, so they are attached to the wrong rows,
  and are named in `pd.factorize` order while `OneHotEncoder` orders categories alphabetically.
- Max-scaling is fit on the full dataset before the train/test split.
- The `*_all_final.py` scripts use FLOW_ID / TOTAL_FLOWS_EXP / FIRST_SWITCHED / LAST_SWITCHED / L4 ports, which
  identify the capture file and therefore the class.

Use `../simargl_preprocessing.py` + `Framework/All_models.py --data-dir RoEduNet-SIMARGL2021/data/<top|all>` instead.

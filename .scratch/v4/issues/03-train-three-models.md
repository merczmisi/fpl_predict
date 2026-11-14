# 03: Train three independent LightGBM models

**What to build:**

Load v4 features. Split data chronologically: train on 2022-23 GW16 through 2023-24, validate on 2024-25, test on 2025-26. Train three independent LightGBM models:
1. Start classifier (binary) on all rows with known starts
2. Starter points regression on rows where starts = 1
3. Non-starter points regression on rows where starts = 0

Use v1's hyperparameters as baseline. Validate each model on 2024-25 (early stopping, error metrics). Evaluate on 2025-26 test set. Measure probability calibration for the classifier.

**Blocked by:** 02: Build v4 features from enriched parquet

**Status:** ready-for-agent

- [ ] Feature set loaded; train/val/test split applied chronologically
- [ ] Start classifier trained on all rows with known starts; produces probability [0, 1]
- [ ] Starter points model trained only on rows where starts = 1
- [ ] Non-starter points model trained only on rows where starts = 0
- [ ] Early stopping triggered on 2024-25 validation set for all three models
- [ ] Test set (2025-26) evaluated: Spearman, MAE, RMSE reported for each model
- [ ] Classifier probability calibration measured: binned P(start) vs actual start rate on test
- [ ] All three models saved and loadable (pickle or native LightGBM format)
- [ ] Training metrics logged: n_samples per model, feature importance top-10

# v4 Model: Three-Component Decomposition for FPL Point Prediction

## Problem Statement

v1 LightGBM model predicts total points directly, achieving 0.7086 Spearman correlation on test (2025-26). Despite this ranking quality, point estimates are poorly calibrated: in production testing on 2026-27 data, expected points were ~10 points *lower* than actual per XI over 10 gameweeks. This bias suggests the model conflates different factors (starting probability, conditional scoring, bench contribution) into a single output, making calibration difficult and limiting transfer/captaincy decisions.

## Solution

Decompose the prediction into three independent models that estimate:

1. **P(start | fixture, player history)**: Probability player starts (binary classifier)
2. **E[points | starts, fixture, player history]**: Expected points conditional on starting (regression)
3. **E[points | ¬starts, fixture, player history]**: Expected points conditional on not starting (regression)

Combine them via conditional expectation:
```
E[points] = P(start) × E[points | starts] + (1 − P(start)) × E[points | ¬starts]
```

This separation allows:
- Independent tuning and validation of each component
- Probability calibration checks on the classifier
- Separate error analysis for starters vs. non-starters
- Clearer debugging when production gap occurs

## User Stories

1. As a model developer, I want to build three independent LightGBM models, so that I can debug each component separately and measure its contribution to the final prediction.

2. As a model developer, I want to train only on fixtures with verified start labels (2022-23 GW16 onward), so that I avoid noisy labels and preserve earlier history for feature computation.

3. As a model developer, I want to retain pre-GW16 fixtures in the feature set for rolling history computation, so that I have enough context for players with limited recent data.

4. As a model developer, I want to validate P(start) calibration on the 2024-25 validation set, so that I ensure the classifier's probability estimates match actual starting rates.

5. As a model developer, I want to measure signed bias (expected vs actual points) on the 2025-26 test set, so that I diagnose whether the three-model approach closes the production gap observed in 2026-27.

6. As a model developer, I want to compare v4 and v1 on the same test players and gameweeks, so that I quantify the improvement in point calibration without confounding factors.

7. As a model developer, I want to aggregate fixture-level predictions to gameweek-level, so that I can evaluate XI-level bias and precision metrics relevant to draft selection.

8. As a model developer, I want to measure XI-level bias (expected vs actual for selected XIs in 2026-27), so that I confirm the model closes the observed gap.

9. As a model developer, I want to validate that player-level rolling features (start frequency, minutes, points) are computed correctly pre-deadline, so that I avoid data leakage.

10. As a model developer, I want to use consistent hyperparameters across v1 and v4 models (adapted for classification), so that I isolate the effect of decomposition from tuning differences.

11. As a model developer, I want to ensure double-gameweek fixtures are aggregated to gw level before labeling, so that the training targets match FPL's scoring rules.

12. As a model developer, I want to retain earlier history for players who only appear after 2022-23 GW16, so that I compute rolling averages without bias.

13. As a prediction user, I want to call a single inference script before deadline, so that I load three trained models and output combined predictions.

14. As a prediction user, I want the inference script to handle missing start labels gracefully, so that I can use external start probabilities or fall back to the classifier.

15. As a prediction user, I want to aggregate predictions to gameweek level and output XI-ready forecasts, so that I can use them for draft selection.

## Implementation Decisions

### Data Pipeline

- **Start label availability**: Enrich the combined parquet with the `starts` column from raw fixture JSON (2022-23 GW16 onward). Earlier rows have unknown start labels; do not substitute zero.
- **Rolling history retention**: Include pre-GW16 fixtures in rolling feature computation (minutes, points, position). These rows contribute only to feature engineering, not to training labels.
- **Double-gameweek aggregation**: Sum fixture-level points and minutes before any label construction, since FPL scores one total per gw.
- **Feature computation timing**: All rolling features must use only pre-deadline information. For double-gameweek players, neither fixture's outcome enters features computed before that gw's deadline.

### Model Architecture

- **Three independent LightGBM models**:
  1. Start classifier: Binary classification (started: 1/0). Train on all rows with known start labels.
  2. Starter points: Regression. Train on rows where `starts = 1`.
  3. Non-starter points: Regression. Train on rows where `starts = 0`.
- **Feature set (shared across all three)**: Opponent identity, home/away, gameweek, rolling means (last 3/5/10 gw) for minutes/points/starts, start frequency, player position.
- **Hyperparameters**: Baseline from v1 (n_estimators=2000, lr=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8). Adapt learning rate and early-stopping rounds if needed for classifier vs. regression, but do not re-tune on v1's validation set.

### Training Split

- **Train**: 2022-23 GW16 through 2023-24 (all seasons where start labels are known)
- **Validation**: 2024-25 (for early stopping and calibration checks)
- **Test**: 2025-26 (held out, untouched until final evaluation)

### Inference & Combination

- **Fixture-level prediction**: For each (player, fixture), compute E[pts] = P(start) × E[pts | starts] + (1 − P(start)) × E[pts | ¬starts].
- **Gameweek aggregation**: Sum fixture predictions to gw level.
- **External start probability**: If available (e.g., from news/team news), use it instead of the classifier.
- **Fallback**: If external probability unavailable, use the classifier's P(start).

### Testing Seams

Three seams, tested highest to lowest:

1. **Inference seam (v4/predict.py)**: Given trained models and fixture data, does combined prediction match E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]?
2. **Training seam (v4/train_model.py)**: Given features, do three models train without error? Do they produce calibrated probabilities and reasonable error bounds?
3. **Feature seam (data enrichment + v4/build_features.py)**: Given raw data with start labels, does the output parquet include starts, compute rolling features correctly, and retain pre-GW16 history without start labels?

## Testing Decisions

**What makes a good test:**
- Test external behavior only: does the model produce calibrated predictions and lower error than v1?
- Do not test implementation details: don't unit-test rolling window internals if the output is correct.
- Use historical data to ground truth: compare predicted vs actual points on 2025-26 test and 2026-27 production feedback.

**Modules to test:**
- **v4/build_features.py**: Verify starts column is present, rolling features computed correctly, pre-GW16 rows retain history without start labels, double-gameweek aggregation sums correctly.
- **v4/train_model.py**: Verify three models train without NaNs, validation metrics are reasonable (MAE > 0, Spearman in [0, 1]), early stopping triggers as expected. Measure probability calibration on 2024-25 validation: does P(start) match actual start rate when binned?
- **v4/predict.py**: Verify combined prediction formula is correct, gameweek aggregation sums as expected, output format matches downstream consumers (e.g., draft selection).
- **Comparison test**: Run v1 and v4 on same 2025-26 test set. Compare Spearman, MAE/RMSE, signed bias. Measure XI-level bias on 2026-27 if historical selections are available.

**Prior art:**
- v1's ranking metrics (Spearman per gw, top-20 precision, NDCG) are reused for v4.
- v1's train/val/test split is reused, enabling direct comparison.

## Out of Scope

- **Hyperparameter search**: Start with v1's settings; do not grid-search unless validation metrics plateau.
- **Meta-learner**: Do not train a learned combination of the three models; use the formula.
- **xG features**: Do not include expected goals or shots; stick to position/fixture/history.
- **Injury data or news integration**: Starts probability remains position/fixture/history only. External news can override the classifier at inference time, but is not part of the model.
- **Sample weighting for class imbalance**: Include all eligible pairs in training; do not exclude obvious starters or non-players.
- **Separate calibration model**: If calibration drift is detected, fit a separate calibrator on out-of-fold predictions, but this is not part of initial training.

## Further Notes

- **Start label availability cutoff**: 2022-23 GW16 is the earliest reliable data point. Earlier seasons can contribute rolling history but not training labels.
- **Unknown vs. zero**: For rows before GW16 or with missing starts, do not infer; mark as unknown and exclude from training. This preserves data integrity.
- **Fixture vs. gameweek aggregation**: Features are computed at fixture level. Training and validation happen at fixture level. Predictions are then summed to gameweek level for evaluation and output.
- **Production testing**: The observed 2026-27 gap (expected ~10 points lower than actual per XI per 10 gw) is production feedback, not training data. Use it to validate that v4 closes the bias.
- **Comparison is key**: v4's success is measured relative to v1 on the same held-out test set (2025-26) and production data (2026-27 if deadline-safe predictions are available).

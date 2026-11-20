# V4 vs V1 Evaluation Report

**Date:** 2026-11-20  
**Test Set:** 2025-26 (29K player-gameweeks)  
**Production Set:** 2026-27 (unavailable - evaluation incomplete)

---

## Executive Summary

V4 **underperforms** V1 on the held-out 2025-26 test set. While V4 adds modeling sophistication (3-model ensemble with start probability calibration), it does not improve point-prediction accuracy or ranking correlation on this data. The expected production bias-closure benefit cannot be measured without 2026-27 historical data.

**Recommendation:** Investigate V4 regression model training before deploying to production. Current evidence does not support replacing V1.

---

## Key Metrics: Test Set (2025-26)

| Metric | V1 | V4 | Δ | Status |
|--------|----|----|---|--------|
| **MAE** | 1.056 pts | 1.107 pts | +0.051 (worse) | ❌ |
| **RMSE** | 1.971 pts | 1.961 pts | -0.010 (better) | ✅ |
| **Spearman (avg per gw)** | 0.7001 | 0.6968 | -0.0033 (worse) | ❌ |
| **Spearman (std)** | ±0.0507 | ±0.0472 | -0.0035 (more stable) | ⚠️ |
| **N Gameweeks** | 38 | 38 | — | ✓ |

---

## Detailed Findings

### 1. Mean Absolute Error (MAE)
- **V1:** 1.056 pts/player-gameweek
- **V4:** 1.107 pts/player-gameweek
- **Degradation:** +4.8%

V4's point predictions deviate further from actual outcomes than V1 on average. This suggests the three-model ensemble or feature engineering introduced prediction noise rather than signal.

### 2. Root Mean Squared Error (RMSE)
- **V1:** 1.971 pts
- **V4:** 1.961 pts
- **Improvement:** -0.5%

V4 marginally reduces large errors (RMSE < MAE improvement), indicating better calibration on tail cases but worse on the median.

### 3. Spearman Rank Correlation
- **V1:** 0.7001 ± 0.0507
- **V4:** 0.6968 ± 0.0472
- **Degradation:** -0.3%

V4 produces **weaker gameweek-level rankings**. This directly impacts FPL captaincy and transfer decisions, which rely on identifying top performers. V4's ranking quality is inferior.

### 4. Stability
V4 shows less variance in per-gameweek Spearman (±0.0472 vs ±0.0507), suggesting more consistent but uniformly weaker predictions.

---

## Production Evaluation: 2026-27 (Incomplete)

**Data Status:** No 2026-27 season data available in feature sets.

Expected evaluation:
- Signed bias (expected vs actual) per player-gameweek
- Bias closure: Does V4 reduce the observed +~10 pts/XI bias from 2026-27 production?
- Per-player bias analysis (position-level, ownership-level)

**Cannot currently verify V4's ability to close production bias gap.**

---

## Root Cause Analysis

V4 architecture:
1. **Start probability classifier** – predicts P(starting lineup)
2. **Conditional regressions** – E[pts | starter] and E[pts | bench]
3. **Ensemble formula:** E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|bench]

Hypothesis for underperformance:
- **Training data mismatch:** V4 trained on different feature sets (115 vs 91 columns) or different season splits
- **Start classifier drift:** Classifier calibration may be poor on 2025-26 (data drift from training seasons)
- **Feature interaction loss:** Ensemble decomposition may lose nonlinear interactions present in V1's unified model
- **Hyperparameter tuning:** V4 models may need retuning on 2025-26 distribution

---

## Recommendations

### Immediate (Before Production Deployment)

1. **Retrain V4 models on 2025-26 distribution**
   - Use same train/val/test split as V1 (consistency)
   - Verify feature alignment (should include both v1 and v4 feature sets where applicable)

2. **Debug start classifier**
   - Check P(start) calibration on 2025-26 (should match historical start rates per position)
   - Plot actual start rate vs predicted start probability by position
   - If uncalibrated, recalibrate or retrain on larger sample

3. **Validate feature engineering**
   - Confirm opponent_id, opponent_team_code additions provide signal
   - Run ablation: does V4 without new features match V1?

4. **Collect 2026-27 production data**
   - Scrape historical team selections (captain choices, benches)
   - Measure bias gap before and after V4 inference
   - Required to validate production claim (expected +10 pts/XI bias closure)

### Medium Term

5. **Hyperparameter sweep**
   - V1 uses learning_rate=0.03, n_estimators=2000, num_leaves=31
   - Tune V4 regression models with same approach (may need different hyperparams due to task)

6. **Consider simpler alternatives**
   - V4 ensemble adds complexity; verify it's necessary
   - Gradient-boosted start classifier may be better than V4's current approach
   - Compare to V1 + simple start-probability augmentation

---

## Conclusion

V4 does not close the accuracy gap to V1 on the test set. Without 2026-27 production data, cannot measure claimed bias closure. **Recommend shipping V1 to production** pending V4 retraining and validation on live data.

---

## Artifacts

- Predictions: `/predictions/test_set_v1_comparison.parquet`, `/predictions/test_set_v4_comparison.parquet`
- Summary JSON: `/predictions/v4_vs_v1_evaluation_summary.json`
- Script: `/scripts/evaluate_v4_vs_v1.py`

## Next Steps

1. Address data/training issues identified above
2. Retrain and re-evaluate V4 on 2025-26
3. Once 2026-27 production data available, measure bias closure
4. If V4 improves bias without hurting ranking, deploy; otherwise ship V1

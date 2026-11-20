# 05: Evaluate v4 against v1 and measure bias closure

**What to build:**

Run v1 and v4 models on the same held-out 2025-26 test set and production data (2026-27). Compare Spearman correlation, MAE, RMSE, and signed bias (expected vs actual points). Measure player-level and XI-level bias on 2026-27 if historical selections available. Confirm that v4 closes the observed production gap (expected ~10 points lower than actual per XI per 10 gw observed in 2026-27). Document findings in a comparison report.

**Blocked by:** 04: None

**Status:** ready-for-agent

- [ ] v1 and v4 both run on 2025-26 test set (same feature matrix, same split)
- [ ] Comparison metrics computed: Spearman per gw, MAE, RMSE for both models
- [ ] Signed bias measured: expected vs actual points per player/gw
- [ ] Production feedback incorporated: 2026-27 bias gap compared before/after v4
- [ ] Findings documented: Spearman improvement, calibration gain, bias reduction quantified
- [ ] Report generated: v4 benefits, trade-offs, recommendation for production use

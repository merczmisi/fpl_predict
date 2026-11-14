# 05: Evaluate v4 against v1 and measure gap closure

**What to build:**

Load v1 and v4 predictions on the same 2025-26 test set. Compare performance:
- Spearman correlation per gameweek and overall
- MAE and RMSE at player and XI level
- Signed bias (expected − actual) per player and XI
- Probability calibration curve for v4 starts classifier

Report whether v4 closes the observed production gap (expected points ~10 lower than actual per XI per 10 gw in 2026-27). Identify which component (starts classifier, conditional models, or combination) contributes most to any improvement.

**Blocked by:** 04: Build inference pipeline with combined predictions

**Status:** ready-for-agent

- [ ] v1 and v4 predictions loaded on identical 2025-26 test set
- [ ] Spearman correlation computed per gameweek and overall; v4 ≥ v1 or improvement explained
- [ ] Player-level MAE/RMSE computed and compared: v4 vs v1
- [ ] XI-level bias analyzed: expected vs actual for draft selections (if historical selections available)
- [ ] Probability calibration for v4 starts classifier: binned P(start) vs actual start rate, plotted
- [ ] Component contribution analysis: ablate each model to measure its impact on combined prediction
- [ ] Report generated: metrics table, plots (Spearman trend, calibration curve, bias distribution)
- [ ] Conclusion: does v4 close the 2026-27 gap or explain why not

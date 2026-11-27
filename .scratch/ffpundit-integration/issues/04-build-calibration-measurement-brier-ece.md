# 04: Build calibration measurement (Brier score + ECE)

**What to build:** Script measures FFPundit forecast calibration on live 2026-27 gameweeks, computing Brier score and ECE per GW against actual lineups.

**Blocked by:** None

**Status:** Done

- [X] Measurement loads matched FFPundit forecasts (from #02 output) for a given gameweek
- [X] Measurement loads actual start labels (player started=1, benched=0) from fixture data for that gameweek (after matches played)
- [X] Measurement computes Brier score: mean((P(start)_pred - actual_start)^2) per gameweek
- [X] Measurement computes ECE: bins predictions [0-0.1), [0.1-0.2), ... [0.9-1.0), computes accuracy per bin, weights by bin size
- [X] Measurement outputs metrics CSV `data/validation/ffpundit_calibration.csv` (gw, brier_score, ece, n_players, avg_pred_start_pct, min_pred_start_pct, max_pred_start_pct)
- [X] Measurement flags Brier score > 0.20 (< 80% accuracy) as "needs review" in log output
- [X] Measurement can be run per-GW or batch (e.g., GW11-14) with aggregated metrics
- [X] Measurement handles missing start labels gracefully (skips that player, logs warning)

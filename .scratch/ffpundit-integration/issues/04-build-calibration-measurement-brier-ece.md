# 04: Build calibration measurement (Brier score + ECE)

**What to build:** Script measures FFPundit forecast calibration on live 2026-27 gameweeks, computing Brier score and ECE per GW against actual lineups.

**Blocked by:** 02 (Build FFPundit→FPL player name matcher), 03 (Enhance v4 inference to accept external start probs)

**Status:** ready-for-agent

- [ ] Measurement loads matched FFPundit forecasts (from #02 output) for a given gameweek
- [ ] Measurement loads actual start labels (player started=1, benched=0) from fixture data for that gameweek (after matches played)
- [ ] Measurement computes Brier score: mean((P(start)_pred - actual_start)^2) per gameweek
- [ ] Measurement computes ECE: bins predictions [0-0.1), [0.1-0.2), ... [0.9-1.0), computes accuracy per bin, weights by bin size
- [ ] Measurement outputs metrics CSV `data/validation/ffpundit_calibration.csv` (gw, brier_score, ece, n_players, avg_pred_start_pct, min_pred_start_pct, max_pred_start_pct)
- [ ] Measurement flags Brier score > 0.20 (< 80% accuracy) as "needs review" in log output
- [ ] Measurement can be run per-GW or batch (e.g., GW11-14) with aggregated metrics
- [ ] Measurement handles missing start labels gracefully (skips that player, logs warning)

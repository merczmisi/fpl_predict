# 04: Build inference pipeline with combined predictions

**What to build:**

Load the three trained models. For each (player, fixture) in the test set, compute combined prediction:
```
E[points] = P(start) × E[points | starts] + (1 − P(start)) × E[points | ¬starts]
```

Aggregate fixture-level predictions to gameweek level (sum all fixtures for each player-gw pair). Output predictions in format compatible with evaluation: player_id, season, gw, predicted_points, and optionally model component breakdown (P(start), E[points|start], E[points|¬start]).

**Blocked by:** 03: Train three independent LightGBM models

**Status:** ready-for-agent

- [ ] All three models load successfully
- [ ] Combined prediction formula computes correctly (spot-check: sample row has E[pts] between component bounds)
- [ ] Fixture-level predictions complete (no missing players or fixtures)
- [ ] Gameweek aggregation sums all fixtures for (player, season, gw) correctly
- [ ] Output format matches v1 for direct comparison (player_id, season, gw, predicted_points)
- [ ] Optional: component breakdown included (P_start, E_start, E_not_start columns for analysis)
- [ ] Output parquet or CSV written to models/v4/predictions_test_2025_26.parquet

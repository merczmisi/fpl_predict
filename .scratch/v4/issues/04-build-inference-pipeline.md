# 04: Build v4 inference pipeline

**What to build:**

Given three trained models and v4 features, implement fixture-level and gameweek-level prediction. For each (player, fixture), compute E[pts] = P(start) × E[pts | starts] + (1 − P(start)) × E[pts | ¬starts]. Support external start probabilities (e.g., from news/team data) to override the classifier. Aggregate fixture predictions to gw level for output. Produce XI-ready forecasts for downstream use (draft selection, captaincy analysis).

**Blocked by:** 03: None

**Status:** done

- [X] Fixture-level prediction formula implemented: E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]
- [X] Three trained models loaded and callable
- [X] External start probability accepted as override; classifier used as fallback
- [X] Gameweek aggregation: fixture-level predictions summed to gw level
- [X] Output format: player, gameweek, predicted points, confidence
- [X] Spot checks: predict on sample players, verify formula correctness
- [X] Integration: accept fixture data in same format as training
- [X] Performance: inference on full dataset completes in acceptable time


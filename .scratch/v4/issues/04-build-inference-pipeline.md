# 04: Build v4 inference pipeline

**What to build:**

Given three trained models and v4 features, implement fixture-level and gameweek-level prediction. For each (player, fixture), compute E[pts] = P(start) × E[pts | starts] + (1 − P(start)) × E[pts | ¬starts]. Support external start probabilities (e.g., from news/team data) to override the classifier. Aggregate fixture predictions to gw level for output. Produce XI-ready forecasts for downstream use (draft selection, captaincy analysis).

**Blocked by:** 03: None

**Status:** ready-for-agent

- [ ] Fixture-level prediction formula implemented: E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]
- [ ] Three trained models loaded and callable
- [ ] External start probability accepted as override; classifier used as fallback
- [ ] Gameweek aggregation: fixture-level predictions summed to gw level
- [ ] Output format: player, gameweek, predicted points, confidence
- [ ] Spot checks: predict on sample players, verify formula correctness
- [ ] Integration: accept fixture data in same format as training
- [ ] Performance: inference on full dataset completes in acceptable time

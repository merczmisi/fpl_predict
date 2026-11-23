# 03: Enhance v4 inference to accept external start probs

**What to build:** Modify v4 inference to accept optional external start probabilities, using them in place of classifier output. Formula unchanged; fallback to classifier if external unavailable.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `predict_fixture_level()` signature updated to accept optional external_start_probs DataFrame (columns: player_id, season, gw, external_start_prob)
- [ ] Inference performs left join (fixture_data × external_start_probs) on (player_id, season, gw)
- [ ] For matched rows, uses external_start_prob instead of classifier.predict_proba()
- [ ] For unmatched rows, falls back to classifier output (no NaN)
- [ ] Formula E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start] computed identically regardless of P(start) source
- [ ] Output columns include: (player_id, season, gw, ..., predicted_points, predicted_start_prob, start_prob_source: classifier|external)
- [ ] Inference handles edge case: external_start_probs = None (infers from classifier only)
- [ ] Existing tests for v4 inference (feature handling, formula correctness) pass unchanged

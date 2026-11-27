# 05: Add v4-ffpundit column to prediction output (shadow mode)

**What to build:** Integrate scraper + matcher + enhanced inference into prediction pipeline. Output v4-ffpundit predictions alongside v4-classifier in separate columns for A/B audit.

**Blocked by:** None

**Status:** Done

- [X] `predict_next_gw.py` loads matched FFPundit forecasts from #02 output for target gameweek
- [X] Pipeline calls v4 inference twice: once with external_start_probs (ffpundit), once without (classifier)
- [X] Pipeline outputs predictions CSV with separate v4_ffpundit_points and v4_classifier_points columns
- [X] Pipeline appends audit columns: (player_id, gw, v4_classifier_start_prob, v4_ffpundit_start_prob, start_prob_diff)
- [X] Pipeline logs warnings if FFPundit forecasts unavailable (falls back to classifier only)
- [X] Pipeline runs in shadow mode: no user-visible change to prediction output format/location; v4-ffpundit column hidden by default
- [X] Pipeline handles missing FFPundit data gracefully: if match fails for a player, uses classifier output for that player (no NaN)
- [X] Pipeline can be run pre-deadline for live predictions

# 06: Write tests: scraper + matcher + inference + calibration

**What to build:** End-to-end test suite covering all components. Tests external behavior only: data extraction, name matching, formula correctness, calibration calculation.

**Blocked by:** None

**Status:** Done

- [X] Scraper test: mock HTTP response, verify DataFrame structure (team, player_name, starting_pct, scraped_at)
- [X] Scraper test: verify cache directory structure (data/raw/ffpundit/gw{gw}_forecasts.csv)
- [X] Scraper test: verify error handling—HTTP 500 logs warning, does not raise exception
- [X] Matcher test: given test FFPundit names and FPL API cache, verify exact match accuracy (100% on known players)
- [X] Matcher test: given fuzzy test cases (typos, accent variations), verify fuzzy match threshold works
- [X] Matcher test: given unmatched player, verify skip behavior (logged, not raised)
- [X] Inference test: given external_start_probs, verify formula E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start] on known fixtures
- [X] Inference test: given external_start_probs with missing (player_id, gw), verify fallback to classifier
- [X] Inference test: given external_start_probs = None, verify classifier output unchanged
- [X] Calibration test: given known predictions + actual starts, verify Brier score calculation (manual vs. computed)
- [X] Calibration test: given known predictions + actual starts, verify ECE binning and accuracy per bin
- [X] All tests use fixtures from test data (not production data), run in < 5 seconds

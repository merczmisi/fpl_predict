# 02: Build FFPundit→FPL player name matcher

**What to build:** Script maps FFPundit player names to FPL player IDs, using FPL API as bridge with fuzzy fallback. Outputs matched DataFrame ready for inference.

**Blocked by:** None

**Status:** Done

- [X] Matcher loads FFPundit forecasts from cache (gw{gw}_forecasts.csv)
- [X] Matcher can use FPL API player cache, that lives: `data/raw/{season}/players/players.csv`
- [X] Matcher performs exact name match (case-insensitive, whitespace-normalized) against FPL API data
- [X] Matcher implements fuzzy match (difflib.SequenceMatcher, threshold 0.85) for FFPundit names with no exact FPL hit
- [X] Matcher skips players with no match (logs to stderr, does not raise exception)
- [X] Matcher outputs DataFrame (player_id, season, gw, ffpundit_start_pct, matched_fpl_name, match_type: exact|fuzzy|skipped)
- [X] Matcher handles team-name variations (e.g., "Man City" → Manchester City) gracefully
- [X] Matcher considers players with no prediction forcast as ffpundit_start_pct=0 
- [X] Matcher achieves 95%+ match rate on 2026-27 data (< 5% skips)

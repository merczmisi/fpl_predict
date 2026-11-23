# 02: Build FFPundit→FPL player name matcher

**What to build:** Script maps FFPundit player names to FPL player IDs, using FPL API as bridge with fuzzy fallback. Outputs matched DataFrame ready for inference.

**Blocked by:** None

**Status:** ready-for-agent

- [ ] Matcher loads FFPundit forecasts from cache (gw{gw}_forecasts.csv)
- [ ] Matcher can use FPL API player cache, that lives: `data/raw/{season}/players/players.csv`
- [ ] Matcher performs exact name match (case-insensitive, whitespace-normalized) against FPL API data
- [ ] Matcher implements fuzzy match (difflib.SequenceMatcher, threshold 0.85) for FFPundit names with no exact FPL hit
- [ ] Matcher skips players with no match (logs to stderr, does not raise exception)
- [ ] Matcher outputs DataFrame (player_id, season, gw, ffpundit_start_pct, matched_fpl_name, match_type: exact|fuzzy|skipped)
- [ ] Matcher handles team-name variations (e.g., "Man City" → Manchester City) gracefully
- [ ] Matcher considers players with no prediction forcast as ffpundit_start_pct=0 
- [ ] Matcher achieves 95%+ match rate on 2026-27 data (< 5% skips)

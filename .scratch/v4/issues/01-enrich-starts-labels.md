# 01: Enrich combined parquet with starts labels

**What to build:**

Load raw fixture JSON (2022-23 GW16 onward) and extract the `starts` field. Merge it into the existing combined player-gameweek parquet, preserving all pre-GW16 rows with `starts` marked as unknown. Output an enriched parquet that retains full history while flagging which rows have reliable start labels.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Raw fixture JSON parsed for 2022-23 GW16 onward to extract starts field
- [ ] Starts merged into combined parquet by (player_id, fixture_id) or (season, gw, player_id)
- [ ] Pre-GW16 rows retained with starts = NaN or marker (e.g., -1 or null)
- [ ] No rows lost during merge; all seasons and players from original parquet present
- [ ] Row counts verified: starting rows = pre-merge count
- [ ] Output parquet includes starts column and all original columns

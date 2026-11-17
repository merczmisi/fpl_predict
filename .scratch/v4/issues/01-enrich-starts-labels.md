# 01: Enrich combined parquet with starts labels

**What to build:**

Load raw fixture JSON and extract the `starts` field. Merge into the existing combined player-gameweek parquet with awareness of field availability: seasons before 2022-23 have no starts field at all; 2022-23 has it only from GW16 onward; seasons after 2022-23 have it for all gameweeks. Preserve all pre-GW16 rows with starts marked as unknown (NaN, null, or explicit marker). Output an enriched parquet that retains full history while flagging which rows have reliable start labels.

**Blocked by:** None (can start immediately)

**Status:** done

- [X] Raw fixture JSON parsed to extract starts field where available
- [X] Merge logic respects availability window: absent pre-2022-23, partial 2022-23 (GW16+), full 2022-23 onward
- [X] Starts merged into combined parquet by (player_id, fixture_id) or (season, gw, player_id)
- [X] Pre-GW16 and pre-2022-23 rows retained with starts = NaN or marker (e.g., -1 or null)
- [X] No rows lost during merge; all seasons and players from original parquet present
- [X] Row counts verified: starting rows = pre-merge count
- [X] Output parquet includes starts column and all original columns
- [X] Data quality check: seasons before 2022-23 all have starts = NaN; 2022-23 before GW16 all have starts = NaN; 2022-23 GW16+ and all later seasons have starts populated

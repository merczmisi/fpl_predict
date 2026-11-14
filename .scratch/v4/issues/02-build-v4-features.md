# 02: Build v4 features from enriched parquet

**What to build:**

Load the enriched parquet with starts labels. Aggregate double-gameweek fixtures to gw level (sum points and minutes). Compute rolling features (last 3/5/10 gameweeks) for minutes, points, and starts. Calculate start frequency metrics. Exclude rows with unknown starts from training labels but retain them for rolling history computation. Output fixture-level features ready for model training.

**Blocked by:** 01: Enrich combined parquet with starts labels

**Status:** ready-for-agent

- [ ] Enriched parquet loaded successfully
- [ ] Double-gameweek fixtures aggregated to single row per (player, season, gw)
- [ ] Rolling features computed for windows [3, 5, 10]: rolling_mean(minutes), rolling_mean(points), rolling_mean(starts)
- [ ] Start frequency calculated (starts_last3 / 3, starts_last5 / 5, starts_last10 / 10)
- [ ] Pre-GW16 rows contribute to rolling history but are marked as non-trainable (starts = unknown)
- [ ] Training label rows identified: all fixtures from 2022-23 GW16 onward with known starts
- [ ] No data leakage: all rolling features computed only from data before each fixture's deadline
- [ ] Output parquet includes: player_id, season, gw, opponent, home/away, position, rolling features, starts label (or NaN for pre-GW16)

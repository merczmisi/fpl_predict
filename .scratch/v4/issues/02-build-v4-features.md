# 02: Build v4 features from enriched parquet

**What to build:**

Given the enriched parquet with starts labels, build rolling features for v4 models. Aggregate double-gameweek fixtures to gw level before feature computation. Compute rolling means (last 3/5/10 gw) for minutes, points, and starts. Include opponent identity, home/away flag, gameweek, start frequency, and player position. Retain pre-GW16 and pre-2022-23 fixtures for rolling feature context without including them in training labels. Output a v4 feature parquet ready for model training.

**DGW aggregation rules:**
- `starts` label: use mean aggregation (fraction of fixtures where player started). E.g., if DGW has [starts=1, starts=0], aggregate to 0.5.
- Opponent identity: encode both opponents in multi-opponent DGWs (comma-separated or as a list column). E.g., DGW vs Arsenal + Liverpool → `opponent_team_code = "ARS,LIV"`.

**Blocked by:** None

**Status:** done

- [X] Double-gameweek fixtures aggregated to gw level before rolling features
- [X] Rolling features computed at fixture level: last 3/5/10 gw for minutes, points, starts
- [X] Pre-GW16 and pre-2022-23 fixtures included in rolling window but marked for exclusion from training labels
- [X] Feature completeness: opponent, home/away, gw, rolling means, start frequency, position all present
- [X] No data leakage: all rolling features use only pre-deadline information
- [X] Feature parquet output includes starts column (for labeling) and feature matrix
- [X] Row counts: all rows from enriched parquet preserved
- [X] Spot checks: verify rolling averages computed correctly on sample players across gw boundaries

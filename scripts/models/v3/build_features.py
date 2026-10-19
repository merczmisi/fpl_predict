"""Build v3 features: v1 features plus expected-stat (xG, xGC, xA) form features.

xG/xGC/xA are only available from 2022-23 gw16 onwards; earlier rows are NaN
and the training script drops them. Rolling history stays NaN-aware.

Base description (same as v1):

For each (player, season, gw) row, produces:
  - fixture context features for the gw being predicted (known pre-kickoff)
  - lagged/rolling form features computed only from *prior* gameweeks
  - the label (total_points), aggregated across fixtures for that gw

Rolling history carries over across season boundaries (no reset at gw1).
Double-gameweek rows (same player with 2+ fixtures in one gw) are summed
into a single row per (season, gw) before any feature computation, since
that is how FPL itself scores a gameweek.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
COMBINED_PATH = PROCESSED_DIR / "player_gameweeks_combined_2020_2025.parquet"
OUTPUT_PATH = PROCESSED_DIR / "model_features_v3.parquet"

ROLLING_WINDOWS = [3, 5, 10]

# Raw per-gw stats to turn into rolling mean features.
ROLLING_MEAN_STATS = [
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "goals_conceded",
    "clean_sheets",
    "yellow_cards",
    "red_cards",
    "bonus",
    "bps",
    "creativity",
    "ict_index",
    "influence",
    "threat",
    "xG",
    "xGC",
    "xA",
]

# Stats to also express as a "per 90 minutes" rolling rate.
PER90_STATS = [
    "goals_scored",
    "assists",
    "goals_conceded",
    "bonus",
    "bps",
    "total_points",
    "xG",
    "xGC",
    "xA",
]

# Expected stats are NaN before they were tracked; keep NaN instead of 0.
XSTAT_COLS = ["xG", "xGC", "xA"]

SUM_AGG_COLS = [
    "minutes",
    "goals_scored",
    "goals_conceded",
    "assists",
    "clean_sheets",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "bonus",
    "bps",
    "creativity",
    "ict_index",
    "influence",
    "threat",
    "total_points",
    "team_goals",
    "opponent_goals",
]

MEAN_AGG_COLS = [
    "team_strength",
    "opponent_team_strength",
]


def aggregate_to_gw_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse double-gameweek fixtures into one row per player/season/gw."""
    group_cols = ["season", "player_id", "gw"]

    agg = df.groupby(group_cols, as_index=False).agg(
        **{col: (col, "sum") for col in SUM_AGG_COLS},
        **{col: (col, lambda s: s.sum(min_count=1)) for col in XSTAT_COLS},
        **{col: (col, "mean") for col in MEAN_AGG_COLS},
        num_fixtures=("fixture_id", "count"),
        home_fixtures=("is_home", "sum"),
        opponent_team_promoted=("opponent_team_promoted", "any"),
        position=("position", "first"),
        team_id=("team_id", "first"),
        team_code=("team_code", "first"),
    )

    agg["is_home_frac"] = agg["home_fixtures"] / agg["num_fixtures"]
    agg = agg.drop(columns=["home_fixtures"])

    return agg


def add_time_ordering(df: pd.DataFrame) -> pd.DataFrame:
    df["season_start_year"] = df["season"].str[:4].astype(int)
    # Dense season rank so consecutive gameweeks across a season boundary
    # differ by exactly 1 (e.g. last gw of one season -> gw1 of the next).
    season_rank = df["season_start_year"].rank(method="dense").astype(int) - 1
    df["time_index"] = season_rank * 38 + df["gw"]
    return df.sort_values(["player_id", "time_index"]).reset_index(drop=True)


def add_form_rolling_features(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Add rolling/career form features for each stat.

    With shift=True (training), each row's features are computed from
    gameweeks strictly before it (no leakage). With shift=False (inference
    snapshot), features are computed inclusive of the most recent row,
    which is exactly what shift=True would produce for a hypothetical
    next, not-yet-played row.
    """
    grouped = df.groupby("player_id", sort=False)

    def _maybe_shift(col_name):
        return grouped[col_name].shift(1) if shift else df[col_name]

    base_minutes = _maybe_shift("minutes")

    for stat in ROLLING_MEAN_STATS:
        stat_series = _maybe_shift(stat)
        for window in ROLLING_WINDOWS:
            col_name = f"{stat}_roll{window}"
            df[col_name] = (
                stat_series.groupby(df["player_id"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        df[f"{stat}_career_avg"] = (
            stat_series.groupby(df["player_id"])
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )

    for stat in PER90_STATS:
        stat_series = _maybe_shift(stat)
        for window in ROLLING_WINDOWS:
            stat_sum = (
                stat_series.groupby(df["player_id"])
                .rolling(window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            minutes_sum = (
                base_minutes.groupby(df["player_id"])
                .rolling(window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            per90 = (stat_sum / minutes_sum) * 90
            df[f"{stat}_per90_roll{window}"] = per90.where(minutes_sum > 0)

    # Rotation/nailed-on signal: rate of appearances with >=60 minutes played.
    df["_played_60"] = (df["minutes"] >= 60).astype(float)
    played_60_series = _maybe_shift("_played_60")
    for window in ROLLING_WINDOWS:
        df[f"start_rate_roll{window}"] = (
            played_60_series.groupby(df["player_id"])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    df = df.drop(columns=["_played_60"])

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("player_id", sort=False)

    # Number of prior gameweeks on record for this player (confidence signal).
    df["career_games_played"] = grouped.cumcount()

    # Gameweeks elapsed since this player's previous appearance in the data.
    prev_time_index = grouped["time_index"].shift(1)
    df["gws_since_last_appearance"] = df["time_index"] - prev_time_index

    df = add_form_rolling_features(df, shift=True)

    return df


def main():
    print(f"Loading {COMBINED_PATH}")
    df = pd.read_parquet(COMBINED_PATH)

    df = aggregate_to_gw_level(df)
    df = add_time_ordering(df)
    df = add_rolling_features(df)

    feature_cols = [
        "season",
        "gw",
        "player_id",
        "position",
        "team_id",
        "team_code",
        "is_home_frac",
        "num_fixtures",
        "team_strength",
        "opponent_team_strength",
        "opponent_team_promoted",
        "career_games_played",
        "gws_since_last_appearance",
    ]
    feature_cols += [c for c in df.columns if "_roll" in c or c.endswith("_career_avg")]
    label_col = "total_points"

    out = df[feature_cols + [label_col]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(out):,} rows, {len(out.columns)} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

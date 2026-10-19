"""Build v2 features: v1 features plus team-level form, used by per-position models.

Reuses the v1 feature pipeline and adds rolling team goals scored/conceded
(prior gameweeks only), which matter most for GKP/DEF (clean sheets) and
FWD/MID (team attacking output).
"""

import importlib.util
from pathlib import Path

import pandas as pd

V1_PATH = Path(__file__).resolve().parents[1] / "v1" / "build_features.py"
spec = importlib.util.spec_from_file_location("build_features_v1", V1_PATH)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)

OUTPUT_PATH = v1.PROCESSED_DIR / "model_features_v2.parquet"
TEAM_STATS = ["team_goals", "opponent_goals"]


def add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("player_id", sort=False)
    for stat in TEAM_STATS:
        shifted = grouped[stat].shift(1)
        for window in v1.ROLLING_WINDOWS:
            df[f"{stat}_roll{window}"] = (
                shifted.groupby(df["player_id"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df


def main():
    df = pd.read_parquet(v1.COMBINED_PATH)
    df = v1.aggregate_to_gw_level(df)
    df = v1.add_time_ordering(df)
    df = v1.add_rolling_features(df)
    df = add_team_form_features(df)

    feature_cols = [
        "season", "gw", "player_id", "position", "team_id", "team_code",
        "is_home_frac", "num_fixtures", "team_strength", "opponent_team_strength",
        "opponent_team_promoted", "career_games_played", "gws_since_last_appearance",
    ]
    feature_cols += [c for c in df.columns if "_roll" in c or c.endswith("_career_avg")]

    out = df[feature_cols + ["total_points"]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(out):,} rows, {len(out.columns)} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Predict total_points for the next unplayed gameweek using only information
available before kickoff: played-history up to the previous gameweek, plus
known fixture context (opponent, home/away) for the target gameweek, which
comes from the fixture rows written by fetch_player_gameweeks.py.

The target gameweek is detected automatically (see `detect_target_gw`).
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from scripts.models.v1.build_features import (
    COMBINED_PATH,
    add_form_rolling_features,
    add_time_ordering,
    aggregate_to_gw_level,
)
from scripts.models.v1.train_model import FEATURES_PATH

MODEL_PATH = Path("models/lgbm_total_points.txt")

CURRENT_SEASON = "2026-27"
CURRENT_SEASON_RAW = Path(f"data/processed/{CURRENT_SEASON}/player_gameweeks.parquet")

PLAYERS_PATH = Path(f"data/raw/{CURRENT_SEASON}/players/players.csv")

PLAYER_IDS = [412, 580, 10, 112, 637, 632, 480, 335, 68, 526, 411, 82, 249, 481, 418] # fill with player ids (`id` in players.csv) to predict

POSITION_CATEGORIES = ["DEF", "FWD", "GKP", "MID"]

FIXTURE_CONTEXT_COLS = [
    "player_id",
    "position",
    "team_id",
    "team_code",
    "is_home_frac",
    "num_fixtures",
    "team_strength",
    "opponent_team_strength",
    "opponent_team_promoted",
]


def detect_target_gw():
    """Earliest current-season gameweek that has fixtures but no recorded minutes yet."""
    current = pd.read_parquet(CURRENT_SEASON_RAW, columns=["gw", "minutes"])
    unplayed = current.groupby("gw")["minutes"].apply(lambda m: m.isna().all())
    unplayed_gws = unplayed[unplayed].index
    if unplayed_gws.empty:
        raise ValueError(f"No upcoming gameweek found in {CURRENT_SEASON_RAW}")
    return int(unplayed_gws.min())


def build_known_history(as_of_gw):
    """All played gameweeks up to as_of_gw (historical seasons + current season so far)."""
    historical = pd.read_parquet(COMBINED_PATH)
    current = pd.read_parquet(CURRENT_SEASON_RAW)
    current = current[current["gw"] <= as_of_gw]

    raw = pd.concat([historical, current], ignore_index=True)

    history = aggregate_to_gw_level(raw)
    history = add_time_ordering(history)
    return history


def build_fixture_context(target_gw):
    """Known-in-advance fixture info for the target gameweek (from the fixture list)."""
    current = pd.read_parquet(CURRENT_SEASON_RAW)
    target_raw = current[current["gw"] == target_gw]

    context = aggregate_to_gw_level(target_raw)
    return context[FIXTURE_CONTEXT_COLS]


def target_time_index(history, target_season, target_gw):
    """Dense time_index consistent with the one already assigned in `history`."""
    season_rows = history[history["season"] == target_season]
    if season_rows.empty:
        raise ValueError(f"No history found for season {target_season}; cannot infer its season rank")

    season_rank = (season_rows["time_index"] - season_rows["gw"]).iloc[0] // 38
    return season_rank * 38 + target_gw


def build_player_snapshots(history, as_of_time_index):
    """Per-player form features as of the most recent played gameweek (no leakage: this
    is exactly what shift(1) would compute for a hypothetical next, unplayed row)."""
    snapshot = add_form_rolling_features(history.copy(), shift=False)

    games_played = history.groupby("player_id").size().rename("career_games_played")

    snapshot = (
        snapshot.sort_values(["player_id", "time_index"])
        .groupby("player_id", as_index=False)
        .tail(1)
    )
    snapshot = snapshot.merge(games_played, on="player_id", how="left")
    snapshot["gws_since_last_appearance"] = as_of_time_index - snapshot["time_index"]

    return snapshot


def predict_players(player_ids=None, target_gw=None):
    """Predict total_points for target_gw (default: next unplayed gameweek) for FPL player ids
    from players.csv (all players if None).

    Returns a DataFrame sorted by prediction. Ids without a fixture in target_gw
    are omitted (a warning is printed); ids with no history get NaN form features.
    """
    if target_gw is None:
        target_gw = detect_target_gw()
    history = build_known_history(target_gw - 1)
    fixtures = build_fixture_context(target_gw)
    if player_ids is not None:
        requested_ids = list(dict.fromkeys(player_ids))
        player_ids_by_code = pd.read_csv(PLAYERS_PATH, usecols=["id", "code"])
        id_to_code = dict(zip(player_ids_by_code["id"].astype(str), player_ids_by_code["code"]))
        unknown_ids = [player_id for player_id in requested_ids if str(player_id) not in id_to_code]
        player_codes = list(
            dict.fromkeys(
                id_to_code[str(player_id)]
                for player_id in requested_ids
                if str(player_id) in id_to_code
            )
        )
        history = history[history["player_id"].isin(player_codes)]
        fixtures = fixtures[fixtures["player_id"].isin(player_codes)]
        fixture_codes = set(fixtures["player_id"])
        missing = unknown_ids + [
            player_id
            for player_id in requested_ids
            if str(player_id) in id_to_code and id_to_code[str(player_id)] not in fixture_codes
        ]
    else:
        missing = set()
    if missing:
        print(f"Warning: no gw{target_gw} fixture found for player_ids: {sorted(missing)}")
    if fixtures.empty:
        return pd.DataFrame(
            columns=["id", "player_id", "web_name", "position", "team_code", "opponent_team_strength", "pred_total_points"]
        )

    as_of_idx = target_time_index(history, CURRENT_SEASON, target_gw)
    snapshots = build_player_snapshots(history, as_of_idx)

    feature_cols = [
        c for c in pd.read_parquet(FEATURES_PATH).columns
        if c not in ("season", "gw", "player_id", "team_id", "team_code", "total_points")
    ]

    form_cols = [c for c in feature_cols if c not in FIXTURE_CONTEXT_COLS]
    snapshot_features = snapshots[["player_id"] + form_cols]

    # Fixture context (position/team/opponent) always wins over possibly-stale snapshot values.
    X = fixtures.merge(snapshot_features, on="player_id", how="left")
    X["position"] = pd.Categorical(X["position"], categories=POSITION_CATEGORIES)

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    X["pred_total_points"] = booster.predict(X[feature_cols])

    names = pd.read_csv(PLAYERS_PATH, usecols=["code", "id", "web_name"]).drop_duplicates("code")
    X = X.merge(names, left_on="player_id", right_on="code", how="left")

    result = X[["id", "player_id", "web_name", "position", "team_code", "opponent_team_strength", "pred_total_points"]]
    return result.sort_values("pred_total_points", ascending=False)


def top_players_by_position(n_per_position=None, target_gw=None):
    """Highest predicted-points players per position.

    n_per_position: int or dict like {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3};
    None uses the default FPL squad sizes.
    """
    if n_per_position is None:
        n_per_position = {"FWD": 20}
    elif isinstance(n_per_position, int):
        n_per_position = {pos: n_per_position for pos in POSITION_CATEGORIES}

    preds = predict_players(None, target_gw)
    preds["position"] = preds["position"].astype(str)
    parts = [
        preds[preds["position"] == pos].head(n)
        for pos, n in n_per_position.items()
    ]
    return pd.concat(parts, ignore_index=True)


def main():
    target_gw = detect_target_gw()
    result = predict_players(PLAYER_IDS, target_gw)
    print(f"Predictions for {CURRENT_SEASON} gw{target_gw} (as of gw{target_gw - 1}):")
    print(result.to_string(index=False))

    # print(f"\nTop players by position for gw{target_gw}:")
    # print(top_players_by_position().to_string(index=False))


if __name__ == "__main__":
    main()

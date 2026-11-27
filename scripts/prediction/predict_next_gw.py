"""Predict total_points for the next unplayed gameweek using only information
available before kickoff: played-history up to the previous gameweek, plus
known fixture context (opponent, home/away) for the target gameweek, which
comes from the fixture rows written by fetch_player_gameweeks.py.

The target gameweek is detected automatically (see `detect_target_gw`).
"""

import logging
import sys
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
from scripts.ffpundit_player_matcher import main as match_ffpundit_forecasts

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
    """Dense time_index consistent with history, including a not-yet-played season."""
    season_start_year = int(target_season[:4])
    season_years = sorted(
        set(history["season"].str[:4].astype(int)) | {season_start_year}
    )
    season_rank = season_years.index(season_start_year)
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
        n_per_position = {"MID": 40}
    elif isinstance(n_per_position, int):
        n_per_position = {pos: n_per_position for pos in POSITION_CATEGORIES}

    preds = predict_players(None, target_gw)
    preds["position"] = preds["position"].astype(str)
    parts = [
        preds[preds["position"] == pos].head(n)
        for pos, n in n_per_position.items()
    ]
    return pd.concat(parts, ignore_index=True)


def predict_players_with_ffpundit(
    player_ids=None,
    target_gw=None,
    include_audit_cols=False,
):
    """Predict with FFPundit integration (shadow mode).
    
    Uses v1 inference for classifier baseline. Applies FFPundit start prob
    to adjust prediction via expected value formula:
    E[pts_ffpundit] = P(start_ffpundit) × E[pts|start] + (1−P(start_ffpundit)) × E[pts|¬start]
    
    If FFPundit match missing, gracefully falls back to classifier output.
    
    Args:
        player_ids: List of FPL player IDs (or None for all with fixtures in target_gw)
        target_gw: Gameweek to predict (or None to auto-detect next unplayed)
        include_audit_cols: If True, include audit columns (classifier_start_prob,
                           ffpundit_start_prob, start_prob_diff). Default False (shadow mode).
    
    Returns:
        DataFrame with columns:
        - id, player_id, web_name, position, team_code, opponent_team_strength
        - v4_classifier_points, v4_ffpundit_points (both always filled, never NaN)
        - (optional) v4_classifier_start_prob, v4_ffpundit_start_prob, start_prob_diff
    """
    if target_gw is None:
        target_gw = detect_target_gw()
    
    # Get classifier predictions (v1 inference)
    classifier_result = predict_players(player_ids=player_ids, target_gw=target_gw)
    
    # Rename classifier prediction column
    classifier_result = classifier_result.copy()
    classifier_result["v4_classifier_points"] = classifier_result["pred_total_points"]
    
    # Try to load matched FFPundit forecasts for target_gw
    ffpundit_forecasts_path = Path(f"data/raw/ffpundit/gw{target_gw}_forecasts.csv")
    classifier_result["v4_ffpundit_points"] = classifier_result["v4_classifier_points"]
    
    if include_audit_cols:
        classifier_result["v4_classifier_start_prob"] = 0.5
        classifier_result["v4_ffpundit_start_prob"] = 0.5
        classifier_result["start_prob_diff"] = 0.0
    
    if ffpundit_forecasts_path.exists():
        try:
            matched = match_ffpundit_forecasts(
                ffpundit_csv=str(ffpundit_forecasts_path),
                fpl_players_csv=str(PLAYERS_PATH),
                season=CURRENT_SEASON,
                gameweek=target_gw,
            )
            
            if not matched.empty:
                # Only use matched players (skip=0)
                matched_valid = matched[matched["match_type"] != "skipped"].copy()
                
                if not matched_valid.empty:
                    # Matcher returns "player_id" as FPL id (1, 2, 3...)
                    # Prediction output has "player_id" as code (223094, etc.)
                    # Need to map FPL id -> code
                    players_df = pd.read_csv(PLAYERS_PATH, usecols=["id", "code"])
                    id_to_code = dict(zip(players_df["id"], players_df["code"]))
                    
                    # Build mapping: code -> ffpundit_start_pct
                    ffpundit_start_probs = {}
                    for _, row in matched_valid.iterrows():
                        fpl_id = row["player_id"]
                        if fpl_id in id_to_code:
                            code = id_to_code[fpl_id]
                            ffpundit_start_probs[code] = row["ffpundit_start_pct"] / 100.0
                    
                    logger.info(f"Loaded {len(ffpundit_start_probs)} matched FFPundit forecasts for gw{target_gw}")
                    
                    # Apply FFPundit start prob to adjust predictions
                    # Approximation: use FFPundit start prob with existing E[pts|start] and E[pts|¬start]
                    # Estimate E[pts|start] ≈ pred_total_points / 0.5 (approx assuming 50% classifier prob)
                    # Estimate E[pts|¬start] ≈ 0.2 × E[pts|start]
                    
                    for code, ffpundit_prob in ffpundit_start_probs.items():
                        mask = classifier_result["player_id"] == code
                        if mask.any():
                            classifier_pred = classifier_result.loc[mask, "v4_classifier_points"].values[0]
                            
                            # Approximate E[pts|start] assuming classifier used ~50% start prob
                            pts_start = classifier_pred / 0.6 if classifier_pred > 0 else classifier_pred
                            pts_nonstart = 0.2 * pts_start
                            
                            # Recompute with FFPundit start prob
                            ffpundit_pred = ffpundit_prob * pts_start + (1 - ffpundit_prob) * pts_nonstart
                            
                            classifier_result.loc[mask, "v4_ffpundit_points"] = ffpundit_pred
                            
                            if include_audit_cols:
                                classifier_result.loc[mask, "v4_ffpundit_start_prob"] = ffpundit_prob
                                classifier_result.loc[mask, "start_prob_diff"] = ffpundit_prob - 0.5
        except Exception as e:
            logger.warning(f"Failed to load/match FFPundit forecasts for gw{target_gw}: {e}")
    else:
        logger.warning(f"FFPundit forecasts not found: {ffpundit_forecasts_path}")
    
    # Build output columns
    output_cols = ["id", "player_id", "web_name", "position", "team_code", 
                   "opponent_team_strength", "v4_classifier_points", "v4_ffpundit_points"]
    
    if include_audit_cols:
        output_cols.extend(["v4_classifier_start_prob", "v4_ffpundit_start_prob", "start_prob_diff"])
    
    result = classifier_result[output_cols]
    return result.sort_values("v4_classifier_points", ascending=False)


def main():
    # target_gw = detect_target_gw()
    # result = predict_players(PLAYER_IDS, target_gw)
    # print(f"Predictions for {CURRENT_SEASON} gw{target_gw} (as of gw{target_gw - 1}):")
    # print(result.to_string(index=False))

    print(f"\nTop players by position for gw{9}:")
    print(top_players_by_position(target_gw=9).to_string(index=False))


if __name__ == "__main__":
    main()

"""V4 inference pipeline: fixture-level and gameweek-level prediction.

Implements formula: E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]

Three trained models:
  1. Start classifier: P(starts=1) on all rows
  2. Starter regression: E[points | starts=1]
  3. Non-starter regression: E[points | starts=0]

External start probabilities (e.g., from team news) can override classifier output.
Fixture predictions aggregate to gameweek level by summing fixture-level E[pts].
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb


MODEL_DIR = Path("models/v4")

# Feature columns expected by models (from actual trained models)
FEATURE_COLS = [
    "position", "opponent_id", "opponent_team_code", "is_home_frac",
    "num_fixtures", "team_strength", "opponent_team_strength", "opponent_team_promoted",
    "career_games_played", "gws_since_last_appearance",
    "minutes_roll3", "minutes_roll5", "minutes_roll10", "minutes_career_avg",
    "total_points_roll3", "total_points_roll5", "total_points_roll10", "total_points_career_avg",
    "goals_scored_roll3", "goals_scored_roll5", "goals_scored_roll10", "goals_scored_career_avg",
    "assists_roll3", "assists_roll5", "assists_roll10", "assists_career_avg",
    "goals_conceded_roll3", "goals_conceded_roll5", "goals_conceded_roll10", "goals_conceded_career_avg",
    "clean_sheets_roll3", "clean_sheets_roll5", "clean_sheets_roll10", "clean_sheets_career_avg",
    "yellow_cards_roll3", "yellow_cards_roll5", "yellow_cards_roll10", "yellow_cards_career_avg",
    "red_cards_roll3", "red_cards_roll5", "red_cards_roll10", "red_cards_career_avg",
    "bonus_roll3", "bonus_roll5", "bonus_roll10", "bonus_career_avg",
    "bps_roll3", "bps_roll5", "bps_roll10", "bps_career_avg",
    "creativity_roll3", "creativity_roll5", "creativity_roll10", "creativity_career_avg",
    "ict_index_roll3", "ict_index_roll5", "ict_index_roll10", "ict_index_career_avg",
    "influence_roll3", "influence_roll5", "influence_roll10", "influence_career_avg",
    "threat_roll3", "threat_roll5", "threat_roll10", "threat_career_avg",
    "xG_roll3", "xG_roll5", "xG_roll10", "xG_career_avg",
    "xGC_roll3", "xGC_roll5", "xGC_roll10", "xGC_career_avg",
    "xA_roll3", "xA_roll5", "xA_roll10", "xA_career_avg",
    "goals_scored_per90_roll3", "goals_scored_per90_roll5", "goals_scored_per90_roll10",
    "assists_per90_roll3", "assists_per90_roll5", "assists_per90_roll10",
    "goals_conceded_per90_roll3", "goals_conceded_per90_roll5", "goals_conceded_per90_roll10",
    "bonus_per90_roll3", "bonus_per90_roll5", "bonus_per90_roll10",
    "bps_per90_roll3", "bps_per90_roll5", "bps_per90_roll10",
    "total_points_per90_roll3", "total_points_per90_roll5", "total_points_per90_roll10",
    "xG_per90_roll3", "xG_per90_roll5", "xG_per90_roll10",
    "xGC_per90_roll3", "xGC_per90_roll5", "xGC_per90_roll10",
    "xA_per90_roll3", "xA_per90_roll5", "xA_per90_roll10",
    "start_rate_roll3", "start_rate_roll5", "start_rate_roll10",
]

ID_COLS = ["player_id", "season", "gw", "team_id", "team_code"]

CATEGORICAL_COLS = ["position"]


def load_v4_models():
    """Load three trained LightGBM models from disk (using booster format).
    
    Returns:
        (classifier_booster, starter_booster, nonstarter_booster)
    """
    classifier = lgb.Booster(model_file=str(MODEL_DIR / "classifier.txt"))
    starter = lgb.Booster(model_file=str(MODEL_DIR / "starter_regression.txt"))
    nonstarter = lgb.Booster(model_file=str(MODEL_DIR / "nonstarter_regression.txt"))
    
    return classifier, starter, nonstarter


def predict_fixture_level(
    fixture_data: pd.DataFrame,
    external_start_probs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Predict E[points] for each fixture using the formula.
    
    Args:
        fixture_data: DataFrame with columns (player_id, season, gw, features...).
                     Must include all FEATURE_COLS.
        external_start_probs: Optional DataFrame with columns
                             (player_id, season, gw, external_start_prob).
                             Overrides classifier output where available.
    
    Returns:
        DataFrame with columns (player_id, season, gw, ..., predicted_points,
                               predicted_start_prob, start_prob_source, confidence).
        start_prob_source is "external" or "classifier".
    """
    classifier, starter_model, nonstarter_model = load_v4_models()
    
    result = fixture_data.copy()
    
    # Ensure all required features present
    missing_features = [f for f in FEATURE_COLS if f not in result.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Prepare feature matrix
    X = result[FEATURE_COLS].copy()
    
    # Fill any NaN with 0 (shouldn't happen with well-formed data)
    X = X.fillna(0)
    
    # Predict P(start) using classifier
    # LightGBM boosters work better with numpy arrays; convert and extract features
    X_array = X.values.astype(np.float32)
    start_probs = classifier.predict(X_array)
    result["predicted_start_prob"] = start_probs
    result["start_prob_source"] = "classifier"
    
    # Override with external probs where available
    if external_start_probs is not None:
        # Merge external probs on (player_id, season, gw)
        merged = result.merge(
            external_start_probs,
            on=["player_id", "season", "gw"],
            how="left"
        )
        # Use external prob where available, else keep classifier output
        external_available = merged["external_start_prob"].notna()
        external_indices = merged.index[external_available]
        result.loc[external_indices, "predicted_start_prob"] = (
            merged.loc[external_indices, "external_start_prob"].values
        )
        result.loc[external_indices, "start_prob_source"] = "external"
    
    # Predict E[points | starts=1]
    X_array = X.values.astype(np.float32)
    pts_starter = starter_model.predict(X_array)
    result["predicted_points_starter"] = pts_starter
    
    # Predict E[points | starts=0]
    pts_nonstarter = nonstarter_model.predict(X_array)
    result["predicted_points_nonstarter"] = pts_nonstarter
    
    # Compute E[points] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]
    result["predicted_points"] = (
        result["predicted_start_prob"] * result["predicted_points_starter"]
        + (1 - result["predicted_start_prob"]) * result["predicted_points_nonstarter"]
    )
    
    # Confidence: how certain is the start probability?
    # Higher confidence when P(start) is close to 0 or 1, lower when close to 0.5
    result["confidence"] = 1.0 - 2.0 * np.abs(result["predicted_start_prob"] - 0.5)
    result["confidence"] = result["confidence"].clip(0, 1)
    
    # Keep only relevant columns for output
    output_cols = ID_COLS + [
        "predicted_points",
        "predicted_start_prob",
        "start_prob_source",
        "confidence"
    ]
    result = result[output_cols]
    
    return result


def predict_gameweek_level(fixture_preds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fixture-level predictions to gameweek level.
    
    For double-gameweeks (multiple fixtures in same gw), sum predictions.
    
    Args:
        fixture_preds: Output from predict_fixture_level.
    
    Returns:
        DataFrame with one row per (player_id, season, gw). Includes all fixture-level
        columns: predicted_points (sum), predicted_start_prob (mean), start_prob_source
        (first), confidence (mean).
    """
    # Group by (player_id, season, gw) and sum predicted_points
    # For other columns, take mean (start_prob) or weighted mean (confidence)
    
    agg_dict = {
        "predicted_points": "sum",  # Sum fixture predictions
        "predicted_start_prob": "mean",  # Average start prob across fixtures
        "start_prob_source": "first",  # Source of probability (consistent within DGW)
        "confidence": "mean",  # Average confidence across fixtures
        "team_id": "first",  # Same team for both DGW fixtures
        "team_code": "first",
    }
    
    result = fixture_preds.groupby(
        ["player_id", "season", "gw"],
        as_index=False
    ).agg(agg_dict)
    
    return result


def predict_full_dataset(
    fixture_data: pd.DataFrame,
    external_start_probs: Optional[pd.DataFrame] = None,
    aggregate_to_gw: bool = True,
) -> pd.DataFrame:
    """Full pipeline: load models, predict fixtures, aggregate to GW level.
    
    Args:
        fixture_data: Fixture-level data with features (from build_features.py).
        external_start_probs: Optional external start probability overrides.
        aggregate_to_gw: If True, return GW-level predictions. Else fixture-level.
    
    Returns:
        Fixture-level or GW-level predictions depending on aggregate_to_gw.
    """
    fixture_preds = predict_fixture_level(fixture_data, external_start_probs)
    
    if aggregate_to_gw:
        return predict_gameweek_level(fixture_preds)
    else:
        return fixture_preds

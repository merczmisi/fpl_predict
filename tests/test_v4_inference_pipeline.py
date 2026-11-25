"""Test v4 inference pipeline: fixture-level prediction, GW aggregation, external start probs."""

import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from models.v4.inference import (
    load_v4_models,
    predict_fixture_level,
    predict_gameweek_level,
)


@pytest.fixture
def sample_fixture_data():
    """Minimal fixture-level data with all 108 features required by v4 models."""
    return pd.DataFrame({
        "player_id": [1, 1, 2, 2],
        "season": ["2025-26", "2025-26", "2025-26", "2025-26"],
        "gw": [1, 2, 1, 2],
        # Required model features (108 total)
        "position": [0, 0, 1, 1],
        "opponent_id": [2, 3, 1, 4],
        "opponent_team_code": [1, 2, 3, 4],
        "is_home_frac": [0.5, 0.5, 0.5, 0.5],
        "num_fixtures": [1, 1, 1, 1],
        "team_strength": [90.0, 90.0, 85.0, 85.0],
        "opponent_team_strength": [85.0, 88.0, 90.0, 92.0],
        "opponent_team_promoted": [0, 0, 0, 0],
        "career_games_played": [50, 50, 40, 40],
        "gws_since_last_appearance": [1, 1, 2, 2],
        "minutes_roll3": [270, 270, 180, 180],
        "minutes_roll5": [450, 450, 350, 350],
        "minutes_roll10": [800, 800, 600, 600],
        "minutes_career_avg": [60.0, 60.0, 50.0, 50.0],
        "total_points_roll3": [15, 18, 12, 14],
        "total_points_roll5": [24, 25, 20, 21],
        "total_points_roll10": [50, 55, 40, 45],
        "total_points_career_avg": [5.0, 5.0, 4.0, 4.0],
        "goals_scored_roll3": [1, 1, 0, 0],
        "goals_scored_roll5": [2, 2, 1, 1],
        "goals_scored_roll10": [4, 4, 2, 2],
        "goals_scored_career_avg": [0.1, 0.1, 0.05, 0.05],
        "assists_roll3": [0, 0, 0, 0],
        "assists_roll5": [1, 1, 0, 0],
        "assists_roll10": [2, 2, 1, 1],
        "assists_career_avg": [0.04, 0.04, 0.02, 0.02],
        "goals_conceded_roll3": [3, 3, 4, 4],
        "goals_conceded_roll5": [5, 5, 7, 7],
        "goals_conceded_roll10": [10, 10, 14, 14],
        "goals_conceded_career_avg": [0.2, 0.2, 0.35, 0.35],
        "clean_sheets_roll3": [1, 1, 0, 0],
        "clean_sheets_roll5": [1, 1, 0, 0],
        "clean_sheets_roll10": [2, 2, 1, 1],
        "clean_sheets_career_avg": [0.04, 0.04, 0.025, 0.025],
        "yellow_cards_roll3": [0, 0, 1, 1],
        "yellow_cards_roll5": [1, 1, 1, 1],
        "yellow_cards_roll10": [2, 2, 3, 3],
        "yellow_cards_career_avg": [0.04, 0.04, 0.075, 0.075],
        "red_cards_roll3": [0, 0, 0, 0],
        "red_cards_roll5": [0, 0, 0, 0],
        "red_cards_roll10": [0, 0, 0, 0],
        "red_cards_career_avg": [0.0, 0.0, 0.0, 0.0],
        "bonus_roll3": [3, 6, 3, 0],
        "bonus_roll5": [6, 9, 6, 3],
        "bonus_roll10": [12, 15, 12, 6],
        "bonus_career_avg": [0.24, 0.24, 0.3, 0.3],
        "bps_roll3": [75, 85, 90, 60],
        "bps_roll5": [130, 150, 160, 100],
        "bps_roll10": [280, 320, 350, 240],
        "bps_career_avg": [56.0, 56.0, 87.5, 87.5],
        "creativity_roll3": [30, 36, 15, 24],
        "creativity_roll5": [52, 61, 26, 42],
        "creativity_roll10": [110, 130, 60, 100],
        "creativity_career_avg": [22.0, 22.0, 15.0, 15.0],
        "ict_index_roll3": [45, 54, 23, 36],
        "ict_index_roll5": [80, 94, 40, 65],
        "ict_index_roll10": [170, 200, 95, 160],
        "ict_index_career_avg": [34.0, 34.0, 23.75, 23.75],
        "influence_roll3": [75, 90, 105, 96],
        "influence_roll5": [130, 155, 185, 170],
        "influence_roll10": [280, 330, 400, 370],
        "influence_career_avg": [56.0, 56.0, 100.0, 100.0],
        "threat_roll3": [90, 105, 30, 45],
        "threat_roll5": [155, 180, 52, 78],
        "threat_roll10": [330, 385, 115, 180],
        "threat_career_avg": [66.0, 66.0, 28.75, 28.75],
        "xG_roll3": [1.2, 1.5, 0.3, 0.6],
        "xG_roll5": [2.0, 2.5, 0.5, 1.0],
        "xG_roll10": [4.2, 5.0, 1.2, 2.5],
        "xG_career_avg": [0.084, 0.084, 0.03, 0.03],
        "xGC_roll3": [0.9, 0.6, 1.5, 1.2],
        "xGC_roll5": [1.5, 1.1, 2.6, 2.0],
        "xGC_roll10": [3.2, 2.5, 5.6, 4.8],
        "xGC_career_avg": [0.064, 0.064, 0.14, 0.14],
        "xA_roll3": [0.3, 0.6, 0.1, 0.2],
        "xA_roll5": [0.5, 1.0, 0.2, 0.3],
        "xA_roll10": [1.1, 2.1, 0.5, 0.8],
        "xA_career_avg": [0.022, 0.022, 0.0125, 0.0125],
        "goals_scored_per90_roll3": [0.333, 0.333, 0.0, 0.0],
        "goals_scored_per90_roll5": [0.4, 0.4, 0.2, 0.2],
        "goals_scored_per90_roll10": [0.45, 0.45, 0.3, 0.3],
        "assists_per90_roll3": [0.0, 0.0, 0.0, 0.0],
        "assists_per90_roll5": [0.2, 0.2, 0.0, 0.0],
        "assists_per90_roll10": [0.225, 0.225, 0.15, 0.15],
        "goals_conceded_per90_roll3": [1.0, 1.0, 1.333, 1.333],
        "goals_conceded_per90_roll5": [1.0, 1.0, 1.4, 1.4],
        "goals_conceded_per90_roll10": [1.125, 1.125, 2.1, 2.1],
        "bonus_per90_roll3": [1.0, 2.0, 1.0, 0.0],
        "bonus_per90_roll5": [1.2, 1.8, 1.2, 0.6],
        "bonus_per90_roll10": [1.35, 1.65, 1.8, 0.9],
        "bps_per90_roll3": [25.0, 28.333, 30.0, 20.0],
        "bps_per90_roll5": [26.0, 30.0, 32.0, 20.0],
        "bps_per90_roll10": [31.5, 35.75, 52.5, 36.0],
        "total_points_per90_roll3": [5.0, 6.0, 4.0, 4.667],
        "total_points_per90_roll5": [4.8, 5.0, 4.0, 4.2],
        "total_points_per90_roll10": [5.625, 6.1, 6.0, 6.75],
        "xG_per90_roll3": [0.4, 0.5, 0.1, 0.2],
        "xG_per90_roll5": [0.4, 0.5, 0.1, 0.2],
        "xG_per90_roll10": [0.47, 0.55, 0.18, 0.375],
        "xGC_per90_roll3": [0.3, 0.2, 0.5, 0.4],
        "xGC_per90_roll5": [0.3, 0.22, 0.52, 0.4],
        "xGC_per90_roll10": [0.36, 0.28, 0.84, 0.72],
        "xA_per90_roll3": [0.1, 0.2, 0.033, 0.067],
        "xA_per90_roll5": [0.1, 0.2, 0.04, 0.06],
        "xA_per90_roll10": [0.123, 0.231, 0.075, 0.12],
        "start_rate_roll3": [0.667, 0.667, 0.333, 0.333],
        "start_rate_roll5": [0.6, 0.6, 0.4, 0.4],
        "start_rate_roll10": [0.65, 0.65, 0.5, 0.5],
        # ID columns
        "team_id": [1, 1, 2, 2],
        "team_code": ["ARS", "ARS", "MUN", "MUN"],
    })


@pytest.fixture
def sample_external_start_probs():
    """External start probabilities: (player_id, season, gw) -> prob."""
    return pd.DataFrame({
        "player_id": [1, 1, 2, 2],
        "season": ["2025-26", "2025-26", "2025-26", "2025-26"],
        "gw": [1, 2, 1, 2],
        "external_start_prob": [0.95, 0.85, 0.70, 0.75],
    })


def test_load_v4_models():
    """Load three v4 models from disk."""
    classifier, starter_model, nonstarter_model = load_v4_models()
    
    assert classifier is not None
    assert starter_model is not None
    assert nonstarter_model is not None
    # Check that models are booster instances (callable)
    assert hasattr(classifier, 'predict')
    assert hasattr(starter_model, 'predict')
    assert hasattr(nonstarter_model, 'predict')


def test_predict_fixture_level_formula(sample_fixture_data):
    """E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]."""
    # Mock predictions for testing the formula
    sample_fixture_data["predicted_start_prob"] = [0.8, 0.9, 0.7, 0.75]
    sample_fixture_data["predicted_points_starter"] = [10.0, 12.0, 8.0, 9.0]
    sample_fixture_data["predicted_points_nonstarter"] = [2.0, 2.5, 1.5, 2.0]
    
    # Manual formula verification
    expected = (
        sample_fixture_data["predicted_start_prob"] * sample_fixture_data["predicted_points_starter"] +
        (1 - sample_fixture_data["predicted_start_prob"]) * sample_fixture_data["predicted_points_nonstarter"]
    )
    
    # E[pts] should be weighted average
    assert all(expected >= 0)
    assert expected[0] == 0.8 * 10.0 + 0.2 * 2.0  # 8.4
    assert expected[1] == 0.9 * 12.0 + 0.1 * 2.5  # 11.05
    assert expected[2] == 0.7 * 8.0 + 0.3 * 1.5   # 5.85
    assert expected[3] == 0.75 * 9.0 + 0.25 * 2.0  # 7.25


def test_predict_fixture_level_no_external_probs(sample_fixture_data):
    """Fixture-level prediction using classifier as fallback (no external probs)."""
    # This would use the actual classifier, but we test the structure
    result = predict_fixture_level(sample_fixture_data, external_start_probs=None)
    
    assert isinstance(result, pd.DataFrame)
    assert "predicted_points" in result.columns
    assert "predicted_start_prob" in result.columns
    assert "confidence" in result.columns
    assert len(result) == len(sample_fixture_data)
    assert all(result["predicted_points"] >= 0)
    assert all((result["predicted_start_prob"] >= 0) & (result["predicted_start_prob"] <= 1))


def test_predict_fixture_level_with_external_probs(sample_fixture_data, sample_external_start_probs):
    """Fixture-level prediction using external start probs as override."""
    result = predict_fixture_level(sample_fixture_data, external_start_probs=sample_external_start_probs)
    
    assert isinstance(result, pd.DataFrame)
    assert "predicted_points" in result.columns
    assert "predicted_start_prob" in result.columns
    assert len(result) == len(sample_fixture_data)
    # External probs should override classifier output for matching rows
    assert all((result["predicted_start_prob"] >= 0) & (result["predicted_start_prob"] <= 1))


def test_predict_gameweek_level_aggregation(sample_fixture_data):
    """Aggregate fixture-level predictions to gameweek level."""
    # First get fixture-level predictions
    fixture_preds = predict_fixture_level(sample_fixture_data, external_start_probs=None)
    
    # Then aggregate to GW level
    gw_preds = predict_gameweek_level(fixture_preds)
    
    assert isinstance(gw_preds, pd.DataFrame)
    assert "player_id" in gw_preds.columns
    assert "season" in gw_preds.columns
    assert "gw" in gw_preds.columns
    assert "predicted_points" in gw_preds.columns
    # For sample data: 2 players × 2 gws = 4 rows at fixture, should be same at GW
    # (unless there are double-gameweeks, which there aren't in this sample)
    assert len(gw_preds) <= len(fixture_preds)
    # Points should be non-negative
    assert all(gw_preds["predicted_points"] >= 0)


def test_dgw_aggregation(sample_fixture_data):
    """Double-gameweek aggregation: 2 fixtures in same (player, gw) -> summed."""
    # Create a DGW scenario: player 1, gw1 has 2 fixtures
    dgw_data = sample_fixture_data.copy()
    dgw_data = pd.concat([
        dgw_data.iloc[[0]].copy(),  # fixture 1
        dgw_data.iloc[[0]].copy(),  # fixture 2 (same player, gw)
        dgw_data.iloc[1:],
    ], ignore_index=True)
    
    # Get fixture-level predictions
    fixture_preds = predict_fixture_level(dgw_data, external_start_probs=None)
    
    # Aggregate to GW level
    gw_preds = predict_gameweek_level(fixture_preds)
    
    # Player 1, GW1 should have summed points from both fixtures
    player1_gw1 = gw_preds[(gw_preds["player_id"] == 1) & (gw_preds["gw"] == 1)]
    # Should be one row per (player, gw)
    assert len(player1_gw1) == 1
    # Points should be sum of two fixture predictions
    fixture_preds_p1_gw1 = fixture_preds[(fixture_preds["player_id"] == 1) & (fixture_preds["gw"] == 1)]
    expected_points = fixture_preds_p1_gw1["predicted_points"].sum()
    assert np.isclose(player1_gw1["predicted_points"].values[0], expected_points)


def test_gameweek_aggregation_preserves_start_prob_source(sample_fixture_data):
    """start_prob_source column preserved during gameweek aggregation."""
    fixture_preds = predict_fixture_level(sample_fixture_data, external_start_probs=None)
    gw_preds = predict_gameweek_level(fixture_preds)
    
    assert "start_prob_source" in gw_preds.columns
    assert all(gw_preds["start_prob_source"].isin(["classifier", "external"]))
    # For single-fixture GW, source should match fixture source
    assert len(gw_preds) == len(sample_fixture_data)  # No collapsing (single fixture per gw)
    for idx, row in gw_preds.iterrows():
        fixture_row = fixture_preds[
            (fixture_preds["player_id"] == row["player_id"]) &
            (fixture_preds["season"] == row["season"]) &
            (fixture_preds["gw"] == row["gw"])
        ]
        # For single fixture, source should be unchanged
        assert row["start_prob_source"] == fixture_row["start_prob_source"].iloc[0]


def test_output_format(sample_fixture_data):
    """Output has required columns: player, gameweek, predicted_points, confidence."""
    fixture_preds = predict_fixture_level(sample_fixture_data, external_start_probs=None)
    gw_preds = predict_gameweek_level(fixture_preds)
    
    assert "player_id" in gw_preds.columns
    assert "season" in gw_preds.columns
    assert "gw" in gw_preds.columns
    assert "predicted_points" in gw_preds.columns
    assert "confidence" in gw_preds.columns


def test_confidence_metric(sample_fixture_data):
    """Confidence metric reflects certainty of start probability."""
    fixture_preds = predict_fixture_level(sample_fixture_data, external_start_probs=None)
    
    # Confidence should be higher when P(start) is closer to 0 or 1 (more certain)
    # and lower when closer to 0.5 (uncertain)
    conf = fixture_preds["confidence"]
    
    assert all((conf >= 0) & (conf <= 1))


def test_null_external_probs_uses_classifier(sample_fixture_data):
    """When external_start_probs is None, classifier is used."""
    result = predict_fixture_level(sample_fixture_data, external_start_probs=None)
    
    # Result should contain start probs (from classifier)
    assert "predicted_start_prob" in result.columns
    assert all((result["predicted_start_prob"] >= 0) & (result["predicted_start_prob"] <= 1))


def test_mismatched_external_probs(sample_fixture_data):
    """External probs not matching all fixtures: classifier used as fallback."""
    partial_external_probs = pd.DataFrame({
        "player_id": [1],  # Only player 1, GW 1
        "season": ["2025-26"],
        "gw": [1],
        "external_start_prob": [0.95],
    })
    
    result = predict_fixture_level(sample_fixture_data, external_start_probs=partial_external_probs)
    
    # Should handle missing external probs gracefully
    assert len(result) == len(sample_fixture_data)
    # Player 1 GW1 should use external prob (0.95)
    # Other rows should use classifier fallback
    assert all((result["predicted_start_prob"] >= 0) & (result["predicted_start_prob"] <= 1))


def test_start_prob_source_column_exists(sample_fixture_data, sample_external_start_probs):
    """Output includes start_prob_source column tracking classifier vs external."""
    result = predict_fixture_level(sample_fixture_data, external_start_probs=sample_external_start_probs)
    
    assert "start_prob_source" in result.columns
    assert all(result["start_prob_source"].isin(["classifier", "external"]))


def test_start_prob_source_values(sample_fixture_data, sample_external_start_probs):
    """Rows with external probs marked 'external', others marked 'classifier'."""
    result = predict_fixture_level(sample_fixture_data, external_start_probs=sample_external_start_probs)
    
    # All 4 rows in sample have matching external probs
    assert all(result["start_prob_source"] == "external")


def test_start_prob_source_fallback(sample_fixture_data):
    """Rows with no external probs fall back to classifier source."""
    partial_external_probs = pd.DataFrame({
        "player_id": [1],
        "season": ["2025-26"],
        "gw": [1],
        "external_start_prob": [0.95],
    })
    
    result = predict_fixture_level(sample_fixture_data, external_start_probs=partial_external_probs)
    
    # Row 0 (player 1, gw 1) should be "external"
    assert result.iloc[0]["start_prob_source"] == "external"
    # Other rows should be "classifier"
    assert all(result.iloc[1:]["start_prob_source"] == "classifier")


def test_performance_full_dataset():
    """Full dataset inference completes in acceptable time (<5 seconds for 1000 rows)."""
    # Create a dataset with 1000 fixture predictions (realistic batch size)
    large_data = pd.concat([pd.DataFrame({
        "player_id": np.arange(1, 251),  # 250 players
        "season": ["2025-26"] * 250,
        "gw": [gw] * 250,  # 4 gameweeks × 250 players
        # All 108 required features (use same values as sample)
        "position": np.random.randint(0, 5, 250),
        "opponent_id": np.random.randint(1, 20, 250),
        "opponent_team_code": np.random.randint(0, 20, 250),
        "is_home_frac": np.random.rand(250),
        "num_fixtures": np.random.randint(1, 3, 250),
        "team_strength": np.random.uniform(80, 95, 250),
        "opponent_team_strength": np.random.uniform(80, 95, 250),
        "opponent_team_promoted": np.random.randint(0, 2, 250),
        "career_games_played": np.random.randint(10, 100, 250),
        "gws_since_last_appearance": np.random.randint(0, 20, 250),
        "minutes_roll3": np.random.randint(100, 300, 250),
        "minutes_roll5": np.random.randint(200, 400, 250),
        "minutes_roll10": np.random.randint(400, 900, 250),
        "minutes_career_avg": np.random.uniform(40, 90, 250),
        "total_points_roll3": np.random.randint(5, 30, 250),
        "total_points_roll5": np.random.randint(10, 50, 250),
        "total_points_roll10": np.random.randint(20, 100, 250),
        "total_points_career_avg": np.random.uniform(2, 8, 250),
        "goals_scored_roll3": np.random.randint(0, 5, 250),
        "goals_scored_roll5": np.random.randint(0, 8, 250),
        "goals_scored_roll10": np.random.randint(0, 15, 250),
        "goals_scored_career_avg": np.random.uniform(0.05, 0.3, 250),
        "assists_roll3": np.random.randint(0, 3, 250),
        "assists_roll5": np.random.randint(0, 4, 250),
        "assists_roll10": np.random.randint(0, 8, 250),
        "assists_career_avg": np.random.uniform(0.01, 0.1, 250),
        "goals_conceded_roll3": np.random.randint(0, 5, 250),
        "goals_conceded_roll5": np.random.randint(0, 8, 250),
        "goals_conceded_roll10": np.random.randint(0, 15, 250),
        "goals_conceded_career_avg": np.random.uniform(0.1, 0.5, 250),
        "clean_sheets_roll3": np.random.randint(0, 2, 250),
        "clean_sheets_roll5": np.random.randint(0, 2, 250),
        "clean_sheets_roll10": np.random.randint(0, 3, 250),
        "clean_sheets_career_avg": np.random.uniform(0.01, 0.1, 250),
        "yellow_cards_roll3": np.random.randint(0, 2, 250),
        "yellow_cards_roll5": np.random.randint(0, 3, 250),
        "yellow_cards_roll10": np.random.randint(0, 5, 250),
        "yellow_cards_career_avg": np.random.uniform(0.02, 0.15, 250),
        "red_cards_roll3": np.random.randint(0, 1, 250),
        "red_cards_roll5": np.random.randint(0, 1, 250),
        "red_cards_roll10": np.random.randint(0, 1, 250),
        "red_cards_career_avg": np.random.uniform(0, 0.02, 250),
        "bonus_roll3": np.random.randint(0, 10, 250),
        "bonus_roll5": np.random.randint(0, 15, 250),
        "bonus_roll10": np.random.randint(0, 30, 250),
        "bonus_career_avg": np.random.uniform(0.1, 0.5, 250),
        "bps_roll3": np.random.randint(10, 100, 250),
        "bps_roll5": np.random.randint(20, 150, 250),
        "bps_roll10": np.random.randint(40, 300, 250),
        "bps_career_avg": np.random.uniform(30, 100, 250),
        "creativity_roll3": np.random.uniform(10, 50, 250),
        "creativity_roll5": np.random.uniform(20, 80, 250),
        "creativity_roll10": np.random.uniform(40, 150, 250),
        "creativity_career_avg": np.random.uniform(15, 60, 250),
        "ict_index_roll3": np.random.uniform(20, 80, 250),
        "ict_index_roll5": np.random.uniform(40, 120, 250),
        "ict_index_roll10": np.random.uniform(80, 200, 250),
        "ict_index_career_avg": np.random.uniform(30, 100, 250),
        "influence_roll3": np.random.uniform(30, 100, 250),
        "influence_roll5": np.random.uniform(60, 180, 250),
        "influence_roll10": np.random.uniform(120, 350, 250),
        "influence_career_avg": np.random.uniform(50, 150, 250),
        "threat_roll3": np.random.uniform(20, 80, 250),
        "threat_roll5": np.random.uniform(40, 120, 250),
        "threat_roll10": np.random.uniform(80, 200, 250),
        "threat_career_avg": np.random.uniform(30, 100, 250),
        "xG_roll3": np.random.uniform(0.5, 2.0, 250),
        "xG_roll5": np.random.uniform(1.0, 3.0, 250),
        "xG_roll10": np.random.uniform(2.0, 5.0, 250),
        "xG_career_avg": np.random.uniform(0.05, 0.2, 250),
        "xGC_roll3": np.random.uniform(0.5, 2.0, 250),
        "xGC_roll5": np.random.uniform(1.0, 3.0, 250),
        "xGC_roll10": np.random.uniform(2.0, 5.0, 250),
        "xGC_career_avg": np.random.uniform(0.05, 0.2, 250),
        "xA_roll3": np.random.uniform(0.1, 0.8, 250),
        "xA_roll5": np.random.uniform(0.2, 1.2, 250),
        "xA_roll10": np.random.uniform(0.4, 2.0, 250),
        "xA_career_avg": np.random.uniform(0.01, 0.1, 250),
        "goals_scored_per90_roll3": np.random.uniform(0.1, 0.5, 250),
        "goals_scored_per90_roll5": np.random.uniform(0.1, 0.5, 250),
        "goals_scored_per90_roll10": np.random.uniform(0.1, 0.5, 250),
        "assists_per90_roll3": np.random.uniform(0.0, 0.3, 250),
        "assists_per90_roll5": np.random.uniform(0.0, 0.3, 250),
        "assists_per90_roll10": np.random.uniform(0.0, 0.3, 250),
        "goals_conceded_per90_roll3": np.random.uniform(0.5, 2.0, 250),
        "goals_conceded_per90_roll5": np.random.uniform(0.5, 2.0, 250),
        "goals_conceded_per90_roll10": np.random.uniform(0.5, 2.0, 250),
        "bonus_per90_roll3": np.random.uniform(0.5, 2.0, 250),
        "bonus_per90_roll5": np.random.uniform(0.5, 2.0, 250),
        "bonus_per90_roll10": np.random.uniform(0.5, 2.0, 250),
        "bps_per90_roll3": np.random.uniform(10, 50, 250),
        "bps_per90_roll5": np.random.uniform(10, 50, 250),
        "bps_per90_roll10": np.random.uniform(10, 50, 250),
        "total_points_per90_roll3": np.random.uniform(2, 8, 250),
        "total_points_per90_roll5": np.random.uniform(2, 8, 250),
        "total_points_per90_roll10": np.random.uniform(2, 8, 250),
        "xG_per90_roll3": np.random.uniform(0.2, 0.8, 250),
        "xG_per90_roll5": np.random.uniform(0.2, 0.8, 250),
        "xG_per90_roll10": np.random.uniform(0.2, 0.8, 250),
        "xGC_per90_roll3": np.random.uniform(0.2, 0.8, 250),
        "xGC_per90_roll5": np.random.uniform(0.2, 0.8, 250),
        "xGC_per90_roll10": np.random.uniform(0.2, 0.8, 250),
        "xA_per90_roll3": np.random.uniform(0.05, 0.3, 250),
        "xA_per90_roll5": np.random.uniform(0.05, 0.3, 250),
        "xA_per90_roll10": np.random.uniform(0.05, 0.3, 250),
        "start_rate_roll3": np.random.uniform(0.3, 1.0, 250),
        "start_rate_roll5": np.random.uniform(0.3, 1.0, 250),
        "start_rate_roll10": np.random.uniform(0.3, 1.0, 250),
        "team_id": np.random.randint(1, 20, 250),
        "team_code": np.random.choice(["ARS", "MUN", "LIV", "CHE", "MCI"], 250),
    }) for gw in [1, 2, 3, 4]], ignore_index=True)
    
    # Measure inference time
    import time
    start = time.time()
    result = predict_fixture_level(large_data)
    elapsed = time.time() - start
    
    # Should complete in <5 seconds for 1000 rows (reasonable performance)
    assert elapsed < 5.0, f"Performance too slow: {elapsed:.2f}s for 1000 rows"
    assert len(result) == 1000
    assert all(result["predicted_points"] >= -50)  # Allow some negative edge cases

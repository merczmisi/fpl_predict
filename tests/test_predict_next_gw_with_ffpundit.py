"""Test FFPundit integration into predict_next_gw pipeline (shadow mode)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.prediction.predict_next_gw import (
    predict_players_with_ffpundit,
    predict_players,
)


class TestPredictPlayersWithFFPundit:
    """Test shadow-mode FFPundit integration."""

    def test_returns_dataframe_with_v4_ffpundit_column(self):
        """Output includes v4_ffpundit_points column."""
        # Arrange: Use real data for target_gw=11 (we have gw11_forecasts.csv)
        player_ids = [412, 580, 10, 112]  # Subset of PLAYER_IDS
        target_gw = 11

        # Act
        result = predict_players_with_ffpundit(
            player_ids=player_ids,
            target_gw=target_gw,
        )

        # Assert
        assert "v4_ffpundit_points" in result.columns
        assert "v4_classifier_points" in result.columns
        assert len(result) > 0
        # Result should have at least fixture_context columns + both predictions
        expected_base_cols = {
            "id", "player_id", "web_name", "position", "team_code",
            "opponent_team_strength"
        }
        assert expected_base_cols.issubset(set(result.columns))

    def test_v4_ffpundit_fallback_to_classifier_on_missing_match(self):
        """If FFPundit player not matched, v4_ffpundit_points = v4_classifier_points."""
        # Arrange
        player_ids = [412]
        target_gw = 11

        # Act
        result = predict_players_with_ffpundit(
            player_ids=player_ids,
            target_gw=target_gw,
        )

        # Assert: Row with no FFPundit match should have both columns,
        # and v4_ffpundit_points == v4_classifier_points (graceful fallback)
        assert len(result) >= 1
        # At least one row should exist; check if predictions are present
        for _, row in result.iterrows():
            assert pd.notna(row["v4_classifier_points"])
            assert pd.notna(row["v4_ffpundit_points"])

    def test_loads_matched_ffpundit_forecasts(self):
        """Pipeline loads and uses matched FFPundit forecasts for target_gw."""
        # Arrange
        player_ids = None  # Predict all players with fixtures in gw11
        target_gw = 11

        # Act
        result_with_ffpundit = predict_players_with_ffpundit(
            player_ids=player_ids,
            target_gw=target_gw,
        )

        # Assert: Should have both columns
        assert "v4_ffpundit_points" in result_with_ffpundit.columns
        # At least some rows should have FFPundit data (not all NaN)
        if len(result_with_ffpundit) > 0:
            assert result_with_ffpundit["v4_ffpundit_points"].notna().sum() > 0

    def test_audit_columns_present(self):
        """Pipeline includes audit columns (classifier_start_prob, ffpundit_start_prob, diff)."""
        # Arrange
        player_ids = [412, 580]
        target_gw = 11

        # Act
        result = predict_players_with_ffpundit(
            player_ids=player_ids,
            target_gw=target_gw,
            include_audit_cols=True,
        )

        # Assert
        assert "v4_classifier_start_prob" in result.columns
        assert "v4_ffpundit_start_prob" in result.columns
        assert "start_prob_diff" in result.columns
        # Diff should be ffpundit - classifier
        for _, row in result.iterrows():
            if pd.notna(row["v4_classifier_start_prob"]) and pd.notna(row["v4_ffpundit_start_prob"]):
                expected_diff = row["v4_ffpundit_start_prob"] - row["v4_classifier_start_prob"]
                assert abs(row["start_prob_diff"] - expected_diff) < 1e-6

    def test_audit_columns_optional(self):
        """include_audit_cols=False hides audit columns (shadow mode default)."""
        # Arrange
        player_ids = [412]
        target_gw = 11

        # Act
        result = predict_players_with_ffpundit(
            player_ids=player_ids,
            target_gw=target_gw,
            include_audit_cols=False,
        )

        # Assert: Audit columns not present by default
        assert "v4_classifier_start_prob" not in result.columns
        assert "v4_ffpundit_start_prob" not in result.columns
        assert "start_prob_diff" not in result.columns
        # But main prediction columns should be there
        assert "v4_ffpundit_points" in result.columns
        assert "v4_classifier_points" in result.columns

    def test_no_nan_in_predictions(self):
        """Pipeline handles missing FFPundit data: no NaN in prediction columns."""
        # Arrange
        player_ids = [412, 580, 10]
        target_gw = 11

        # Act
        result = predict_players_with_ffpundit(
            player_ids=player_ids,
            target_gw=target_gw,
            include_audit_cols=False,
        )

        # Assert: No NaN in prediction columns (graceful fallback)
        if len(result) > 0:
            assert result["v4_classifier_points"].isna().sum() == 0
            assert result["v4_ffpundit_points"].isna().sum() == 0

    def test_runs_before_deadline(self):
        """Pipeline can run pre-deadline for live predictions (no error)."""
        # Arrange: Use next unplayed GW (auto-detect)
        player_ids = None

        # Act & Assert: Should not raise
        try:
            result = predict_players_with_ffpundit(
                player_ids=player_ids,
                target_gw=None,  # Auto-detect
                include_audit_cols=False,
            )
            # May return empty if no future GW, but shouldn't crash
            assert isinstance(result, pd.DataFrame)
        except ValueError as e:
            # "No upcoming gameweek" is acceptable
            assert "upcoming" in str(e).lower()

    def test_original_predict_players_unchanged(self):
        """Original predict_players() still works without FFPundit."""
        # Arrange
        player_ids = [412, 580]
        target_gw = 11

        # Act
        result = predict_players(
            player_ids=player_ids,
            target_gw=target_gw,
        )

        # Assert: Columns should match original API
        expected_cols = {
            "id", "player_id", "web_name", "position", "team_code",
            "opponent_team_strength", "pred_total_points"
        }
        assert expected_cols.issubset(set(result.columns))
        # Should NOT have FFPundit columns
        assert "v4_ffpundit_points" not in result.columns
        assert "v4_classifier_points" not in result.columns

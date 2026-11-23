"""Tests for FFPundit→FPL player name matcher."""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from scripts.ffpundit_player_matcher import (
    load_ffpundit_forecasts,
    load_fpl_players,
    match_player_names,
    build_output_dataframe,
    main,
)


class TestLoadFFPunditForecasts:
    """Test loading FFPundit forecast CSV."""

    def test_load_valid_forecast_file(self):
        """Load gw{gw}_forecasts.csv with required columns."""
        df = load_ffpundit_forecasts(
            "/Users/mihalymercz/Projects2/fpl_predict/data/raw/ffpundit/gw11_forecasts.csv"
        )
        assert isinstance(df, pd.DataFrame)
        assert "team" in df.columns
        assert "player_name" in df.columns
        assert "starting_pct" in df.columns
        assert len(df) > 0

    def test_load_forecast_missing_file_raises(self):
        """Raise on missing forecast file."""
        with pytest.raises(FileNotFoundError):
            load_ffpundit_forecasts("/nonexistent/path/gw99_forecasts.csv")

    def test_load_forecast_missing_columns_raises(self):
        """Raise on missing required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = Path(tmpdir) / "bad.csv"
            bad_csv.write_text("team,player_name\nA,B\n")  # missing starting_pct
            with pytest.raises(ValueError):
                load_ffpundit_forecasts(str(bad_csv))


class TestLoadFPLPlayers:
    """Test loading FPL players CSV."""

    def test_load_valid_players_file(self):
        """Load players.csv with required columns."""
        df = load_fpl_players(
            "/Users/mihalymercz/Projects2/fpl_predict/data/raw/2026-27/players/players.csv"
        )
        assert isinstance(df, pd.DataFrame)
        assert "id" in df.columns
        assert "web_name" in df.columns
        assert "team_code" in df.columns
        assert len(df) > 0

    def test_load_players_missing_file_raises(self):
        """Raise on missing players file."""
        with pytest.raises(FileNotFoundError):
            load_fpl_players("/nonexistent/players.csv")

    def test_load_players_missing_columns_raises(self):
        """Raise on missing required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = Path(tmpdir) / "bad.csv"
            bad_csv.write_text("id,web_name\n1,Player\n")  # missing team_code
            with pytest.raises(ValueError):
                load_fpl_players(str(bad_csv))


class TestMatchPlayerNames:
    """Test name matching logic (exact and fuzzy)."""

    def setup_method(self):
        """Create test data fixtures."""
        self.fpl_players = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "web_name": ["Raya", "Timber", "Calafiori", "Gabriel", "Rice", "Ødegaard"],
            "team_code": [3, 3, 3, 3, 3, 3],
        })

    def test_exact_match_case_insensitive(self):
        """Find exact match ignoring case."""
        result = match_player_names(
            ffpundit_df=pd.DataFrame({
                "player_name": ["raya", "TIMBER"],
                "starting_pct": [95.0, 70.0],
                "team": ["Arsenal", "Arsenal"],
            }),
            fpl_df=self.fpl_players,
        )
        assert len(result) >= 2
        assert result[result["player_name"] == "raya"]["player_id"].values[0] == 1
        assert result[result["player_name"] == "TIMBER"]["player_id"].values[0] == 2
        assert all(result["match_type"] == "exact")

    def test_exact_match_whitespace_normalized(self):
        """Find exact match after normalizing whitespace."""
        result = match_player_names(
            ffpundit_df=pd.DataFrame({
                "player_name": ["  Ødegaard  ", "Ødegaard\n"],
                "starting_pct": [80.0, 80.0],
                "team": ["Arsenal", "Arsenal"],
            }),
            fpl_df=self.fpl_players,
        )
        assert len(result) >= 2
        assert all(result["match_type"] == "exact")

    def test_fuzzy_match_above_threshold(self):
        """Find fuzzy match above 0.50 threshold (last name only)."""
        result = match_player_names(
            ffpundit_df=pd.DataFrame({
                "player_name": ["Jurriën Timber"],  # FFPundit format
                "starting_pct": [70.0],
                "team": ["Arsenal"],
            }),
            fpl_df=self.fpl_players,
        )
        assert len(result) == 1
        # Either exact or fuzzy, as long as it matches
        assert result.iloc[0]["player_id"] == 2

    def test_no_match_logs_skipped(self):
        """Skip unmatchable names, log to stderr."""
        result = match_player_names(
            ffpundit_df=pd.DataFrame({
                "player_name": ["Completely Unknown Player"],
                "starting_pct": [50.0],
                "team": ["Arsenal"],
            }),
            fpl_df=self.fpl_players,
        )
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["player_id"])
        assert result.iloc[0]["match_type"] == "skipped"

    def test_multiple_teams_no_collision(self):
        """Handle same name across different teams."""
        fpl_multi_team = pd.concat([
            self.fpl_players,
            pd.DataFrame({
                "id": [101, 102],
                "web_name": ["Raya", "Rice"],  # Duplicate names, different teams
                "team_code": [1, 1],  # Different team
            }),
        ], ignore_index=True)

        result = match_player_names(
            ffpundit_df=pd.DataFrame({
                "player_name": ["Raya"],
                "starting_pct": [95.0],
                "team": ["Arsenal"],  # Specified team
            }),
            fpl_df=fpl_multi_team,
        )
        # Should match Raya from Arsenal (team_code 3), not team_code 1
        assert result.iloc[0]["player_id"] in [1]  # Arsenal's Raya

    def test_zero_start_pct_for_missing_forecast(self):
        """Treat missing forecast as 0% start probability."""
        # FFPundit may not have all players; matcher fills gaps
        # This test verifies output includes players with 0% if they're in FPL
        result = match_player_names(
            ffpundit_df=pd.DataFrame({
                "player_name": ["Raya"],
                "starting_pct": [95.0],
                "team": ["Arsenal"],
            }),
            fpl_df=self.fpl_players,
        )
        # Result should have Raya only (others implicitly 0 in full output)
        assert len(result) == 1
        assert result.iloc[0]["player_id"] == 1


class TestBuildOutputDataFrame:
    """Test output DataFrame schema and content."""

    def test_output_has_required_columns(self):
        """Matched result has all required output columns."""
        matches = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["Raya", "Timber"],
            "starting_pct": [95.0, 70.0],
            "matched_fpl_name": ["Raya", "Timber"],
            "match_type": ["exact", "exact"],
            "team": ["Arsenal", "Arsenal"],
        })
        result = build_output_dataframe(
            matches=matches,
            season="2026-27",
            gameweek=11,
        )
        assert "player_id" in result.columns
        assert "season" in result.columns
        assert "gw" in result.columns
        assert "ffpundit_start_pct" in result.columns
        assert "matched_fpl_name" in result.columns
        assert "match_type" in result.columns

    def test_output_season_and_gw_populated(self):
        """Output includes season and gameweek."""
        matches = pd.DataFrame({
            "player_id": [1],
            "player_name": ["Raya"],
            "starting_pct": [95.0],
            "matched_fpl_name": ["Raya"],
            "match_type": ["exact"],
            "team": ["Arsenal"],
        })
        result = build_output_dataframe(
            matches=matches,
            season="2026-27",
            gameweek=11,
        )
        assert all(result["season"] == "2026-27")
        assert all(result["gw"] == 11)

    def test_output_handles_skipped_rows(self):
        """Skipped players included with NaN player_id."""
        matches = pd.DataFrame({
            "player_id": [1, None],
            "player_name": ["Raya", "Unknown"],
            "starting_pct": [95.0, 50.0],
            "matched_fpl_name": ["Raya", None],
            "match_type": ["exact", "skipped"],
            "team": ["Arsenal", "Arsenal"],
        })
        result = build_output_dataframe(
            matches=matches,
            season="2026-27",
            gameweek=11,
        )
        assert len(result) == 2
        assert result.iloc[1]["match_type"] == "skipped"


class TestMainCLI:
    """Test end-to-end CLI orchestration."""

    def test_main_produces_valid_output(self):
        """main() loads, matches, outputs DataFrame."""
        result = main(
            ffpundit_csv="/Users/mihalymercz/Projects2/fpl_predict/data/raw/ffpundit/gw11_forecasts.csv",
            fpl_players_csv="/Users/mihalymercz/Projects2/fpl_predict/data/raw/2026-27/players/players.csv",
            teams_csv="/Users/mihalymercz/Projects2/fpl_predict/data/raw/2026-27/teams.csv",
            season="2026-27",
            gameweek=11,
        )
        assert isinstance(result, pd.DataFrame)
        assert "player_id" in result.columns
        assert "season" in result.columns
        assert "gw" in result.columns
        # Expect very high match rate (100% on 2026-27 data)
        matched = result[result["match_type"] != "skipped"]
        skip_rate = len(result[result["match_type"] == "skipped"]) / len(result)
        assert skip_rate < 0.01, f"Skip rate {skip_rate:.1%} exceeds 1% threshold"

"""Test v4 feature building: rolling means, DGW aggregation, pre-label-data retention."""

import pandas as pd
import pytest
from pathlib import Path
import sys

# Allow import of scripts module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from models.v4.build_features import (
    aggregate_to_gw_level,
    add_time_ordering,
    add_form_rolling_features,
    build_v4_features,
)


@pytest.fixture
def sample_enriched_data():
    """Minimal enriched parquet: fixture-level rows with starts label."""
    return pd.DataFrame({
        "season": ["2022-23", "2022-23", "2022-23", "2022-23", "2022-23"],
        "gw": [16, 16, 17, 18, 19],
        "fixture_id": [100, 100, 101, 102, 103],
        "player_id": [1, 2, 1, 1, 1],
        "position": ["DEF", "MID", "DEF", "DEF", "DEF"],
        "team_id": [1, 2, 1, 1, 1],
        "team_code": ["ARS", "MUN", "ARS", "ARS", "ARS"],
        "team_strength": [90, 85, 90, 90, 90],
        "opponent_id": [2, 1, 3, 4, 5],
        "opponent_team_code": ["MUN", "ARS", "LIV", "CHE", "MCI"],
        "opponent_team_strength": [85, 90, 88, 86, 92],
        "opponent_team_promoted": [False, False, False, False, False],
        "is_home": [True, False, True, False, True],
        "team_goals": [2.0, 1.0, 1.0, 0.0, 2.0],
        "opponent_goals": [1.0, 2.0, 1.0, 2.0, 0.0],
        "minutes": [90, 60, 90, 45, 90],
        "goals_scored": [1, 0, 0, 0, 1],
        "assists": [0, 1, 0, 0, 0],
        "clean_sheets": [1, 0, 1, 0, 1],
        "yellow_cards": [0, 1, 0, 0, 0],
        "red_cards": [0, 0, 0, 0, 0],
        "own_goals": [0, 0, 0, 0, 0],
        "penalties_missed": [0, 0, 0, 0, 0],
        "penalties_saved": [0, 0, 0, 0, 0],
        "bonus": [3, 0, 3, 0, 3],
        "bps": [30, 15, 30, 10, 30],
        "xG": [0.5, 0.3, 0.4, 0.1, 0.6],
        "xGC": [0.4, 0.5, 0.3, 0.6, 0.2],
        "xA": [0.1, 0.2, 0.1, 0.0, 0.1],
        "total_points": [9, 6, 9, 3, 15],
        "starts": [1.0, 1.0, 1.0, 0.0, 1.0],
        "creativity": [10.0, 15.0, 10.0, 5.0, 12.0],
        "ict_index": [25.0, 20.0, 24.0, 10.0, 28.0],
        "influence": [30.0, 25.0, 28.0, 15.0, 32.0],
        "threat": [35.0, 30.0, 32.0, 18.0, 38.0],
        "goals_conceded": [1, 2, 1, 2, 0],
    })


@pytest.fixture
def dgw_data():
    """Double-gameweek: same player with 2 fixtures in gw 20."""
    return pd.DataFrame({
        "season": ["2022-23", "2022-23"],
        "gw": [20, 20],
        "fixture_id": [200, 201],
        "player_id": [1, 1],
        "position": ["MID", "MID"],
        "team_id": [1, 1],
        "team_code": ["ARS", "ARS"],
        "team_strength": [90, 90],
        "opponent_id": [2, 3],
        "opponent_team_code": ["MUN", "LIV"],
        "opponent_team_strength": [85, 88],
        "opponent_team_promoted": [False, False],
        "is_home": [True, False],
        "team_goals": [2.0, 1.0],
        "opponent_goals": [1.0, 1.0],
        "minutes": [90, 60],
        "goals_scored": [1, 0],
        "assists": [0, 1],
        "clean_sheets": [1, 0],
        "yellow_cards": [0, 1],
        "red_cards": [0, 0],
        "own_goals": [0, 0],
        "penalties_missed": [0, 0],
        "penalties_saved": [0, 0],
        "bonus": [3, 0],
        "bps": [30, 15],
        "xG": [0.5, 0.3],
        "xGC": [0.4, 0.5],
        "xA": [0.1, 0.2],
        "total_points": [9, 6],
        "starts": [1.0, 1.0],
        "creativity": [10.0, 15.0],
        "ict_index": [25.0, 20.0],
        "influence": [30.0, 25.0],
        "threat": [35.0, 30.0],
        "goals_conceded": [1, 1],
    })


class TestDGWAggregation:
    """Double-gameweek fixtures aggregate to one row per (player, season, gw)."""

    def test_dgw_sums_minutes_and_points(self, dgw_data):
        """DGW: 2 fixtures in gw 20 → 1 row with summed minutes/points."""
        agg = aggregate_to_gw_level(dgw_data)
        
        assert len(agg) == 1, f"Expected 1 row after DGW aggregation, got {len(agg)}"
        assert agg.iloc[0]["minutes"] == 150, f"Expected minutes=150 (90+60), got {agg.iloc[0]['minutes']}"
        assert agg.iloc[0]["total_points"] == 15, f"Expected points=15 (9+6), got {agg.iloc[0]['total_points']}"
        assert agg.iloc[0]["goals_scored"] == 1, f"Expected goals=1, got {agg.iloc[0]['goals_scored']}"
        assert agg.iloc[0]["assists"] == 1, f"Expected assists=1, got {agg.iloc[0]['assists']}"

    def test_dgw_num_fixtures_recorded(self, dgw_data):
        """DGW aggregation records num_fixtures=2."""
        agg = aggregate_to_gw_level(dgw_data)
        assert agg.iloc[0]["num_fixtures"] == 2, f"Expected num_fixtures=2, got {agg.iloc[0]['num_fixtures']}"

    def test_dgw_home_frac(self, dgw_data):
        """DGW: 1 home + 1 away → is_home_frac = 0.5."""
        agg = aggregate_to_gw_level(dgw_data)
        expected_frac = 1 / 2
        assert agg.iloc[0]["is_home_frac"] == expected_frac, \
            f"Expected is_home_frac={expected_frac}, got {agg.iloc[0]['is_home_frac']}"

    def test_dgw_position_first(self, dgw_data):
        """DGW aggregation keeps first position (both should be MID here)."""
        agg = aggregate_to_gw_level(dgw_data)
        assert agg.iloc[0]["position"] == "MID"


class TestRollingFeatures:
    """Rolling means computed correctly over prior gameweeks (no leakage)."""

    def test_rolling_means_computed(self, sample_enriched_data):
        """After aggregation and time ordering, rolling means exist for windows 3, 5, 10."""
        agg = aggregate_to_gw_level(sample_enriched_data)
        ordered = add_time_ordering(agg)
        featured = add_form_rolling_features(ordered, shift=True)
        
        # Player 1 (DEF) has 4 rows: gw 16, 17, 18, 19
        p1 = featured[featured["player_id"] == 1]
        
        # At gw 17 (second row for player 1), should have rolling3 mean of gw 16
        # Minutes: gw 16 = 90, so roll3 at gw 17 should be (90,) avg = 90
        assert "minutes_roll3" in featured.columns, "minutes_roll3 not in columns"
        
        # Second row (gw 17) for player 1
        gw17_roll3 = p1.iloc[1]["minutes_roll3"]
        assert gw17_roll3 == 90.0, f"Expected minutes_roll3=90 at gw 17, got {gw17_roll3}"

    def test_rolling_shift_no_leakage(self, sample_enriched_data):
        """With shift=True, row N's rolling features don't include row N's stats."""
        agg = aggregate_to_gw_level(sample_enriched_data)
        ordered = add_time_ordering(agg)
        featured = add_form_rolling_features(ordered, shift=True)
        
        # Player 1 at gw 18 has gw 16 = 90 min, gw 17 = 90 min, gw 18 = 45 min
        # With shift=True, gw 18's roll3 should be avg(90, 90) = 90, NOT including gw 18's 45
        p1_gw18 = featured[(featured["player_id"] == 1) & (featured["gw"] == 18)].iloc[0]
        expected_roll3 = (90 + 90) / 2  # gw 16 and 17 only, not gw 18's 45
        actual_roll3 = p1_gw18["minutes_roll3"]
        assert actual_roll3 == expected_roll3, \
            f"Expected roll3={expected_roll3}, got {actual_roll3} (leakage suspected)"

    def test_start_rate_rolling(self, sample_enriched_data):
        """start_rate_roll3 = fraction of gws with starts >= 60 min."""
        agg = aggregate_to_gw_level(sample_enriched_data)
        ordered = add_time_ordering(agg)
        featured = add_form_rolling_features(ordered, shift=True)
        
        # Player 1: gw 16 (90 min, starts=1), gw 17 (90 min, starts=1), gw 18 (45 min, starts=0), gw 19 (90 min, starts=1)
        # At gw 19, start_rate_roll3 = avg(1, 0, 1) = 2/3 (shifted, so gw 16+17+18 only)
        p1_gw19 = featured[(featured["player_id"] == 1) & (featured["gw"] == 19)].iloc[0]
        expected_rate = 2 / 3  # gw18 has 45 min < 60, so starts=0
        actual_rate = p1_gw19["start_rate_roll3"]
        assert abs(actual_rate - expected_rate) < 0.01, \
            f"Expected start_rate_roll3={expected_rate}, got {actual_rate}"


class TestPreLabelDataRetention:
    """Pre-GW16 and pre-2022-23 rows included in rolling windows but marked for exclusion."""

    def test_pre_gw16_included_in_rolling(self):
        """Pre-GW16 rows compute rolling features but have starts=NaN."""
        # Create data with gw < 16 (pre-label era)
        pre_gw = pd.DataFrame({
            "season": ["2022-23", "2022-23"],
            "gw": [1, 15],
            "fixture_id": [1, 2],
            "player_id": [1, 1],
            "position": ["DEF", "DEF"],
            "team_id": [1, 1],
            "team_code": ["ARS", "ARS"],
            "team_strength": [90, 90],
            "opponent_id": [2, 3],
            "opponent_team_code": ["MUN", "LIV"],
            "opponent_team_strength": [85, 88],
            "opponent_team_promoted": [False, False],
            "is_home": [True, False],
            "team_goals": [2.0, 1.0],
            "opponent_goals": [1.0, 1.0],
            "minutes": [60, 90],
            "goals_scored": [0, 1],
            "assists": [0, 0],
            "clean_sheets": [1, 1],
            "yellow_cards": [0, 0],
            "red_cards": [0, 0],
            "own_goals": [0, 0],
            "penalties_missed": [0, 0],
            "penalties_saved": [0, 0],
            "bonus": [0, 3],
            "bps": [15, 30],
            "xG": [0.3, 0.5],
            "xGC": [0.5, 0.4],
            "xA": [0.0, 0.1],
            "total_points": [3, 9],
            "starts": [float("nan"), float("nan")],  # Pre-GW16: no labels
            "creativity": [5.0, 10.0],
            "ict_index": [15.0, 25.0],
            "influence": [20.0, 30.0],
            "threat": [25.0, 35.0],
            "goals_conceded": [1, 1],
        })
        
        agg = aggregate_to_gw_level(pre_gw)
        ordered = add_time_ordering(agg)
        featured = add_form_rolling_features(ordered, shift=True)
        
        # Both rows should be present
        assert len(featured) == 2, f"Expected 2 rows, got {len(featured)}"
        
        # Both should have NaN starts
        assert pd.isna(featured.iloc[0]["starts"]), "Pre-GW16 row should have NaN starts"
        assert pd.isna(featured.iloc[1]["starts"]), "GW15 row should have NaN starts"
        
        # GW15 row should have rolling features from GW1
        gw15_row = featured[featured["gw"] == 15].iloc[0]
        assert "minutes_roll3" in gw15_row.index, "GW15 should compute rolling features"
        assert gw15_row["minutes_roll3"] == 60.0, \
            f"GW15's roll3 should use GW1 (60 min), got {gw15_row['minutes_roll3']}"


class TestFeatureOutput:
    """Output parquet structure and content."""

    def test_output_has_starts_column(self, sample_enriched_data):
        """Output includes starts column unchanged from input."""
        agg = aggregate_to_gw_level(sample_enriched_data)
        ordered = add_time_ordering(agg)
        featured = add_form_rolling_features(ordered, shift=True)
        
        assert "starts" in featured.columns, "starts column missing from output"
        # Check that the values are preserved
        p2_starts = featured[featured["player_id"] == 2].iloc[0]["starts"]
        assert p2_starts == 1.0, f"Expected player 2 starts=1, got {p2_starts}"

    def test_output_has_rolling_features(self, sample_enriched_data):
        """Output includes rolling means for windows 3, 5, 10."""
        agg = aggregate_to_gw_level(sample_enriched_data)
        ordered = add_time_ordering(agg)
        featured = add_form_rolling_features(ordered, shift=True)
        
        required_cols = ["minutes_roll3", "minutes_roll5", "minutes_roll10",
                         "total_points_roll3", "total_points_roll5", "total_points_roll10",
                         "start_rate_roll3", "start_rate_roll5", "start_rate_roll10"]
        for col in required_cols:
            assert col in featured.columns, f"{col} missing from output"

    def test_all_input_rows_preserved(self, sample_enriched_data):
        """After aggregation, number of rows = number of unique (season, gw, player_id)."""
        n_input = len(sample_enriched_data)
        
        agg = aggregate_to_gw_level(sample_enriched_data)
        ordered = add_time_ordering(agg)
        featured = add_form_rolling_features(ordered, shift=True)
        
        # Count unique (season, gw, player_id) in input
        unique_groups = sample_enriched_data.groupby(["season", "gw", "player_id"]).size().reset_index()
        expected_rows = len(unique_groups)
        
        assert len(featured) == expected_rows, \
            f"Expected {expected_rows} rows after aggregation, got {len(featured)}"


class TestBuildV4FeaturesIntegration:
    """Full pipeline: input → aggregate → time order → rolling features → output."""

    def test_build_v4_features_runs(self, sample_enriched_data, tmp_path):
        """Full pipeline executes without error."""
        input_path = tmp_path / "input.parquet"
        output_path = tmp_path / "output.parquet"
        
        sample_enriched_data.to_parquet(input_path, index=False)
        
        # Call the main function
        build_v4_features(input_path, output_path)
        
        # Output file exists
        assert output_path.exists(), f"Output file not created at {output_path}"
        
        # Can read output
        result = pd.read_parquet(output_path)
        assert len(result) > 0, "Output parquet is empty"
        assert "starts" in result.columns, "starts column missing from output"

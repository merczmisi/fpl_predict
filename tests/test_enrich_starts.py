"""Tests for enrich_with_starts script."""

import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def enriched_df():
    """Load the enriched parquet."""
    path = Path("data/processed/player_gameweeks_combined_2020_2025_with_starts.parquet")
    return pd.read_parquet(path)


def test_starts_column_exists(enriched_df):
    """Verify starts column was added."""
    assert "starts" in enriched_df.columns


def test_no_row_loss(enriched_df):
    """Verify no rows were lost during enrichment."""
    combined_path = Path("data/processed/player_gameweeks_combined_2020_2025.parquet")
    combined = pd.read_parquet(combined_path)
    assert len(enriched_df) == len(combined)


def test_pre_2022_23_no_starts(enriched_df):
    """Pre-2022-23 seasons should have no starts data."""
    pre_2223 = enriched_df[enriched_df["season"].isin(["2020-21", "2021-22"])]
    assert pre_2223["starts"].isna().all()


def test_2022_23_pre_gw16_no_starts(enriched_df):
    """2022-23 before GW16 should have no starts data."""
    pre_gw16 = enriched_df[(enriched_df["season"] == "2022-23") & (enriched_df["gw"] < 16)]
    assert pre_gw16["starts"].isna().all()


def test_2022_23_from_gw16_has_starts(enriched_df):
    """2022-23 from GW16 onward should have starts data."""
    from_gw16 = enriched_df[(enriched_df["season"] == "2022-23") & (enriched_df["gw"] >= 16)]
    assert from_gw16["starts"].notna().sum() > 0


def test_later_seasons_have_starts(enriched_df):
    """2023-24 onward should have starts data."""
    later = enriched_df[enriched_df["season"].isin(["2023-24", "2024-25", "2025-26"])]
    assert later["starts"].notna().sum() > 0


def test_starts_values_are_binary_or_na(enriched_df):
    """Starts should be 0, 1, or NaN."""
    valid_values = {0.0, 1.0}
    starts_notna = enriched_df["starts"].dropna().unique()
    assert all(v in valid_values for v in starts_notna)


def test_starts_coverage_2023_24_complete(enriched_df):
    """2023-24 should have full starts coverage."""
    season_2324 = enriched_df[enriched_df["season"] == "2023-24"]
    assert season_2324["starts"].notna().sum() == len(season_2324)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

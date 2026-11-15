"""Enrich combined parquet with starts labels from raw fixture JSON."""

from pathlib import Path
import json
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Season availability windows for starts field
STARTS_AVAILABILITY = {
    "2020-21": None,  # No starts field
    "2021-22": None,  # No starts field
    "2022-23": 16,    # Available from GW16 onward
    "2023-24": 1,     # Available from GW1 onward
    "2024-25": 1,
    "2025-26": 1,
    "2026-27": 1,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_starts_from_season(season_str):
    """Extract starts field from all gameweeks in a season."""
    season_dir = RAW_DIR / season_str
    if not season_dir.exists():
        print(f"Season {season_str} raw data not found")
        return pd.DataFrame()

    starts_availability = STARTS_AVAILABILITY.get(season_str)
    
    # If season has no starts at all, return empty DataFrame
    if starts_availability is None:
        print(f"Season {season_str} has no starts field")
        return pd.DataFrame()

    # Load player mapping
    try:
        players = pd.read_csv(
            season_dir / "players" / "players.csv",
            usecols=["id", "code"]
        ).rename(columns={"id": "gameweek_player_id", "code": "player_id"})
    except FileNotFoundError:
        print(f"Players CSV not found for {season_str}")
        return pd.DataFrame()

    records = []
    for gw_file in sorted((season_dir / "players").glob("gw_*.json")):
        gw = int(gw_file.stem.split("_")[1])

        # Skip GWs before availability window
        if gw < starts_availability:
            continue

        gw_data = load_json(gw_file)
        if not gw_data:
            continue

        for entry in gw_data:
            records.append({
                "season": season_str,
                "gw": gw,
                "gameweek_player_id": entry.get("player_id"),
                "starts": entry.get("starts"),
            })

    if not records:
        return pd.DataFrame()

    df_starts = pd.DataFrame(records)
    
    # Deduplicate: in case of double gameweeks, keep first occurrence per (season, gw, gameweek_player_id)
    df_starts = df_starts.drop_duplicates(
        subset=["season", "gw", "gameweek_player_id"],
        keep="first",
    )
    
    # Merge with player mapping to get real player_id
    df_starts = df_starts.merge(
        players,
        on="gameweek_player_id",
        how="left",
    )
    
    return df_starts[["season", "gw", "player_id", "starts"]]


def main():
    print("Loading combined parquet...")
    input_path = PROCESSED_DIR / "player_gameweeks_combined_2020_2025.parquet"
    df_combined = pd.read_parquet(input_path)
    
    print(f"Loaded {len(df_combined):,} rows")
    initial_row_count = len(df_combined)
    
    print("\nExtracting starts from raw data...")
    
    # Process each season
    all_starts = []
    for season in STARTS_AVAILABILITY.keys():
        df_starts = extract_starts_from_season(season)
        if len(df_starts) > 0:
            print(f"  {season}: extracted {len(df_starts):,} records")
            all_starts.append(df_starts)
    
    if not all_starts:
        print("No starts data found!")
        return
    
    df_starts_all = pd.concat(all_starts, ignore_index=True)
    print(f"Total starts records: {len(df_starts_all):,}")
    
    # Merge starts into combined df by (season, gw, player_id)
    print("\nMerging starts into combined parquet...")
    
    # Use left merge to preserve all rows from combined
    df_merged = df_combined.merge(
        df_starts_all,
        on=["season", "gw", "player_id"],
        how="left",
    )
    
    # Verify no row loss
    assert len(df_merged) == initial_row_count, \
        f"Row count mismatch: {len(df_merged)} vs {initial_row_count}"
    
    print(f"Merged: {len(df_merged):,} rows (no data loss)")
    
    # Validate data quality
    print("\nValidating starts data quality...")
    
    # Check pre-2022-23 seasons have starts = NaN
    pre_2223 = df_merged[df_merged["season"].isin(["2020-21", "2021-22"])]
    pre_2223_with_starts = pre_2223[pre_2223["starts"].notna()]
    assert len(pre_2223_with_starts) == 0, \
        f"Pre-2022-23 should have no starts: {len(pre_2223_with_starts)} found"
    print("✓ Pre-2022-23 seasons have starts = NaN")
    
    # Check 2022-23 pre-GW16 have starts = NaN
    gw_2223_pre = df_merged[(df_merged["season"] == "2022-23") & (df_merged["gw"] < 16)]
    gw_2223_pre_with_starts = gw_2223_pre[gw_2223_pre["starts"].notna()]
    assert len(gw_2223_pre_with_starts) == 0, \
        f"2022-23 pre-GW16 should have no starts: {len(gw_2223_pre_with_starts)} found"
    print("✓ 2022-23 pre-GW16 have starts = NaN")
    
    # Check 2022-23 GW16+ and later seasons have starts populated (except where data missing)
    gw_2223_post = df_merged[(df_merged["season"] == "2022-23") & (df_merged["gw"] >= 16)]
    gw_2223_post_na = gw_2223_post[gw_2223_post["starts"].isna()]
    print(f"  2022-23 GW16+: {len(gw_2223_post):,} rows, {len(gw_2223_post_na):,} with NaN starts")
    
    later_seasons = df_merged[df_merged["season"].isin(["2023-24", "2024-25", "2025-26", "2026-27"])]
    later_na = later_seasons[later_seasons["starts"].isna()]
    print(f"  2023-24+: {len(later_seasons):,} rows, {len(later_na):,} with NaN starts")
    
    print("\nStarts value counts:")
    print(df_merged["starts"].value_counts(dropna=False))
    
    # Save output
    output_path = PROCESSED_DIR / "player_gameweeks_combined_2020_2025_with_starts.parquet"
    df_merged.to_parquet(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Rows: {len(df_merged):,}")
    print(f"  Columns: {len(df_merged.columns)}")


if __name__ == "__main__":
    main()

"""
Enrich combined parquet with starts labels.

Loads raw fixture JSON (2022-23 GW16 onward) and extracts the starts field.
Merges into existing combined player-gameweek parquet, preserving all pre-GW16 rows
with starts marked as NaN. No rows lost during merge.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

# Paths
raw_data_dir = Path("data/raw")
processed_dir = Path("data/processed")
input_parquet = processed_dir / "player_gameweeks_combined_2020_2025.parquet"
output_parquet = processed_dir / "player_gameweeks_combined_2020_2025_with_starts.parquet"

# Load combined parquet
print("Loading combined parquet...")
df = pd.read_parquet(input_parquet)
initial_row_count = len(df)
print(f"  Initial rows: {initial_row_count}")

# Initialize starts column as NaN (all rows marked unknown)
df["starts"] = np.nan

# Load player mapping for 2022-23 (FPL element_id -> player_id/code)
print("\nLoading player mapping for 2022-23...")
players_csv = pd.read_csv(raw_data_dir / "2022-23" / "players" / "players.csv")
element_to_player_id = dict(zip(players_csv['id'], players_csv['code']))
print(f"  Loaded {len(element_to_player_id)} player mappings")

# Process 2022-23 season GW16+ (where starts data exists)
season_2022_23_gws = sorted(df[(df['season'] == '2022-23') & (df['gw'] >= 16)]['gw'].unique())
print(f"\n2022-23 GWs to enrich (GW16+): {season_2022_23_gws}")

merged_count = 0

for gw in season_2022_23_gws:
    gw_file = raw_data_dir / "2022-23" / "players" / f"gw_{gw}.json"
    
    if not gw_file.exists():
        print(f"  GW{gw}: File not found, skipping")
        continue
    
    print(f"  Processing GW{gw}...")
    
    # Load raw gameweek data
    with open(gw_file) as f:
        gw_data = json.load(f)
    
    # Extract element_id (FPL id) and starts, then map to player_id
    starts_map = {}
    for entry in gw_data:
        element_id = entry['element']
        starts = entry.get('starts', np.nan)
        # Map FPL element_id to player_id using players.csv
        if element_id in element_to_player_id:
            player_id = element_to_player_id[element_id]
            starts_map[player_id] = starts
    
    # Merge into dataframe for this GW
    gw_mask = (df['season'] == '2022-23') & (df['gw'] == gw)
    gw_player_ids = df.loc[gw_mask, 'player_id'].values
    
    # Update starts for matching players
    for player_id in gw_player_ids:
        if player_id in starts_map:
            mask = gw_mask & (df['player_id'] == player_id)
            df.loc[mask, 'starts'] = starts_map[player_id]
            merged_count += 1

print(f"\nTotal cells merged with starts data: {merged_count}")

# Verify no rows lost
final_row_count = len(df)
assert final_row_count == initial_row_count, \
    f"Row count mismatch: {initial_row_count} -> {final_row_count}"
print(f"✓ Row count preserved: {final_row_count}")

# Check starts column
starts_known = df['starts'].notna().sum()
starts_unknown = df['starts'].isna().sum()
print(f"✓ Starts known: {starts_known}, unknown: {starts_unknown}")

# Verify no duplicate keys
duplicates = df.duplicated(subset=['season', 'fixture_id', 'player_id'])
assert not duplicates.any(), "Duplicate rows detected"
print(f"✓ No duplicates")

# Verify all original columns present
expected_cols = set(pd.read_parquet(input_parquet).columns)
actual_cols = set(df.columns)
missing = expected_cols - actual_cols
extra = actual_cols - expected_cols
assert not missing, f"Missing columns: {missing}"
print(f"✓ All original columns present")
print(f"✓ New columns: {extra}")

# Save enriched parquet
print(f"\nSaving enriched parquet to {output_parquet}...")
df.to_parquet(output_parquet, index=False)
print(f"✓ Done. Output: {output_parquet}")

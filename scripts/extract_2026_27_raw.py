"""Extract 2026-27 raw player-gameweek data for bias evaluation.

Loads raw 2026-27 gameweek JSON files and extracts player actual points.
Since we don't have full features for 2026-27, we cannot run full inference,
but we can measure bias on players with available features.

For now: extract actual points from raw data only.
Real inference would require building full features (future work).
"""

from pathlib import Path
import json
import pandas as pd

RAW_DATA_DIR = Path("data/raw/2026-27/players")


def load_raw_gameweeks():
    """Load all 2026-27 gameweek JSON files."""
    gws = {}
    for gw_file in sorted(RAW_DATA_DIR.glob("gw_*.json")):
        gw_num = int(gw_file.stem.split("_")[1])
        with open(gw_file) as f:
            players = json.load(f)
        gws[gw_num] = players
    return gws


def extract_player_gameweeks(gws):
    """Extract player-gameweek dataframe from raw data."""
    rows = []
    for gw, players in gws.items():
        for player in players:
            # Skip players without total_points (incomplete GW)
            if "total_points" not in player:
                continue
            rows.append({
                "gw": gw,
                "player_id": player["player_id"],
                "total_points": player["total_points"],
                "starts": player["starts"],
                "minutes": player["minutes"],
                "team_code": player["team_name"],
                "opponent_team_code": player["opponent_team_name"],
                "was_home": player["was_home"],
                "fixture": player["fixture"],
            })
    
    df = pd.DataFrame(rows)
    df["season"] = "2026-27"
    return df[["season", "gw", "player_id", "total_points", "starts", "minutes", 
               "team_code", "opponent_team_code", "was_home", "fixture"]]


def main():
    print("Loading 2026-27 raw gameweek data...")
    gws = load_raw_gameweeks()
    print(f"Loaded {len(gws)} gameweeks")
    
    print("Extracting player-gameweek data...")
    df = extract_player_gameweeks(gws)
    
    print(f"Extracted {len(df)} player-gameweek records")
    print(f"GWs: {sorted(df['gw'].unique())}")
    print(f"Unique players: {df['player_id'].nunique()}")
    
    # Save for analysis
    output_path = Path("data/processed/player_gameweeks_2026_27.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")
    
    return df


if __name__ == "__main__":
    main()

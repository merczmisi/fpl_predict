from pathlib import Path
import json

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

OPTIONAL_COLUMNS = [
    "xG",
    "xA",
    "xGC",
]

STRING_COLUMNS = [
    "season",
    "position",
    "team_name",
    "opponent_team_code",
]

NUMERIC_COLUMNS = [
    "gw",
    "fixture_id",
    "player_id",
    "team_id",
    "team_strength",
    "opponent_id",
    "opponent_team_strength",
    "team_goals",
    "opponent_goals",
    "minutes",
    "goals_scored",
    "goals_conceded",
    "xG",
    "xGC",
    "xA",
    "assists",
    "clean_sheets",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "bonus",
    "bps",
    "creativity",
    "ict_index",
    "influence",
    "threat",
    "total_points",
]

SEASON = "2026-27"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_season(season_dir):
    season = season_dir.name

    # -------------------------
    # Player metadata
    # -------------------------
    players = pd.read_csv(
        season_dir / "players" / "players.csv",
        usecols=["id", "code", "element_type"],
    ).rename(
        columns={
            "id": "gameweek_player_id",
            "code": "player_id",
        }
    )
    players["position"] = players["element_type"].map(
        {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    )
    players = players[["gameweek_player_id", "player_id", "position"]]

    # -------------------------
    # Team metadata
    # -------------------------
    teams = pd.read_csv(season_dir / "teams.csv")

    teams = teams[
        [
            "id",
            "name",
            "short_name",
            "strength",
        ]
    ].rename(
        columns={
            "id": "team_id",
            "name": "team_name",
            "short_name": "team_code",
            "strength": "team_strength",
        }
    )

    # -------------------------
    # Gameweeks
    # -------------------------
    all_gws = []

    for gw_file in sorted((season_dir / "players").glob("gw_*.json")):
        gw = int(gw_file.stem.split("_")[1])

        records = load_json(gw_file)

        if not records:
            continue

        df = pd.DataFrame(records)

        df["season"] = season
        df["gw"] = gw

        all_gws.append(df)

    if not all_gws:
        return pd.DataFrame()

    df = pd.concat(all_gws, ignore_index=True)

    # -------------------------
    # Rename gameweek columns
    # -------------------------

    df = df.rename(
        columns={
            "fixture": "fixture_id",
            "expected_goals": "xG",
            "expected_assists": "xA",
            "expected_goals_conceded": "xGC",
            "was_home": "is_home",
            "team_name": "team_code",
            "opponent_team_name": "opponent_team_code",
        }
    )

    # -------------------------
    # Add missing optional columns
    # -------------------------
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["team_goals"] = df["team_h_score"].where(
        df["is_home"],
        df["team_a_score"],
    )

    df["opponent_goals"] = df["team_a_score"].where(
        df["is_home"],
        df["team_h_score"],
    )


    df["player_id"] = df["player_id"].astype(int)
    df = df.rename(columns={"player_id": "gameweek_player_id"})

    # -------------------------
    # Add player position
    # -------------------------
    df = df.merge(
        players,
        on="gameweek_player_id",
        how="left",
        validate="many_to_one",
    )

    if df["player_id"].isna().any():
        unmatched = (
            df.loc[df["player_id"].isna(), "gameweek_player_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"Could not match these gameweek player IDs to players.csv: {unmatched}"
        )

    df = df.drop(columns="gameweek_player_id")

    # -------------------------
    # Add team metadata
    # -------------------------
    team_lookup = teams[
        [
            "team_code",
            "team_id",
            "team_name",
            "team_strength",
        ]
    ]

    df = df.merge(
        team_lookup,
        left_on="team_code",
        right_on="team_code",
        how="left",
        validate="many_to_one",
        suffixes=("", "_from_teams"),
    )

    # -------------------------
    # Opponent team metadata
    # -------------------------
    opponent_teams = teams[
        [
            "team_id",
            "team_code",
            "team_strength",
        ]
    ].rename(
        columns={
            "team_id": "opponent_id",
            "team_code": "opponent_team_code",
            "team_strength": "opponent_team_strength",
        }
    )

    df = df.merge(
        opponent_teams,
        on="opponent_team_code",
        how="left",
        validate="many_to_one",
    )

    # -------------------------
    # Convert numeric and string columns
    # -------------------------
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string")


    # -------------------------
    # Check for failed team matches
    # -------------------------
    if df["team_id"].isna().any():
        unmatched = (
            df.loc[df["team_id"].isna(), "team_name"]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Could not match these team names to teams.csv: {unmatched}"
        )

    return df


# -------------------------
# Load season
# -------------------------
season_dir = RAW_DIR / SEASON
df = load_season(season_dir)

# -------------------------
# Select output columns
# -------------------------
columns = [
    "season",
    "gw",
    "fixture_id",

    "player_id",
    "position",

    "team_id",
    "team_code",
    "team_strength",

    "opponent_id",
    "opponent_team_code",
    "opponent_team_strength",
    "opponent_team_promoted",
    
    "is_home",
    "team_goals",
    "opponent_goals",

    "minutes",

    "goals_scored",
    "goals_conceded",
    "xG",
    "xGC",
    "xA",
    "assists",
    "clean_sheets",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "bonus",
    "bps",
    "creativity",
    "ict_index",
    "influence",
    "threat",
    
    "total_points",
]

df = df[columns]

# -------------------------
# Save
# -------------------------
output_dir = PROCESSED_DIR / SEASON
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "player_gameweeks.parquet"

# selected = [
#     "season",
#     "gw",
#     "fixture_id",
#     "player_id",
#     "team_code",
#     "opponent_team_code",
#     "opponent_team_promoted",
# ]

# print(df[selected].sample(10))

df.to_parquet(output_path, index=False)

print(f"Saved {len(df):,} rows to {output_path}")
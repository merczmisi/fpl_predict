from pathlib import Path
import pandas as pd

processed_dir = Path("data/processed")

start_season = 2026
end_season = 2026

seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(start_season, end_season + 1)]

dfs = []

for season in seasons:
    file = processed_dir / season / "player_gameweeks.parquet"

    if not file.exists():
        print(f"Skipping missing season: {season}")
        continue

    print(f"Reading {season}")
    df = pd.read_parquet(file)
    df["season"] = season
    dfs.append(df)

if not dfs:
    raise FileNotFoundError("No season files found.")

combined = pd.concat(dfs, ignore_index=True)

duplicates = combined.duplicated(
    subset=["season", "fixture_id", "player_id"]
)
assert not duplicates.any()

assert combined["opponent_id"].notna().all()

assert combined["gw"].between(1, 38).all()

# print(
#     combined
#     .groupby(["season", "gw"])
#     .size()
#     .to_string()
# )

combined["fixture_data_missing"] = (
    combined["team_goals"].isna() &
    combined["opponent_goals"].isna()
)

selected = [
    "season",
    "gw",
    "fixture_id",
    "player_id",
    "fixture_data_missing",
]

# print(combined[selected].sample(10))

# print(
#     combined.loc[combined["fixture_data_missing"],
#            ["season", "gw", "fixture_id"]]
#     .drop_duplicates()
# )

output_path = processed_dir / f"player_gameweeks_combined_{start_season}_{end_season}.parquet"
combined.to_parquet(output_path, index=False)

print(f"\nCombined {len(dfs)} seasons")
print(f"Rows: {len(combined):,}")
print(f"Saved to: {output_path}")
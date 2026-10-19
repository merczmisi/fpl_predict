import json
import requests

BASE_URL = "https://www.fantasynutmeg.com/api/history/season/2022-23"

session = requests.Session()

# Get all teams
response = session.get(BASE_URL, timeout=10)
response.raise_for_status()

teams = response.json()["team_history"]

print(f"Found {len(teams)} teams")

# Fetch and save each team's history
for team in teams:
    team_id = team["team_id"]
    team_name = team["team_name"]

    print(f"Fetching {team_name} ({team_id})...")

    response = session.get(BASE_URL, params={"team": team_id}, timeout=10)
    response.raise_for_status()

    history = response.json()

    filename = f"raw/2022-23/teams/{team_id}_history.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"  Saved {filename}")

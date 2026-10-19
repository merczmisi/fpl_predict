import json
import requests

URL = "https://www.fantasynutmeg.com/api/history/season/2020-21"

response = requests.get(URL, timeout=10)
response.raise_for_status()

players = response.json()["history"]

players_by_id = {
    str(player["id"]): {
        "web_name": player["web_name"],
        "position": player["position"],
        "team_name": player["team_name"],
    }
    for player in players
}


web_names = [player["web_name"] for player in players]

if len(web_names) == len(set(web_names)):
    print("All web_names are unique.")
else:
    print("Duplicate web_names found!")

    duplicates = {
        (name, str(player["id"]))
        for name in web_names
        for player in players
        if player["web_name"] == name and web_names.count(name) > 1
    }

    for name, id in sorted(duplicates):
        print(name, id)

# with open("players.json", "w", encoding="utf-8") as f:
#     json.dump(players_by_id, f, indent=2, ensure_ascii=False)

# print(f"Saved {len(players_by_id)} players")

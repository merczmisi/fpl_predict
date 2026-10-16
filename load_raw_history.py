import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.fantasynutmeg.com/api/history/season/2019-20"

session = requests.Session()


def get_player_history(player):
    player_id = player["id"]
    web_name = player["web_name"]

    try:
        response = session.get(
            BASE_URL,
            params={"player": player_id},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        history = data.get("history") or []

        return player_id, web_name, history

    except requests.RequestException as e:
        print(f"ERROR {web_name} ({player_id}): {e}")
        return player_id, web_name, []


# Get all players
response = session.get(BASE_URL, timeout=10)
response.raise_for_status()

players = response.json()["history"]

print(f"Found {len(players)} players")


# GW -> player -> stats
gameweeks = {gw: {} for gw in range(1, 39)}


# Fetch players concurrently
with ThreadPoolExecutor(max_workers=10) as executor:

    futures = [
        executor.submit(get_player_history, player)
        for player in players
    ]

    for i, future in enumerate(as_completed(futures), 1):

        player_id, web_name, history = future.result()
        
        for game in history:
            gw = game["round"]

            if gw in gameweeks:
                game["id"] = player_id
                game["web_name"] = web_name

                gameweeks[gw][player_id] = game

        print(f"{i}/{len(players)}: {web_name}")


# Save GW files
for gw, players_data in gameweeks.items():

    filename = f"raw/2019-20/gw{gw}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            players_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved {filename}: {len(players_data)} players")
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SEASON = "2025-26"
BASE_URL = f"https://www.fantasynutmeg.com/api/history/season/{SEASON}"
OUTPUT_DIR = Path("data","raw", SEASON, "players")

MAX_WORKERS = 20


def create_session():
    """Create a requests session with retries."""
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def get_player_ids():
    """Get all player IDs from the season endpoint."""
    session = create_session()

    response = session.get(BASE_URL, timeout=30)
    response.raise_for_status()

    data = response.json()

    # The first endpoint returns one record per player.
    player_ids = [
        player["id"]
        for player in data["history"]
        if player.get("id") is not None
    ]

    return player_ids


def fetch_player_history(player_id):
    """Fetch the complete gameweek history for one player."""
    session = create_session()

    response = session.get(
        BASE_URL,
        params={"player": player_id},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    return player_id, data.get("history", [])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching player IDs...")
    player_ids = get_player_ids()

    # Remove duplicates while preserving order.
    player_ids = list(dict.fromkeys(player_ids))

    print(f"Found {len(player_ids)} players")

    # round -> list of records
    gameweeks = {
        gw: []
        for gw in range(1, 39)
    }

    completed = 0
    failed = []

    print("Fetching player histories...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_player_history, player_id): player_id
            for player_id in player_ids
        }

        for future in as_completed(futures):
            player_id = futures[future]

            try:
                player_id, history = future.result()

                for record in history:
                    round_number = record.get("round")

                    if round_number not in gameweeks:
                        continue

                    # Keep the original player ID explicitly.
                    record = {
                        "player_id": player_id,
                        **record,
                    }

                    gameweeks[round_number].append(record)

                completed += 1

                if completed % 50 == 0 or completed == len(player_ids):
                    print(
                        f"Completed {completed}/{len(player_ids)} players"
                    )

            except Exception as e:
                print(f"FAILED player {player_id}: {e}")
                failed.append(player_id)

    # Sort records to make the output deterministic.
    for gw in gameweeks:
        gameweeks[gw].sort(
            key=lambda x: (
                x.get("player_id", 0),
                x.get("kickoff_time") or "",
            )
        )

    # Write gw_1.json ... gw_38.json
    print("Writing gameweek files...")

    for gw in range(1, 39):
        output_file = OUTPUT_DIR / f"gw_{gw}.json"

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(
                gameweeks[gw],
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"GW {gw:2d}: "
            f"{len(gameweeks[gw]):5d} records -> {output_file}"
        )

    if failed:
        print()
        print(f"Failed players: {len(failed)}")
        print(failed)

        with (OUTPUT_DIR / "failed_players.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(failed, f, indent=2)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
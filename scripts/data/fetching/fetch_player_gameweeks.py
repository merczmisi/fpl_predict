import csv
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

SEASON = "2026-27"
BASE = Path("data/raw") / SEASON
API = "https://draft.premierleague.com/api"
DETAIL_RE = re.compile(r"^(?P<opp>\S+)\s+\((?P<ha>[HA])\)(?:\s+(?P<a>\d+)-(?P<b>\d+))?\s*$")


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def load_teams():
    path = BASE / "teams.csv"
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        promoted = {
            int(r["id"]): r["opponent_team_promoted"].strip().lower() == "true"
            for r in rows
        }
        code_to_id = {int(r["code"]): int(r["id"]) for r in rows}
        return {int(r["id"]): r["short_name"] for r in rows}, promoted, code_to_id
    teams = get_json(f"{API}/bootstrap-static")["teams"]
    return (
        {t["id"]: t["short_name"] for t in teams},
        {},
        {t["code"]: t["id"] for t in teams},
    )


def fetch_history(pid):
    for attempt in range(3):
        try:
            data = get_json(f"{API}/element-summary/{pid}")
            return pid, data.get("history", []), data.get("fixtures", [])
        except Exception as e:
            if attempt == 2:
                print(f"Failed player {pid}: {e}", file=sys.stderr)
    return pid, [], []


def main():
    teams, promoted, code_to_id = load_teams()
    with open(BASE / "players" / "players.csv", newline="", encoding="utf-8") as f:
        players = {int(r["id"]): code_to_id[int(r["team_code"])] for r in csv.DictReader(f)}

    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(fetch_history, players))

    by_gw = defaultdict(list)
    for pid, history, fixtures in results:
        for h in history:
            row = {"player_id": pid, **{k: v for k, v in h.items() if k != "id"}}
            m = DETAIL_RE.match(h.get("detail", ""))
            home = m["ha"] == "H" if m else None
            if m and m["a"] is not None:
                hs, as_ = (int(m["a"]), int(m["b"]))
                row["team_h_score"], row["team_a_score"] = hs, as_
            else:
                row["team_h_score"] = row["team_a_score"] = None
            row["was_home"] = home
            row["team_name"] = teams.get(players[pid])
            row["opponent_team_name"] = teams.get(h.get("opponent_team"))
            row["opponent_team_promoted"] = promoted.get(h.get("opponent_team"))
            by_gw[h["event"]].append(row)

    for pid, _, fixtures in results:
        if not fixtures:
            continue
        f = fixtures[0]
        by_gw[f["event"]].append({
            "player_id": pid,
            "event": f["event"],
            "team_name": teams.get(players[pid]),
            "opponent_team_name": teams.get(f["opponent"]),
            "opponent_team_promoted": promoted.get(f["opponent"]),
            "was_home": f["is_home"],
            "fixture": f["id"],
            "kickoff_time": f["kickoff_time"],
            "difficulty": f["difficulty"],
        })

    out = BASE / "players"
    out.mkdir(parents=True, exist_ok=True)
    for gw, rows in sorted(by_gw.items()):
        rows.sort(key=lambda r: r["player_id"])
        with open(out / f"gw_{gw}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"gw_{gw}.json: {len(rows)} rows")


if __name__ == "__main__":
    main()

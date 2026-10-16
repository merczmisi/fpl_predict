import csv
import json
from pathlib import Path
from urllib.request import urlopen

SEASON = "2026-27"
URL = "https://draft.premierleague.com/api/bootstrap-static"
OUT = Path("data/raw") / SEASON / "players" / "players.csv"
COLUMNS = ["code", "id", "team_code", "web_name", "element_type"]


def main():
    with urlopen(URL, timeout=30) as r:
        data = json.load(r)

    team_codes = {t["id"]: t["code"] for t in data["teams"]}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for e in data["elements"]:
            w.writerow([e["code"], e["id"], team_codes[e["team"]], e["web_name"], e["element_type"]])

    print(f"Wrote {len(data['elements'])} players to {OUT}")


if __name__ == "__main__":
    main()

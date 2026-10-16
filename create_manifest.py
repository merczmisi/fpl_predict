import json
from pathlib import Path

SEASON = "2019-20"
EXPECTED_MINIMUM = 333

manifest = {
    "season": SEASON,
    "expected_minimum": EXPECTED_MINIMUM,
    "gameweeks": {}
}

for gw in range(1, 39):
    filename = Path(f"raw/{SEASON}/gw{gw}.json")

    if not filename.exists():
        count = 0
    else:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = len(data)

    if count == 0:
        status = "missing"
    elif count < EXPECTED_MINIMUM:
        status = "suspicious"
    else:
        status = "ok"

    manifest["gameweeks"][str(gw)] = {
        "records": count,
        "status": status
    }

with open(f"raw/{SEASON}/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
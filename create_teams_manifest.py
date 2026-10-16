import json
import glob
import os

SEASON = "2022-23"

INPUT_DIR = f"data/raw/{SEASON}/teams"

OUTPUT_FILE = os.path.join(INPUT_DIR, "manifest.json")
GW_START = 1
GW_END = 38

manifest = {
    "season": SEASON,
    "teams": {}

}

pattern = os.path.join(INPUT_DIR, "*_history.json")

for filepath in glob.glob(pattern):
    filename = os.path.basename(filepath)
    
    # Extract team_id from e.g. "6_history.json"
    team_id = filename.removesuffix("_history.json")

    with open(filepath, "r", encoding="utf-8") as f:

        history = json.load(f)

    # Index records by gameweek
    by_gw = {
        record["gw"]: record
        for record in history
        if "gw" in record
    }

    ok = 0
    suspicious = 0
    missing = []

    for gw in range(GW_START, GW_END + 1):
        record = by_gw.get(gw)

        if record is None:
            missing.append(gw)
        elif record.get("total_points", 0) > 0:
            ok += 1
        else:
            suspicious += 1

    manifest["teams"][team_id] = {
        "ok": ok,
        "suspicious": suspicious,
        "missing": missing
    }

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(manifest, f, indent=2)

print(f"Created {OUTPUT_FILE}")
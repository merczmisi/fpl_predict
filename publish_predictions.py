"""Publish next-gameweek predictions for all players as static JSON.

Writes predictions/gw_{N}.json and predictions/latest.json.
Players without a fixture in the target gameweek get pred_total_points = 0.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from predict_next_gw import (
    CURRENT_SEASON,
    MODEL_PATH,
    PLAYERS_PATH,
    detect_target_gw,
    predict_players,
)

OUT_DIR = Path("predictions")


def build_payload(target_gw):
    preds = predict_players(None, target_gw)
    players = pd.read_csv(PLAYERS_PATH, usecols=["id", "code", "web_name", "team_code", "element_type"])
    position_by_type = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    preds = preds[["id", "pred_total_points", "opponent_team_strength"]]
    merged = players.merge(preds, on="id", how="left")
    merged["has_fixture"] = merged["pred_total_points"].notna()
    merged["pred_total_points"] = merged["pred_total_points"].fillna(0).round(3)
    merged["position"] = merged["element_type"].map(position_by_type)
    merged = merged.sort_values("pred_total_points", ascending=False)

    records = [
        {
            "id": int(r.id),
            "web_name": r.web_name,
            "position": r.position,
            "team_code": int(r.team_code),
            "has_fixture": bool(r.has_fixture),
            "pred_total_points": float(r.pred_total_points),
        }
        for r in merged.itertuples()
    ]
    return {
        "season": CURRENT_SEASON,
        "gw": target_gw,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, timezone.utc).strftime("%Y%m%d"),
        "players": records,
    }


def write_atomic(path, payload):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main():
    target_gw = detect_target_gw()
    payload = build_payload(target_gw)
    OUT_DIR.mkdir(exist_ok=True)
    write_atomic(OUT_DIR / f"gw_{target_gw}.json", payload)
    write_atomic(OUT_DIR / "latest.json", payload)
    n_fix = sum(p["has_fixture"] for p in payload["players"])
    print(f"gw{target_gw}: {len(payload['players'])} players ({n_fix} with fixture) -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

"""Publish next-gameweek predictions for all players as static JSON.

Writes predictions/gw_{N}.json and predictions/latest.json.
Players without a fixture in the target gameweek get pred_total_points = 0.

Supports both v1 classifier predictions and FFPundit shadow-mode predictions.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.prediction.predict_next_gw import (
    CURRENT_SEASON,
    MODEL_PATH,
    PLAYERS_PATH,
    detect_target_gw,
    predict_players,
    predict_players_with_ffpundit,
)

OUT_DIR = Path("predictions")


def build_payload(target_gw, use_ffpundit=False):
    """Build prediction payload for JSON output.
    
    Args:
        target_gw: Gameweek to predict
        use_ffpundit: If True, use v4_ffpundit_points; else use v1 pred_total_points
    
    Returns:
        Dict with season, gw, generated_at, model_version, and players array
    """
    if use_ffpundit:
        preds = predict_players_with_ffpundit(None, target_gw, include_audit_cols=False)
        pred_col = "v4_ffpundit_points"
    else:
        preds = predict_players(None, target_gw)
        pred_col = "pred_total_points"
    
    players = pd.read_csv(PLAYERS_PATH, usecols=["id", "code", "web_name", "team_code", "element_type"])
    position_by_type = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    preds = preds[["id", pred_col, "opponent_team_strength"]]
    merged = players.merge(preds, on="id", how="left")
    merged["has_fixture"] = merged[pred_col].notna()
    merged[pred_col] = merged[pred_col].fillna(0).round(3)
    merged["position"] = merged["element_type"].map(position_by_type)
    merged = merged.sort_values(pred_col, ascending=False)

    records = [
        {
            "id": int(r.id),
            "web_name": r.web_name,
            "position": r.position,
            "team_code": int(r.team_code),
            "has_fixture": bool(r.has_fixture),
            "pred_total_points": float(getattr(r, pred_col)),
        }
        for r in merged.itertuples()
    ]
    
    model_suffix = "_ffpundit" if use_ffpundit else ""
    return {
        "season": CURRENT_SEASON,
        "gw": target_gw,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, timezone.utc).strftime("%Y%m%d") + model_suffix,
        "players": records,
    }


def write_atomic(path, payload):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Publish next-gameweek predictions")
    parser.add_argument("--gw", type=int, default=None, help="Gameweek (auto-detect if not specified)")
    parser.add_argument("--ffpundit", action="store_true", help="Publish FFPundit predictions instead of v1 classifier")
    args = parser.parse_args()
    
    target_gw = args.gw or detect_target_gw()
    payload = build_payload(target_gw, use_ffpundit=args.ffpundit)
    
    OUT_DIR.mkdir(exist_ok=True)
    
    model_suffix = "_ffpundit" if args.ffpundit else ""
    write_atomic(OUT_DIR / f"gw_{target_gw}{model_suffix}.json", payload)
    if not args.ffpundit:
        write_atomic(OUT_DIR / "latest.json", payload)
    else:
        write_atomic(OUT_DIR / "latest_ffpundit.json", payload)
    
    n_fix = sum(p["has_fixture"] for p in payload["players"])
    model_name = "FFPundit" if args.ffpundit else "v1 classifier"
    print(f"gw{target_gw} ({model_name}): {len(payload['players'])} players ({n_fix} with fixture) -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

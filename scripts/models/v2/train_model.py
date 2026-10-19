"""Train one LightGBM model per position (GKP, DEF, MID, FWD).

Same splits, metrics and hyperparameters as v1; each position gets its own
model, with irrelevant features dropped. Test predictions from all four
models are concatenated and evaluated both overall and per position.
"""

import importlib.util
from pathlib import Path

import lightgbm as lgb
import pandas as pd

V1_PATH = Path(__file__).resolve().parents[1] / "v1" / "train_model.py"
spec = importlib.util.spec_from_file_location("train_model_v1", V1_PATH)
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)

FEATURES_PATH = Path("data/processed/model_features_v2.parquet")
MODEL_DIR = Path("models/v2")
POSITIONS = ["GKP", "DEF", "MID", "FWD"]

# Stats that carry no signal for a position (goalkeepers rarely score/assist).
DROP_PATTERNS = {
    "GKP": ["goals_scored", "assists", "creativity", "threat"],
    "DEF": [],
    "MID": [],
    "FWD": ["clean_sheets", "goals_conceded", "opponent_goals"],
}


def position_features(feature_cols, position):
    patterns = DROP_PATTERNS[position]
    cols = [c for c in feature_cols if c != "position"]
    return [c for c in cols if not any(p in c for p in patterns)]


def main():
    df = pd.read_parquet(FEATURES_PATH)
    all_feature_cols = [c for c in df.columns if c not in v1.ID_COLS + [v1.LABEL_COL]]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for position in POSITIONS:
        pos_df = df[df["position"] == position]
        feats = position_features(all_feature_cols, position)
        train = pos_df[pos_df["season"].isin(v1.TRAIN_SEASONS)]
        val = pos_df[pos_df["season"].isin(v1.VAL_SEASONS)]
        test = pos_df[pos_df["season"].isin(v1.TEST_SEASONS)].copy()

        print(f"\n=== {position}: train {len(train):,} / val {len(val):,} / "
              f"test {len(test):,}, {len(feats)} features ===")

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(
            train[feats],
            train[v1.LABEL_COL],
            eval_set=[(val[feats], val[v1.LABEL_COL])],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100),
            ],
        )

        test["pred"] = model.predict(test[feats])
        metrics, _ = v1.evaluate_predictions(test, "pred")
        v1.print_metrics(position, metrics)

        model.booster_.save_model(str(MODEL_DIR / f"lgbm_total_points_{position}.txt"))
        results.append(test)

    combined = pd.concat(results)
    metrics, _ = v1.evaluate_predictions(combined, "pred")
    v1.print_metrics("ALL POSITIONS (combined)", metrics)
    print(f"\nSaved models to {MODEL_DIR}")


if __name__ == "__main__":
    main()

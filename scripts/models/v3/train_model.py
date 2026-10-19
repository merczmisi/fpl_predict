"""Train a v3 LightGBM model to predict `total_points` for a player-gameweek.

Split strategy (walk-forward across seasons, no shuffling):
  - train:      2022-23 (gw16+), 2023-24
  - validation: 2024-25  (used for early stopping / hyperparameter checks)
  - test:       2025-26  (held out, reported as final numbers)

v3 adds xG/xGC/xA form features, which exist only from 2022-23 gw16, so
earlier rows are excluded.

Primary training objective is plain regression (L2). We report both
error metrics (MAE/RMSE) and ranking metrics (per-gameweek Spearman
correlation, precision at several cutoffs, and NDCG) since ranking
quality matters more for FPL decisions (captaincy/transfers) than exact
point accuracy.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURES_PATH = Path("data/processed/model_features_v3.parquet")
MODEL_DIR = Path("models")

TRAIN_SEASONS = ["2022-23", "2023-24"]
XSTATS_FIRST_SEASON = "2022-23"
XSTATS_FIRST_GW = 16
VAL_SEASONS = ["2024-25"]
TEST_SEASONS = ["2025-26"]

ID_COLS = ["season", "gw", "player_id", "team_id", "team_code"]
LABEL_COL = "total_points"
CATEGORICAL_COLS = ["position"]

PRECISION_CUTOFFS = (1, 3, 5, 10, 20)
NDCG_CUTOFFS = (5, 10, 20)


def load_splits():
    df = pd.read_parquet(FEATURES_PATH)
    df["position"] = df["position"].astype("category")

    # Drop rows from before expected stats were tracked.
    has_xstats = (df["season"] > XSTATS_FIRST_SEASON) | (
        (df["season"] == XSTATS_FIRST_SEASON) & (df["gw"] >= XSTATS_FIRST_GW)
    )
    df = df[has_xstats]

    feature_cols = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]

    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"].isin(VAL_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]

    return train, val, test, feature_cols


def train_model(train, val, feature_cols):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        train[feature_cols],
        train[LABEL_COL],
        eval_set=[(val[feature_cols], val[LABEL_COL])],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def ranking_metrics(df, y_true_col, y_pred_col):
    """Return mean per-gameweek rank metrics.

    Precision uses overlap between the predicted and actual top-k players.
    NDCG uses nonnegative actual points as linear relevance gains.
    """
    spearmans = []
    precisions = {k: [] for k in PRECISION_CUTOFFS}
    ndcgs = {k: [] for k in NDCG_CUTOFFS}

    for _, group in df.groupby(["season", "gw"]):
        if group.empty:
            continue

        if len(group) >= 2 and group[y_true_col].nunique() >= 2:
            corr, _ = spearmanr(group[y_true_col], group[y_pred_col])
            if not np.isnan(corr):
                spearmans.append(corr)

        for k in precisions:
            cutoff = min(k, len(group))
            true_top = set(group.nlargest(cutoff, y_true_col)["player_id"])
            pred_top = set(group.nlargest(cutoff, y_pred_col)["player_id"])
            precisions[k].append(len(true_top & pred_top) / cutoff)

        relevance = group[y_true_col].clip(lower=0).to_numpy()
        predicted_order = np.argsort(-group[y_pred_col].to_numpy(), kind="stable")
        ideal_order = np.argsort(-relevance, kind="stable")
        for k in ndcgs:
            discounts = np.log2(np.arange(2, min(k, len(group)) + 2))
            dcg = np.sum(relevance[predicted_order[:k]] / discounts)
            idcg = np.sum(relevance[ideal_order[:k]] / discounts)
            ndcgs[k].append(float(dcg / idcg) if idcg else 0.0)

    metrics = {
        "mean_spearman_per_gw": float(np.mean(spearmans)) if spearmans else float("nan"),
        "num_gameweeks_evaluated": len(spearmans),
    }
    metrics.update({
        f"mean_p@{k}_per_gw": float(np.mean(values))
        for k, values in precisions.items()
    })
    metrics.update({
        f"mean_ndcg@{k}_per_gw": float(np.mean(values))
        for k, values in ndcgs.items()
    })
    return metrics


def evaluate_predictions(test, pred_col):
    """Compute error + ranking metrics given a test df with a prediction column."""
    result = test[["season", "gw", "player_id", LABEL_COL, pred_col]].copy()
    result = result.rename(columns={pred_col: "pred"})

    mae = (result[LABEL_COL] - result["pred"]).abs().mean()
    rmse = np.sqrt(((result[LABEL_COL] - result["pred"]) ** 2).mean())
    overall_spearman, _ = spearmanr(result[LABEL_COL], result["pred"])

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "overall_spearman": float(overall_spearman),
    }
    metrics.update(ranking_metrics(result, LABEL_COL, "pred"))

    return metrics, result


def evaluate(model, test, feature_cols):
    test = test.copy()
    test["pred"] = model.predict(test[feature_cols])
    return evaluate_predictions(test, "pred")


def print_metrics(name, metrics):
    print(f"\n{name} test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def main():
    train, val, test, feature_cols = load_splits()

    print(f"Train rows: {len(train):,} ({TRAIN_SEASONS})")
    print(f"Val rows:   {len(val):,} ({VAL_SEASONS})")
    print(f"Test rows:  {len(test):,} ({TEST_SEASONS})")
    print(f"Features:   {len(feature_cols)}")

    model = train_model(train, val, feature_cols)

    metrics, result = evaluate(model, test, feature_cols)

    print_metrics("LightGBM", metrics)

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(importances.head(15).to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_DIR / "lgbm_total_points_v3.txt"))
    print(f"\nSaved model to {MODEL_DIR / 'lgbm_total_points_v3.txt'}")


if __name__ == "__main__":
    main()

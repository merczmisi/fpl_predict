"""Train three independent LightGBM models using v4 features.

Models:
  1. Start classifier (binary): P(starts=1) on all rows with known starts
  2. Starter points regression: E[points | starts=1]
  3. Non-starter points regression: E[points | starts=0]

Split strategy (chronological, no shuffling):
  - train:      2022-23 GW16 onward through 2023-24 (known starts labels only)
  - validation: 2024-25 (early stopping, error metrics)
  - test:       2025-26 (final evaluation, probability calibration for classifier)

Hyperparameters baseline from v1:
  n_estimators=2000, learning_rate=0.03, num_leaves=31,
  subsample=0.8, colsample_bytree=0.8
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr

FEATURES_PATH = Path("data/processed/model_features_v4.parquet")
MODEL_DIR = Path("models/v4")

ID_COLS = ["season", "gw", "player_id", "team_id", "team_code"]
LABEL_COL = "total_points"
STARTS_COL = "starts"
CATEGORICAL_COLS = ["position"]

# Training split: 2022-23 GW16 onward, chronologically through 2023-24
TRAIN_SEASONS = ["2022-23", "2023-24"]
TRAIN_START_GW = 16  # Only 2022-23 GW16+

VAL_SEASONS = ["2024-25"]
TEST_SEASONS = ["2025-26"]

# LightGBM hyperparameters (baseline from v1)
LIGHTGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def load_splits():
    """Load features, apply training split (known starts only), return splits."""
    df = pd.read_parquet(FEATURES_PATH)
    df["position"] = df["position"].astype("category")

    # Convert string columns to category for LightGBM
    string_cols = df.select_dtypes(include=['string', 'object']).columns.tolist()
    for col in string_cols:
        if col not in ID_COLS:
            df[col] = df[col].astype("category")

    feature_cols = [
        c
        for c in df.columns
        if c not in ID_COLS + [LABEL_COL, STARTS_COL]
    ]

    # Training: 2022-23 GW16+ and 2023-24, with known starts only
    train_mask = (
        ((df["season"] == "2022-23") & (df["gw"] >= TRAIN_START_GW))
        | (df["season"] == "2023-24")
    ) & (df[STARTS_COL].notna())

    # Validation: 2024-25, with known starts only
    val_mask = (df["season"].isin(VAL_SEASONS)) & (df[STARTS_COL].notna())

    # Test: 2025-26, with known starts only
    test_mask = (df["season"].isin(TEST_SEASONS)) & (df[STARTS_COL].notna())

    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()

    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Test samples: {len(test)}")

    # Splits by starts label
    train_starters = train[train[STARTS_COL] == 1]
    train_nonstarters = train[train[STARTS_COL] == 0]
    print(f"Train starters: {len(train_starters)}, non-starters: {len(train_nonstarters)}")

    return train, val, test, feature_cols


def train_classifier(train, val, feature_cols):
    """Train binary starts classifier on all rows with known starts."""
    model = lgb.LGBMClassifier(
        **LIGHTGBM_PARAMS,
        objective="binary",
        is_unbalance=True,
    )

    model.fit(
        train[feature_cols],
        train["starts"],
        eval_set=[(val[feature_cols], val["starts"])],
        eval_metric="auc",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def train_regression(train, val, feature_cols, starts_label, label_value):
    """Train regression model on subset of rows.

    starts_label: "starts" column name
    label_value: value to filter on (1 for starters, 0 for non-starters)
    """
    train_subset = train[train[starts_label] == label_value]
    val_subset = val[val[starts_label] == label_value]

    if len(train_subset) < 10:
        print(f"Warning: only {len(train_subset)} training samples for {starts_label}={label_value}")

    model = lgb.LGBMRegressor(
        **LIGHTGBM_PARAMS,
        objective="regression",
    )

    model.fit(
        train_subset[feature_cols],
        train_subset["total_points"],
        eval_set=[(val_subset[feature_cols], val_subset["total_points"])],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def compute_per_gw_metrics(df, actual_col, predicted_col):
    """Return mean per-gameweek rank metrics (Spearman, precision, NDCG, top-k)."""
    precision_cutoffs = (1, 3, 5, 10, 20)
    ndcg_cutoffs = (5, 10, 20)
    top_pick_cutoffs = (5, 11)

    spearmans = []
    precisions = {k: [] for k in precision_cutoffs}
    ndcgs = {k: [] for k in ndcg_cutoffs}
    top_pick_points = {k: [] for k in top_pick_cutoffs}

    for _, group in df.groupby(["season", "gw"]):
        if group.empty:
            continue

        if len(group) >= 2 and group[actual_col].nunique() >= 2:
            corr, _ = spearmanr(group[actual_col], group[predicted_col])
            if not np.isnan(corr):
                spearmans.append(corr)

        for k in precisions:
            cutoff = min(k, len(group))
            true_top = set(group.nlargest(cutoff, actual_col)["player_id"])
            pred_top = set(group.nlargest(cutoff, predicted_col)["player_id"])
            overlap = len(true_top & pred_top)
            precisions[k].append(overlap / cutoff if cutoff > 0 else 0)

        for k in ndcgs:
            cutoff = min(k, len(group))
            pred_order = group.nlargest(cutoff, predicted_col).index
            gains = (group.loc[pred_order, actual_col].clip(lower=0).values
                     / np.log2(np.arange(2, cutoff + 2)))
            ndcgs[k].append(gains.sum())

        for k in top_pick_cutoffs:
            cutoff = min(k, len(group))
            top_pred = group.nlargest(cutoff, predicted_col)
            top_pick_points[k].append(top_pred["total_points"].mean())

    result = {
        "spearman": np.mean(spearmans) if spearmans else np.nan,
    }

    for k in precisions:
        result[f"precision@{k}"] = np.mean(precisions[k]) if precisions[k] else np.nan

    for k in ndcgs:
        result[f"ndcg@{k}"] = np.mean(ndcgs[k]) if ndcgs[k] else np.nan

    for k in top_pick_cutoffs:
        result[f"top{k}_mean_points"] = (
            np.mean(top_pick_points[k]) if top_pick_points[k] else np.nan
        )

    return result


def evaluate_classifier(model, test, feature_cols):
    """Evaluate classifier: ROC-AUC, calibration bins."""
    from sklearn.metrics import roc_auc_score, brier_score_loss

    probs = model.predict_proba(test[feature_cols])[:, 1]
    test_copy = test.copy()
    test_copy["predicted_prob"] = probs

    auc = roc_auc_score(test["starts"], probs)
    brier = brier_score_loss(test["starts"], probs)

    # Probability calibration: bin predictions, compare to actual start rate
    bins = np.arange(0, 1.01, 0.1)
    test_copy["bin"] = pd.cut(probs, bins=bins, include_lowest=True)

    calibration = test_copy.groupby("bin", observed=True).agg({
        "predicted_prob": "mean",
        "starts": "mean",
    }).rename(columns={"starts": "actual_start_rate"})

    print("Classifier Metrics (Test 2025-26):")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print("\nProbability Calibration (binned P(start) vs actual start rate):")
    print(calibration)

    return {"auc": auc, "brier": brier, "calibration": calibration}


def evaluate_regression(model, test, feature_cols, model_name):
    """Evaluate regression model: Spearman, MAE, RMSE, ranking metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    preds = model.predict(test[feature_cols])
    test_copy = test.copy()
    test_copy["predicted_points"] = preds

    mae = mean_absolute_error(test["total_points"], preds)
    rmse = np.sqrt(mean_squared_error(test["total_points"], preds))

    rank_metrics = compute_per_gw_metrics(test_copy, "total_points", "predicted_points")

    print(f"\n{model_name} Metrics (Test 2025-26):")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Spearman: {rank_metrics.get('spearman', np.nan):.4f}")
    for k in (1, 3, 5, 10, 20):
        print(f"  Precision@{k}: {rank_metrics.get(f'precision@{k}', np.nan):.4f}")

    return {"mae": mae, "rmse": rmse, "rank_metrics": rank_metrics}


def save_models(classifier, starter_model, nonstarter_model):
    """Save all three models."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    classifier.booster_.save_model(str(MODEL_DIR / "classifier.txt"))
    starter_model.booster_.save_model(str(MODEL_DIR / "starter_regression.txt"))
    nonstarter_model.booster_.save_model(str(MODEL_DIR / "nonstarter_regression.txt"))

    # Also save as pickle for portability
    with open(MODEL_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)
    with open(MODEL_DIR / "starter_regression.pkl", "wb") as f:
        pickle.dump(starter_model, f)
    with open(MODEL_DIR / "nonstarter_regression.pkl", "wb") as f:
        pickle.dump(nonstarter_model, f)

    print(f"\nModels saved to {MODEL_DIR}")


def log_training_metrics(classifier, starter_model, nonstarter_model, train, val):
    """Log training metrics: n_samples, feature importance, validation curves."""
    print("\n=== Training Summary ===")
    print(f"Train samples (all): {len(train)}")
    print(f"Train starters: {len(train[train['starts'] == 1])}")
    print(f"Train non-starters: {len(train[train['starts'] == 0])}")
    print(f"Validation samples: {len(val)}")

    print("\nClassifier feature importance (top-10):")
    importances = classifier.feature_importances_
    feature_cols = classifier.feature_names_in_
    top_indices = np.argsort(importances)[-10:][::-1]
    for idx in top_indices:
        print(f"  {feature_cols[idx]}: {importances[idx]:.4f}")

    print("\nStarter regression feature importance (top-10):")
    importances = starter_model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    for idx in top_indices:
        print(f"  {feature_cols[idx]}: {importances[idx]:.4f}")

    print("\nNon-starter regression feature importance (top-10):")
    importances = nonstarter_model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    for idx in top_indices:
        print(f"  {feature_cols[idx]}: {importances[idx]:.4f}")


def main():
    print("Loading v4 features and splits...")
    train, val, test, feature_cols = load_splits()

    print("\n=== Training Start Classifier ===")
    classifier = train_classifier(train, val, feature_cols)

    print("\n=== Training Starter Points Regression ===")
    starter_model = train_regression(train, val, feature_cols, "starts", 1)

    print("\n=== Training Non-Starter Points Regression ===")
    nonstarter_model = train_regression(train, val, feature_cols, "starts", 0)

    print("\n=== Evaluating on Test Set (2025-26) ===")
    classifier_metrics = evaluate_classifier(classifier, test, feature_cols)
    starter_metrics = evaluate_regression(starter_model, test[test["starts"] == 1], feature_cols, "Starter Regression")
    nonstarter_metrics = evaluate_regression(nonstarter_model, test[test["starts"] == 0], feature_cols, "Non-Starter Regression")

    log_training_metrics(classifier, starter_model, nonstarter_model, train, val)

    save_models(classifier, starter_model, nonstarter_model)

    print("\n✓ Training complete")


if __name__ == "__main__":
    main()

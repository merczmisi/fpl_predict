"""Evaluate v4 against v1: Spearman, MAE, RMSE, signed bias on test and production data.

Runs both models on:
  - 2025-26: held-out test set (compare accuracy metrics)
  - 2026-27: production data (measure bias closure vs v1)

Output:
  - Console report with metrics side-by-side
  - CSV with per-gameweek and per-player results
  - JSON summary for downstream pipelines
"""

from pathlib import Path
from typing import Tuple
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

# Paths
FEATURES_V1 = Path("data/processed/model_features.parquet")
FEATURES_V4 = Path("data/processed/model_features_v4.parquet")
MODEL_V1_PATH = Path("models/lgbm_total_points.txt")
MODEL_V4_DIR = Path("models/v4")

# Feature configuration from v1 and v4
V1_FEATURE_COLS = None  # Loaded from trained model
V4_FEATURE_COLS = [
    "position", "opponent_id", "opponent_team_code", "is_home_frac",
    "num_fixtures", "team_strength", "opponent_team_strength", "opponent_team_promoted",
    "career_games_played", "gws_since_last_appearance",
    "minutes_roll3", "minutes_roll5", "minutes_roll10", "minutes_career_avg",
    "total_points_roll3", "total_points_roll5", "total_points_roll10", "total_points_career_avg",
    "goals_scored_roll3", "goals_scored_roll5", "goals_scored_roll10", "goals_scored_career_avg",
    "assists_roll3", "assists_roll5", "assists_roll10", "assists_career_avg",
    "goals_conceded_roll3", "goals_conceded_roll5", "goals_conceded_roll10", "goals_conceded_career_avg",
    "clean_sheets_roll3", "clean_sheets_roll5", "clean_sheets_roll10", "clean_sheets_career_avg",
    "yellow_cards_roll3", "yellow_cards_roll5", "yellow_cards_roll10", "yellow_cards_career_avg",
    "red_cards_roll3", "red_cards_roll5", "red_cards_roll10", "red_cards_career_avg",
    "bonus_roll3", "bonus_roll5", "bonus_roll10", "bonus_career_avg",
    "bps_roll3", "bps_roll5", "bps_roll10", "bps_career_avg",
    "creativity_roll3", "creativity_roll5", "creativity_roll10", "creativity_career_avg",
    "ict_index_roll3", "ict_index_roll5", "ict_index_roll10", "ict_index_career_avg",
    "influence_roll3", "influence_roll5", "influence_roll10", "influence_career_avg",
    "threat_roll3", "threat_roll5", "threat_roll10", "threat_career_avg",
    "xG_roll3", "xG_roll5", "xG_roll10", "xG_career_avg",
    "xGC_roll3", "xGC_roll5", "xGC_roll10", "xGC_career_avg",
    "xA_roll3", "xA_roll5", "xA_roll10", "xA_career_avg",
    "goals_scored_per90_roll3", "goals_scored_per90_roll5", "goals_scored_per90_roll10",
    "assists_per90_roll3", "assists_per90_roll5", "assists_per90_roll10",
    "goals_conceded_per90_roll3", "goals_conceded_per90_roll5", "goals_conceded_per90_roll10",
    "bonus_per90_roll3", "bonus_per90_roll5", "bonus_per90_roll10",
    "bps_per90_roll3", "bps_per90_roll5", "bps_per90_roll10",
    "total_points_per90_roll3", "total_points_per90_roll5", "total_points_per90_roll10",
    "xG_per90_roll3", "xG_per90_roll5", "xG_per90_roll10",
    "xGC_per90_roll3", "xGC_per90_roll5", "xGC_per90_roll10",
    "xA_per90_roll3", "xA_per90_roll5", "xA_per90_roll10",
    "start_rate_roll3", "start_rate_roll5", "start_rate_roll10",
]

ID_COLS = ["player_id", "season", "gw", "team_id", "team_code"]
LABEL_COL = "total_points"


def load_v1_model() -> lgb.Booster:
    """Load v1 LightGBM model."""
    return lgb.Booster(model_file=str(MODEL_V1_PATH))


def load_v4_models() -> Tuple[lgb.Booster, lgb.Booster, lgb.Booster]:
    """Load v4 three-model ensemble."""
    classifier = lgb.Booster(model_file=str(MODEL_V4_DIR / "classifier.txt"))
    starter = lgb.Booster(model_file=str(MODEL_V4_DIR / "starter_regression.txt"))
    nonstarter = lgb.Booster(model_file=str(MODEL_V4_DIR / "nonstarter_regression.txt"))
    return classifier, starter, nonstarter


def predict_v1(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Predict with v1 (single regression model)."""
    feature_cols = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]
    X = df[feature_cols].copy()
    
    # Handle categorical and object columns
    for col in X.columns:
        if X[col].dtype == "category":
            X[col] = X[col].cat.codes
        elif X[col].dtype == "object" or X[col].dtype == "string":
            # Encode string columns
            X[col] = pd.factorize(X[col])[0]
    
    X = X.fillna(0).values.astype(np.float32)
    return model.predict(X)


def predict_v4(
    classifier: lgb.Booster,
    starter: lgb.Booster,
    nonstarter: lgb.Booster,
    df: pd.DataFrame,
) -> np.ndarray:
    """Predict with v4 (three-model ensemble with start probability)."""
    X = df[V4_FEATURE_COLS].copy()
    
    # Handle categorical and object columns
    for col in X.columns:
        if X[col].dtype == "category":
            X[col] = X[col].cat.codes
        elif X[col].dtype == "object" or X[col].dtype == "string":
            # Encode string columns
            X[col] = pd.factorize(X[col])[0]
    
    X = X.fillna(0).values.astype(np.float32)
    
    # P(start)
    p_start = classifier.predict(X)
    
    # E[pts | start=1] and E[pts | start=0]
    pts_start = starter.predict(X)
    pts_no_start = nonstarter.predict(X)
    
    # E[pts] = P(start) × E[pts|start] + (1-P(start)) × E[pts|¬start]
    pred = p_start * pts_start + (1 - p_start) * pts_no_start
    return pred


def compute_metrics_per_gw(actual: np.ndarray, pred: np.ndarray, gw: np.ndarray) -> dict:
    """Compute MAE, RMSE, Spearman per gameweek."""
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    
    # Spearman per gameweek
    spearman_scores = []
    for g in sorted(np.unique(gw)):
        mask = gw == g
        if mask.sum() > 1:
            corr, _ = spearmanr(actual[mask], pred[mask])
            if not np.isnan(corr):
                spearman_scores.append(corr)
    
    spearman_avg = np.mean(spearman_scores) if spearman_scores else np.nan
    spearman_std = np.std(spearman_scores) if spearman_scores else np.nan
    
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman_avg": spearman_avg,
        "spearman_std": spearman_std,
        "num_gws": len(spearman_scores),
    }


def compute_signed_bias(actual: np.ndarray, pred: np.ndarray, gw: np.ndarray) -> dict:
    """Compute mean signed error per gameweek."""
    mse = np.mean(pred - actual)
    
    # Per-gw signed bias
    gw_bias = {}
    for g in sorted(np.unique(gw)):
        mask = gw == g
        gw_bias[int(g)] = float(np.mean(pred[mask] - actual[mask]))
    
    return {
        "mean_signed_error": mse,
        "per_gw": gw_bias,
    }


def main():
    # Load data for each model from their respective feature sets
    print("Loading v1 feature data...")
    df_v1_full = pd.read_parquet(FEATURES_V1)
    df_v1_full["position"] = df_v1_full["position"].astype("category")
    
    print("Loading v4 feature data...")
    df_v4_full = pd.read_parquet(FEATURES_V4)
    df_v4_full["position"] = df_v4_full["position"].astype("category")
    
    # Split: 2025-26 test, 2026-27 production
    test_data_v1 = df_v1_full[df_v1_full["season"] == "2025-26"].copy()
    test_data_v4 = df_v4_full[df_v4_full["season"] == "2025-26"].copy()
    
    prod_data_v1 = df_v1_full[df_v1_full["season"] == "2026-27"].copy()
    prod_data_v4 = df_v4_full[df_v4_full["season"] == "2026-27"].copy()
    
    print(f"V1 Test set (2025-26): {len(test_data_v1)} rows")
    print(f"V4 Test set (2025-26): {len(test_data_v4)} rows")
    print(f"V1 Production set (2026-27): {len(prod_data_v1)} rows")
    print(f"V4 Production set (2026-27): {len(prod_data_v4)} rows")
    
    # Load models
    print("\nLoading models...")
    model_v1 = load_v1_model()
    classifier, starter, nonstarter = load_v4_models()
    
    # Results storage
    results = {
        "test_set": {},
        "production_set": {},
    }
    
    # === TEST SET (2025-26) ===
    if len(test_data_v1) > 0 and len(test_data_v4) > 0:
        print("\n=== Test Set (2025-26) ===")
        
        actual_test = test_data_v1[LABEL_COL].values
        gw_test = test_data_v1["gw"].values
        
        # V1 predictions
        pred_v1_test = predict_v1(model_v1, test_data_v1)
        metrics_v1_test = compute_metrics_per_gw(actual_test, pred_v1_test, gw_test)
        results["test_set"]["v1"] = metrics_v1_test
        
        print("\nV1 Metrics:")
        print(f"  MAE:            {metrics_v1_test['mae']:.4f}")
        print(f"  RMSE:           {metrics_v1_test['rmse']:.4f}")
        print(f"  Spearman (avg): {metrics_v1_test['spearman_avg']:.4f} ± {metrics_v1_test['spearman_std']:.4f}")
        print(f"  Gameweeks:      {metrics_v1_test['num_gws']}")
        
        # V4 predictions
        pred_v4_test = predict_v4(classifier, starter, nonstarter, test_data_v4)
        actual_test_v4 = test_data_v4[LABEL_COL].values
        gw_test_v4 = test_data_v4["gw"].values
        metrics_v4_test = compute_metrics_per_gw(actual_test_v4, pred_v4_test, gw_test_v4)
        results["test_set"]["v4"] = metrics_v4_test
        
        print("\nV4 Metrics:")
        print(f"  MAE:            {metrics_v4_test['mae']:.4f}")
        print(f"  RMSE:           {metrics_v4_test['rmse']:.4f}")
        print(f"  Spearman (avg): {metrics_v4_test['spearman_avg']:.4f} ± {metrics_v4_test['spearman_std']:.4f}")
        print(f"  Gameweeks:      {metrics_v4_test['num_gws']}")
        
        # Comparison
        print("\nImprovement (V4 - V1):")
        print(f"  MAE:            {metrics_v4_test['mae'] - metrics_v1_test['mae']:+.4f}")
        print(f"  RMSE:           {metrics_v4_test['rmse'] - metrics_v1_test['rmse']:+.4f}")
        print(f"  Spearman:       {metrics_v4_test['spearman_avg'] - metrics_v1_test['spearman_avg']:+.4f}")
        
        # Save predictions for analysis
        test_data_v1["pred_v1"] = pred_v1_test
        test_data_v1.to_parquet("predictions/test_set_v1_comparison.parquet")
        
        test_data_v4["pred_v4"] = pred_v4_test
        test_data_v4.to_parquet("predictions/test_set_v4_comparison.parquet")
    else:
        print("Insufficient test data for comparison")
    
    # === PRODUCTION SET (2026-27) ===
    if len(prod_data_v1) > 0 and len(prod_data_v4) > 0:
        print("\n=== Production Set (2026-27) ===")
        
        actual_prod = prod_data_v1[LABEL_COL].values
        gw_prod = prod_data_v1["gw"].values
        
        # V1 predictions
        pred_v1_prod = predict_v1(model_v1, prod_data_v1)
        bias_v1_prod = compute_signed_bias(actual_prod, pred_v1_prod, gw_prod)
        results["production_set"]["v1"] = bias_v1_prod
        
        print("\nV1 Bias:")
        print(f"  Mean Signed Error: {bias_v1_prod['mean_signed_error']:+.4f} pts/player")
        
        # V4 predictions
        pred_v4_prod = predict_v4(classifier, starter, nonstarter, prod_data_v4)
        actual_prod_v4 = prod_data_v4[LABEL_COL].values
        gw_prod_v4 = prod_data_v4["gw"].values
        bias_v4_prod = compute_signed_bias(actual_prod_v4, pred_v4_prod, gw_prod_v4)
        results["production_set"]["v4"] = bias_v4_prod
        
        print("\nV4 Bias:")
        print(f"  Mean Signed Error: {bias_v4_prod['mean_signed_error']:+.4f} pts/player")
        
        print("\nBias Closure (V4 - V1):")
        bias_improvement = bias_v4_prod["mean_signed_error"] - bias_v1_prod["mean_signed_error"]
        print(f"  Mean Signed Error: {bias_improvement:+.4f} pts/player")
        print(f"  Closure Percentage: {abs(bias_improvement) / abs(bias_v1_prod['mean_signed_error']) * 100:.1f}%")
        
        # Save predictions
        prod_data_v1["pred_v1"] = pred_v1_prod
        prod_data_v1.to_parquet("predictions/production_set_v1_comparison.parquet")
        
        prod_data_v4["pred_v4"] = pred_v4_prod
        prod_data_v4.to_parquet("predictions/production_set_v4_comparison.parquet")
    else:
        print("Insufficient production data for comparison")
    
    # Save summary
    print("\n=== Saving Summary ===")
    summary_path = Path("predictions/v4_vs_v1_evaluation_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")
    
    return results


if __name__ == "__main__":
    main()

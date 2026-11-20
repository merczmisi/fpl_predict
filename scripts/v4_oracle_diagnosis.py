"""Diagnostic: v4 performance with oracle start probability (actual starts).

Isolates whether v4 underperformance is from poor start classifier or flawed regressions.

Rebuilds v4 predictions using actual `starts` instead of classifier output,
then compares:
  - v4-classifier: v4 with predicted start probs (current)
  - v4-oracle: v4 with actual starts (hypothetical perfect info)

If oracle >> classifier: start classifier is the bottleneck.
If oracle ≈ classifier: regression models are the problem.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

MODEL_V4_DIR = Path("models/v4")

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

LABEL_COL = "total_points"


def compute_metrics_per_gw(actual, pred, gw):
    """Compute MAE, RMSE, Spearman per gameweek."""
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    
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
        "bias": float(np.mean(pred - actual)),
    }


def predict_v4_oracle(starter, nonstarter, df):
    """Predict v4 using actual starts instead of classifier."""
    X = df[V4_FEATURE_COLS].copy()
    
    # Handle categorical and object columns
    for col in X.columns:
        if X[col].dtype == "category":
            X[col] = X[col].cat.codes
        elif X[col].dtype == "object" or X[col].dtype == "string":
            X[col] = pd.factorize(X[col])[0]
    
    X = X.fillna(0).values.astype(np.float32)
    
    # Use actual starts instead of classifier
    p_start = df["starts"].values.astype(np.float32)
    
    # E[pts | start=1] and E[pts | start=0]
    pts_start = starter.predict(X)
    pts_no_start = nonstarter.predict(X)
    
    # E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]
    pred = p_start * pts_start + (1 - p_start) * pts_no_start
    return pred


# Load test predictions
print("Loading v4 test data...")
test_v4 = pd.read_parquet("predictions/test_set_v4_comparison.parquet")

actual = test_v4[LABEL_COL].values
gw = test_v4["gw"].values

# Load v4 models
print("Loading v4 models...")
starter = lgb.Booster(model_file=str(MODEL_V4_DIR / "starter_regression.txt"))
nonstarter = lgb.Booster(model_file=str(MODEL_V4_DIR / "nonstarter_regression.txt"))

# Predict with oracle starts
print("Computing v4 predictions with oracle (actual) start probability...")
pred_v4_oracle = predict_v4_oracle(starter, nonstarter, test_v4)

# Compute metrics
print("Computing metrics...\n")
pred_v4_classifier = test_v4["pred_v4"].values

metrics_classifier = compute_metrics_per_gw(actual, pred_v4_classifier, gw)
metrics_oracle = compute_metrics_per_gw(actual, pred_v4_oracle, gw)

# Output
print("=" * 70)
print("V4 DIAGNOSTIC: Oracle Start Information")
print("=" * 70)
print(f"\nTest Set: {len(test_v4)} player-gameweeks\n")

print(f"Start distribution in test set:")
print(f"  P(start=1): {test_v4['starts'].mean():.1%}")
print(f"  P(start=0): {(1 - test_v4['starts'].mean()):.1%}\n")

# Comparison table
print(f"{'Metric':<18} {'V4-Classifier':<18} {'V4-Oracle':<15} {'Improvement':<12}")
print("-" * 65)

mae_gain = metrics_classifier['mae'] - metrics_oracle['mae']
rmse_gain = metrics_classifier['rmse'] - metrics_oracle['rmse']
spearman_gain = metrics_oracle['spearman_avg'] - metrics_classifier['spearman_avg']
bias_gain = abs(metrics_oracle['bias']) - abs(metrics_classifier['bias'])

print(f"{'MAE':<18} {metrics_classifier['mae']:<18.4f} {metrics_oracle['mae']:<15.4f} {mae_gain:+.4f}")
print(f"{'RMSE':<18} {metrics_classifier['rmse']:<18.4f} {metrics_oracle['rmse']:<15.4f} {rmse_gain:+.4f}")
print(f"{'Spearman':<18} {metrics_classifier['spearman_avg']:<18.4f} {metrics_oracle['spearman_avg']:<15.4f} {spearman_gain:+.4f}")
print(f"{'Bias |error|':<18} {abs(metrics_classifier['bias']):<18.4f} {abs(metrics_oracle['bias']):<15.4f} {bias_gain:+.4f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if mae_gain > 0.05:
    print(f"\n✓ START CLASSIFIER is the bottleneck.")
    print(f"  Oracle gives {mae_gain:.4f} MAE improvement ({mae_gain/metrics_classifier['mae']*100:.1f}%)")
    print(f"  → Retraining/calibrating the start classifier could fix v4")
else:
    print(f"\n✗ REGRESSION MODELS are the problem (not classifier).")
    print(f"  Oracle only improves {mae_gain:.4f} MAE ({mae_gain/metrics_classifier['mae']*100:.1f}%)")
    print(f"  → Fundamental issue with v4's ensemble or feature engineering")

print(f"\nClassifier cost: {mae_gain:.4f} MAE")
print(f"  ({mae_gain/metrics_oracle['mae']*100:.1f}% of oracle's error)")

print(f"\nBias impact: Classifier makes v4 {'overpredict' if metrics_classifier['bias'] > 0 else 'underpredict'} by {abs(metrics_classifier['bias']):.4f}")
print(f"             Oracle {'overpredicts' if metrics_oracle['bias'] > 0 else 'underpredicts'} by {abs(metrics_oracle['bias']):.4f}")

# Summary
print("\n" + "=" * 70)
if mae_gain > 0.05:
    print("RECOMMENDATION: Retrain v4 start classifier and re-evaluate.")
else:
    print("RECOMMENDATION: v4 regressions need fundamental review; classifier tuning won't help.")

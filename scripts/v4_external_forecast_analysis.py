"""Future analysis: v4 with external start probability forecasts.

V4 architecture becomes pluggable: P(start) can come from multiple sources:
  1. Trained classifier (current)
  2. External forecasts (team news, injury reports, expert predictions)
  3. Ensemble of multiple sources

Scenario: What if P(start) came from an external source at varying quality levels?

This analyzes sensitivity: how does v4 performance depend on P(start) accuracy?
Helps justify investment in external forecast integration.
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
    
    return mae, rmse, spearman_avg


def predict_v4_with_external_forecast(starter, nonstarter, df, p_start_forecast):
    """Predict v4 using external P(start) forecast."""
    X = df[V4_FEATURE_COLS].copy()
    
    # Handle categorical and object columns
    for col in X.columns:
        if X[col].dtype == "category":
            X[col] = X[col].cat.codes
        elif X[col].dtype == "object" or X[col].dtype == "string":
            X[col] = pd.factorize(X[col])[0]
    
    X = X.fillna(0).values.astype(np.float32)
    
    # Use external forecast
    p_start = p_start_forecast.astype(np.float32)
    
    # E[pts | start=1] and E[pts | start=0]
    pts_start = starter.predict(X)
    pts_no_start = nonstarter.predict(X)
    
    # E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]
    pred = p_start * pts_start + (1 - p_start) * pts_no_start
    return pred


def generate_noisy_forecast(actual_starts, accuracy):
    """Generate synthetic forecast at given accuracy level.
    
    accuracy=1.0 → oracle (perfect)
    accuracy=0.5 → random (no signal)
    """
    forecast = actual_starts.copy()
    
    # Number of errors to introduce
    n_errors = int(len(forecast) * (1 - accuracy))
    
    # Randomly flip n_errors positions
    error_indices = np.random.choice(len(forecast), size=n_errors, replace=False)
    forecast[error_indices] = 1 - forecast[error_indices]
    
    return forecast


# Load test data
print("Loading v4 test data...")
test_v4 = pd.read_parquet("predictions/test_set_v4_comparison.parquet")

actual = test_v4[LABEL_COL].values
gw = test_v4["gw"].values
actual_starts = test_v4["starts"].values.astype(np.float32)

# Load v4 models
print("Loading v4 regression models...")
starter = lgb.Booster(model_file=str(MODEL_V4_DIR / "starter_regression.txt"))
nonstarter = lgb.Booster(model_file=str(MODEL_V4_DIR / "nonstarter_regression.txt"))

# Predict with current classifier (baseline)
print("Computing baseline (current classifier)...")
pred_classifier = test_v4["pred_v4"].values
mae_classifier, rmse_classifier, spearman_classifier = compute_metrics_per_gw(actual, pred_classifier, gw)

# Predict with oracle
print("Computing oracle (perfect forecast)...")
pred_oracle = predict_v4_with_external_forecast(starter, nonstarter, test_v4, actual_starts)
mae_oracle, rmse_oracle, spearman_oracle = compute_metrics_per_gw(actual, pred_oracle, gw)

# Predict at varying forecast quality levels
print("Computing scenarios at different forecast accuracies...\n")

accuracies = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
results = []

for acc in accuracies:
    # Generate synthetic forecast
    np.random.seed(42)
    forecast = generate_noisy_forecast(actual_starts.copy(), acc)
    
    # Predict with this forecast
    pred = predict_v4_with_external_forecast(starter, nonstarter, test_v4, forecast)
    mae, rmse, spearman = compute_metrics_per_gw(actual, pred, gw)
    
    results.append({
        'accuracy': acc,
        'mae': mae,
        'rmse': rmse,
        'spearman': spearman,
    })

# Output
print("=" * 80)
print("V4 SENSITIVITY ANALYSIS: External Start Probability Forecasts")
print("=" * 80)
print("\nScenario: P(start) provided by external source (team news, injury reports, etc.)")
print(f"Test Set: {len(test_v4)} player-gameweeks\n")

print(f"{'Forecast':<12} {'MAE':<10} {'vs V1':<10} {'vs Oracle':<12} {'Spearman':<10}")
print(f"{'Accuracy':<12} {'(pts)':<10} {'(δ pts)':<10} {'(δ pts)':<12} {'':10}")
print("-" * 70)

# V1 baseline for comparison
mae_v1 = 1.056
rmse_v1 = 1.971
spearman_v1 = 0.700

print(f"{'V1':<12} {mae_v1:<10.4f} {'—':<10} {mae_v1 - mae_oracle:+.4f}{'':<8} {spearman_v1:<10.4f}")
print(f"{'Current (clf)':<12} {mae_classifier:<10.4f} {mae_classifier - mae_v1:+.4f}{'':<6} {mae_classifier - mae_oracle:+.4f}{'':<8} {spearman_classifier:<10.4f}")

for result in results:
    acc = result['accuracy']
    mae = result['mae']
    spearman = result['spearman']
    
    label = f"{int(acc*100)}%"
    
    if acc == 1.0:
        label = "Oracle"
    
    delta_v1 = mae - mae_v1
    delta_oracle = mae - mae_oracle
    
    print(f"{label:<12} {mae:<10.4f} {delta_v1:+.4f}{'':<6} {delta_oracle:+.4f}{'':<8} {spearman:<10.4f}")

print("\n" + "=" * 80)
print("KEY THRESHOLDS")
print("=" * 80)

# Find accuracy level to match V1
for result in results:
    if result['mae'] <= mae_v1:
        print(f"\n✓ At {int(result['accuracy']*100)}% forecast accuracy, v4 matches V1 (MAE {result['mae']:.4f} vs {mae_v1:.4f})")
        break
else:
    print(f"\n✗ Even at 95% accuracy, v4 doesn't match V1")

# Find accuracy level to beat V1 by 5%
for result in results:
    if result['mae'] < mae_v1 * 0.95:
        print(f"✓ At {int(result['accuracy']*100)}% forecast accuracy, v4 beats V1 by 5% (MAE {result['mae']:.4f} vs {mae_v1:.4f})")
        break
else:
    print(f"✗ Even at 95% accuracy, v4 doesn't beat V1 by 5%")

print(f"\nOracle (100%) performance: MAE {mae_oracle:.4f} (Spearman {spearman_oracle:.4f})")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if mae_oracle < mae_v1 * 0.95:
    print("\n✓ INTEGRATE EXTERNAL FORECASTS.")
    print(f"  Oracle shows {(mae_v1 - mae_oracle)/mae_v1 * 100:.1f}% improvement over V1.")
    print(f"  Sources to explore:")
    print(f"    1. FPL team news API (team selections posted before deadline)")
    print(f"    2. Injury prediction models (trained on team-specific data)")
    print(f"    3. Tactical forecasts (match-day team sheet predictions)")
    print(f"    4. Expert consensus (betting odds, expert predictions)")
    print(f"    5. Ensemble of above")
    print(f"\n  Break-even: Need {int(np.interp(mae_v1, [result['mae'] for result in reversed(results)], [result['accuracy']*100 for result in reversed(results)])):.0f}% forecast accuracy to match V1.")
    print(f"  FPL team news ~90-95% accurate (posted by deadline).")
else:
    print("\n✗ EXTERNAL FORECASTS NOT JUSTIFIED.")
    print(f"  Oracle only improves {(mae_v1 - mae_oracle)/mae_v1 * 100:.1f}% over V1.")
    print(f"  Integration cost not worth the gain.")

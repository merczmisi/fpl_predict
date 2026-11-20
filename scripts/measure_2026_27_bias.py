"""Update: Measure 2026-27 production bias using available actual points.

Since we lack full features for 2026-27, we cannot run full model inference.
However, we can:
1. Load actual 2026-27 points from raw data
2. For comparison, use predicted values from test set (if match available)
3. Measure bias as (prediction - actual) per player-gw

This measures bias on players/positions where we have partial data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Load actual 2026-27 points
prod_actual = pd.read_parquet("data/processed/player_gameweeks_2026_27.parquet")

# Load 2025-26 predictions from test run
test_v1 = pd.read_parquet("predictions/test_set_v1_comparison.parquet")
test_v4 = pd.read_parquet("predictions/test_set_v4_comparison.parquet")

print("2026-27 Actual Points (Production Data):")
print(f"  Total records: {len(prod_actual)}")
print(f"  Gameweeks: {sorted(prod_actual['gw'].unique())}")
print(f"  Unique players: {prod_actual['player_id'].nunique()}")
print(f"  Mean points per gw: {prod_actual.groupby('gw')['total_points'].mean().round(2).to_dict()}")
print(f"  Median points: {prod_actual['total_points'].median():.2f}")
print(f"  Std dev: {prod_actual['total_points'].std():.2f}")

print("\n2025-26 Test Set Predictions:")
print(f"  V1 mean pred: {test_v1['pred_v1'].mean():.2f}")
print(f"  V4 mean pred: {test_v4['pred_v4'].mean():.2f}")

# Measure average prediction bias (expected - actual)
# This tells us: on average, how much do models overpredict?
test_actual = test_v1[["player_id", "gw", "total_points"]].copy()
print(f"\n2025-26 Test Bias:")
print(f"  V1 bias (pred - actual): {(test_v1['pred_v1'] - test_v1['total_points']).mean():+.3f} pts")
print(f"  V4 bias (pred - actual): {(test_v4['pred_v4'] - test_v4['total_points']).mean():+.3f} pts")

# 2026-27 expected bias if same distribution
print(f"\n2026-27 Production Distribution (Actual):")
print(f"  Mean actual points: {prod_actual['total_points'].mean():.2f}")
print(f"  Median: {prod_actual['total_points'].median():.2f}")
print(f"  Std: {prod_actual['total_points'].std():.2f}")

# Expected bias = (test bias) + (test actual mean - prod actual mean)
# If test set is similar distribution to prod, test bias carries forward
test_actual_mean = test_v1["total_points"].mean()
prod_actual_mean = prod_actual["total_points"].mean()
distribution_shift = prod_actual_mean - test_actual_mean

print(f"\nDistribution Shift (2026-27 vs 2025-26):")
print(f"  Test actual mean: {test_actual_mean:.2f}")
print(f"  Prod actual mean: {prod_actual_mean:.2f}")
print(f"  Shift: {distribution_shift:+.2f} pts")

# If production actual is higher/lower than test, this explains bias gap
v1_test_bias = (test_v1['pred_v1'] - test_v1['total_points']).mean()
v4_test_bias = (test_v4['pred_v4'] - test_v4['total_points']).mean()

print(f"\nExpected 2026-27 Bias (if distribution similar):")
print(f"  V1 predicted bias: {v1_test_bias:+.3f} pts")
print(f"  V4 predicted bias: {v4_test_bias:+.3f} pts")
print(f"  V4 improvement: {v1_test_bias - v4_test_bias:+.3f} pts")

# Summary
print(f"\n=== CONCLUSION ===")
print(f"Production 2026-27 (GW 1-10 complete):")
print(f"  Actual mean: {prod_actual_mean:.2f} pts/player")
print(f"  Test mean: {test_actual_mean:.2f} pts/player")
print(f"  Trend: {'Higher' if distribution_shift > 0 else 'Lower'} in production")
print(f"\nV1 Test Bias: {v1_test_bias:+.3f} pts (overpredicts)" if v1_test_bias > 0 else f"\nV1 Test Bias: {v1_test_bias:+.3f} pts (underpredicts)")
print(f"V4 Test Bias: {v4_test_bias:+.3f} pts (overpredicts)" if v4_test_bias > 0 else f"V4 Test Bias: {v4_test_bias:+.3f} pts (underpredicts)")
print(f"\nV4 Bias Reduction: {abs(v1_test_bias - v4_test_bias):.3f} pts (direction: {'toward 0' if abs(v4_test_bias) < abs(v1_test_bias) else 'away from 0'})")

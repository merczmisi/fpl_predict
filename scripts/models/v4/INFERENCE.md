# V4 Inference Pipeline

Fixture-level and gameweek-level prediction using three trained LightGBM models.

## Formula

For each fixture, expected points are computed as:

```
E[pts] = P(start) × E[pts | starts] + (1 − P(start)) × E[pts | ¬starts]
```

Where:
- **P(start)**: Classifier output (or external override)
- **E[pts | starts]**: Starter regression model prediction
- **E[pts | ¬starts]**: Non-starter regression model prediction

## Models

Three independent LightGBM models trained on v4 features:

1. **classifier.txt / classifier.pkl** – Binary classification of P(start)
2. **starter_regression.txt / starter_regression.pkl** – Points for starting players
3. **nonstarter_regression.txt / nonstarter_regression.pkl** – Points for non-starters

Models are stored in `models/v4/`.

## Usage

### Basic fixture-level prediction

```python
from scripts.models.v4.inference import predict_fixture_level
import pandas as pd

# fixture_data: DataFrame with 108 required v4 features (from build_features.py)
fixture_data = pd.read_parquet("data/processed/model_features_v4.parquet")

# Filter to target gameweeks
target_gw_data = fixture_data[fixture_data["gw"] == 1]

# Predict
predictions = predict_fixture_level(target_gw_data)

# Output columns: player_id, season, gw, predicted_points, predicted_start_prob, confidence
print(predictions.head())
```

### With external start probabilities

```python
# External start probs (e.g., from team news, manual input)
external_probs = pd.DataFrame({
    "player_id": [1, 2, 3],
    "season": ["2025-26", "2025-26", "2025-26"],
    "gw": [1, 1, 1],
    "external_start_prob": [0.95, 0.80, 0.50],
})

# Override classifier with external probs
predictions = predict_fixture_level(target_gw_data, external_start_probs=external_probs)

# For rows in external_probs, P(start) uses external value.
# For other rows, classifier is used.
```

### Gameweek aggregation

```python
from scripts.models.v4.inference import predict_gameweek_level

# Get fixture-level predictions first
fixture_preds = predict_fixture_level(fixture_data)

# Aggregate to GW level (sums fixture predictions for DGWs)
gw_preds = predict_gameweek_level(fixture_preds)

# Output: one row per (player_id, season, gw)
# predicted_points = sum of fixtures in that GW
# predicted_start_prob = mean of fixtures
# confidence = mean of fixtures
```

### Full pipeline

```python
from scripts.models.v4.inference import predict_full_dataset

# In one call:
gw_level = predict_full_dataset(fixture_data, aggregate_to_gw=True)

# Or keep fixture-level:
fixture_level = predict_full_dataset(fixture_data, aggregate_to_gw=False)
```

## Features

Requires 108 features from v4 feature engineering:

- Fixture context: position, opponent, home/away, promoted flag
- Rolling means (3, 5, 10 GW windows) for: minutes, points, goals, assists, etc.
- Career averages for each stat
- Per-90 metrics for rolling windows
- Start rate rolling means

See [build_features.py](build_features.py) for feature computation.

## Output

### Fixture-level DataFrame

| Column | Type | Range | Notes |
|--------|------|-------|-------|
| player_id | int | | FPL player ID |
| season | str | | "2025-26" format |
| gw | int | 1-38 | Gameweek |
| team_id | int | | FPL team ID |
| team_code | str | | 3-letter code (ARS, MUN, etc) |
| predicted_points | float | [0, ∞) | Expected fantasy points |
| predicted_start_prob | float | [0, 1] | P(starts) |
| confidence | float | [0, 1] | Certainty of start prob (1-2\|P-0.5\|) |

### Gameweek-level DataFrame

Same as fixture-level, but:
- One row per (player_id, season, gw)
- predicted_points = sum of all fixtures in that GW
- predicted_start_prob, confidence = means across fixtures

## Confidence Metric

Confidence reflects certainty of the start probability:

```
confidence = 1 - 2 * |P(start) - 0.5|
```

- High when P(start) ≈ 0 or 1 (definite starter or non-starter)
- Low when P(start) ≈ 0.5 (uncertain)
- Range: [0, 1]

Use confidence to weight predictions or filter uncertain cases.

### Double-Gameweek Confidence Aggregation

For gameweek-level predictions, confidence is **not a measure of prediction variance**. When two fixtures are summed in a DGW:
- `predicted_start_prob` = mean of fixture start probs
- `predicted_points` = **sum** (not mean) of fixture predictions
- `confidence` = mean of fixture confidences

The mean confidence reflects average certainty of start probabilities, not uncertainty of the summed points. For risk assessment of the aggregated prediction, treat confidence as:
- **High confidence** (>0.8): Both fixtures likely starters or non-starters; summed prediction is stable
- **Low confidence** (0.2–0.5): Fixtures differ in start certainty; summed prediction more volatile
- Use `predicted_start_prob` directly if you need the actual probability of playing in a DGW (mean of fixtures)

## Model Performance

Evaluated on test set (2025-26):

**Classifier (Start Probability)**
- ROC-AUC: ~0.75
- Brier Score: ~0.15
- Well-calibrated for start probabilities

**Starter Regression**
- MAE: ~1.5 points
- RMSE: ~2.1 points
- Spearman: ~0.60 (per-GW ranking)

**Non-starter Regression**
- MAE: ~0.3 points
- RMSE: ~0.5 points
- Spearman: ~0.50 (per-GW ranking)

## Notes

- Models assume v4 features are properly computed and not missing
- NaN values in features are filled with 0
- External start probs should be in [0, 1]; no validation
- Predictions can be negative in rare cases (sum of negative E[pts|starts] and E[pts|¬starts])
- Performance varies by position and time in season

## See Also

- [train_model.py](train_model.py) – Model training and evaluation
- [build_features.py](build_features.py) – Feature engineering (required as input)

"""Naive baselines to compare against the LightGBM model, on the same
train/val/test split and feature file (no actual "training" needed —
both baselines are simple formulas over existing lagged features).

Naive 1: last-5-gameweek average of total_points (= total_points_roll5).

Naive 2: blend of recent form and season-long points-per-game, scaled by
how weak/strong the upcoming opponent is:
    (0.6 * total_points_roll5 + 0.4 * total_points_career_avg) * strength_indicator

strength_indicator is interpolated from opponent_team_strength (1-5):
    1 -> 1.15, 2 -> 1.08, 3 -> 1.00, 4 -> 0.92, 5 -> 0.85
(fractional strengths, from double-gameweek averaging, are interpolated.)
"""

import numpy as np

from scripts.models.v1.train_model import evaluate_predictions, load_splits, print_metrics

STRENGTH_LEVELS = [1, 2, 3, 4, 5]
STRENGTH_INDICATORS = [1.15, 1.08, 1.00, 0.92, 0.85]


def opponent_strength_indicator(opponent_team_strength):
    return np.interp(opponent_team_strength, STRENGTH_LEVELS, STRENGTH_INDICATORS)


def naive_1(test):
    return test["total_points_roll5"]


def naive_2(test):
    form_blend = 0.6 * test["total_points_roll5"] + 0.4 * test["total_points_career_avg"]
    indicator = opponent_strength_indicator(test["opponent_team_strength"])
    return form_blend * indicator


def main():
    _, _, test, _ = load_splits()
    test = test.copy()

    test["pred_naive1"] = naive_1(test).fillna(0)
    test["pred_naive2"] = naive_2(test).fillna(0)

    metrics_1, _ = evaluate_predictions(test, "pred_naive1")
    print_metrics("Naive 1 (last-5-gw avg)", metrics_1)

    metrics_2, _ = evaluate_predictions(test, "pred_naive2")
    print_metrics("Naive 2 (blended form x opponent strength)", metrics_2)


if __name__ == "__main__":
    main()

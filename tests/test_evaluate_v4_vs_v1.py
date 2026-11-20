"""Tests for v4 vs v1 evaluation."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.stats import spearmanr

from scripts.evaluate_v4_vs_v1 import (
    compute_metrics_per_gw,
    compute_signed_bias,
)


@pytest.fixture
def sample_data():
    """Create small sample data for testing."""
    n = 100
    gw = np.array([1, 2, 3] * (n // 3 + 1))[:n]
    actual = np.random.uniform(0, 20, n)
    
    # Create predictions with known bias
    pred = actual + 2.0  # Positive bias of +2
    
    return actual, pred, gw


def test_compute_metrics_per_gw_basic(sample_data):
    """Test metric computation."""
    actual, pred, gw = sample_data
    
    metrics = compute_metrics_per_gw(actual, pred, gw)
    
    # Check keys present
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "spearman_avg" in metrics
    assert "spearman_std" in metrics
    assert "num_gws" in metrics
    
    # Check MAE with known bias
    mae = metrics["mae"]
    expected_mae = np.mean(np.abs(actual - pred))
    assert np.isclose(mae, expected_mae, rtol=1e-4)
    
    # With bias of +2, MAE should be ~2
    assert np.isclose(mae, 2.0, rtol=0.2)


def test_compute_metrics_per_gw_rmse(sample_data):
    """Test RMSE calculation."""
    actual, pred, gw = sample_data
    
    metrics = compute_metrics_per_gw(actual, pred, gw)
    
    rmse = metrics["rmse"]
    expected_rmse = np.sqrt(np.mean((actual - pred) ** 2))
    assert np.isclose(rmse, expected_rmse, rtol=1e-4)


def test_compute_metrics_per_gw_spearman(sample_data):
    """Test Spearman correlation per gameweek."""
    actual, pred, gw = sample_data
    
    metrics = compute_metrics_per_gw(actual, pred, gw)
    
    # With constant bias, Spearman should be perfect (1.0)
    assert metrics["spearman_avg"] > 0.95
    assert metrics["num_gws"] >= 2


def test_compute_signed_bias_basic(sample_data):
    """Test signed bias computation."""
    actual, pred, gw = sample_data
    
    # pred = actual + 2, so mean signed error should be ~2
    bias = compute_signed_bias(actual, pred, gw)
    
    assert "mean_signed_error" in bias
    assert "per_gw" in bias
    
    # With constant +2 bias
    mse = bias["mean_signed_error"]
    assert np.isclose(mse, 2.0, rtol=0.1)


def test_compute_signed_bias_per_gw(sample_data):
    """Test per-gameweek signed bias."""
    actual, pred, gw = sample_data
    
    bias = compute_signed_bias(actual, pred, gw)
    
    per_gw = bias["per_gw"]
    assert len(per_gw) >= 2
    
    # All gw biases should be ~2
    for g, b in per_gw.items():
        assert isinstance(g, int)
        assert np.isclose(b, 2.0, rtol=0.2)


def test_metrics_perfect_prediction():
    """Test metrics when prediction is perfect."""
    actual = np.array([5, 10, 15, 20, 25])
    pred = actual.copy()
    gw = np.array([1, 1, 2, 2, 3])
    
    metrics = compute_metrics_per_gw(actual, pred, gw)
    
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert np.isclose(metrics["spearman_avg"], 1.0, rtol=1e-6)


def test_metrics_handling_nan():
    """Test that NaN is handled gracefully in Spearman."""
    actual = np.array([5.0, 10.0, 15.0, 20.0])
    pred = np.array([5.0, 10.0, 15.0, 20.0])
    gw = np.array([1, 1, 2, 2])
    
    # Should not crash
    metrics = compute_metrics_per_gw(actual, pred, gw)
    
    # Should have valid results
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0

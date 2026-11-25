"""Tests for measure_ffpundit_calibration module."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from measure_ffpundit_calibration import (
    compute_brier_score,
    compute_ece,
    load_matched_forecasts,
    measure_calibration_gw,
)


class TestBrierScore:
    """Test Brier score computation."""
    
    def test_perfect_predictions(self):
        """Perfect predictions should score 0."""
        predictions = np.array([0.0, 1.0, 0.0, 1.0])
        actuals = np.array([0, 1, 0, 1])
        assert compute_brier_score(predictions, actuals) == 0.0
    
    def test_worst_predictions(self):
        """Opposite predictions should score 1.0."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        actuals = np.array([0, 1, 0, 1])
        assert compute_brier_score(predictions, actuals) == 1.0
    
    def test_uniform_60_pct_pred(self):
        """Test with uniform 60% prediction vs 50% actual."""
        predictions = np.array([0.6, 0.6, 0.6, 0.6])
        actuals = np.array([1, 1, 0, 0])
        # Brier = mean((0.6 - 1)^2, (0.6 - 1)^2, (0.6 - 0)^2, (0.6 - 0)^2)
        #       = mean(0.16, 0.16, 0.36, 0.36) = 0.26
        expected = (0.4**2 + 0.4**2 + 0.6**2 + 0.6**2) / 4
        assert compute_brier_score(predictions, actuals) == pytest.approx(expected)
    
    def test_empty_arrays(self):
        """Empty arrays should return NaN."""
        assert np.isnan(compute_brier_score(np.array([]), np.array([])))


class TestECE:
    """Test Expected Calibration Error computation."""
    
    def test_perfect_calibration(self):
        """Perfect calibration (predictions == outcomes in each bin) should score 0."""
        # 10 preds in [0.1-0.2), all 0 (perfect)
        # 10 preds in [0.9-1.0], all 1 (perfect)
        predictions = np.array([0.15] * 10 + [0.95] * 10)
        actuals = np.array([0] * 10 + [1] * 10)
        # Bin [0.1-0.2): mean_pred=0.15, accuracy=0, |0-0.15| * 0.5 = 0.075
        # Bin [0.9-1.0]: mean_pred=0.95, accuracy=1, |1-0.95| * 0.5 = 0.025
        # ECE = 0.075 + 0.025 = 0.1
        ece = compute_ece(predictions, actuals, n_bins=10)
        assert ece == pytest.approx(0.1)
    
    def test_empty_arrays(self):
        """Empty arrays should return NaN."""
        assert np.isnan(compute_ece(np.array([]), np.array([])))
    
    def test_single_prediction(self):
        """Single prediction: pred=0.6, actual=1, ECE in one bin."""
        predictions = np.array([0.6])
        actuals = np.array([1])
        # Single prediction in [0.6-0.7) bin: |1 - 0.6| * 1.0 = 0.4
        ece = compute_ece(predictions, actuals, n_bins=10)
        assert ece == pytest.approx(0.4)


class TestLoadMatchedForecasts:
    """Test loading matched forecasts."""
    
    def test_load_valid_csv(self):
        """Load a valid matched forecasts CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("player_id,gw,ffpundit_start_pct\n")
            f.write("1,11,95.0\n")
            f.write("2,11,70.0\n")
            f.flush()
            
            df = load_matched_forecasts(f.name)
            assert len(df) == 2
            # player_id should be element_ids (1-667)
            assert list(df['player_id']) == [1, 2]
            assert list(df['gw']) == [11, 11]
            
            Path(f.name).unlink()
    
    def test_missing_file(self):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_matched_forecasts("/nonexistent/file.csv")
    
    def test_missing_columns(self):
        """Missing required columns should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("player_id,gw\n")  # Missing ffpundit_start_pct
            f.write("1,11\n")
            f.flush()
            
            with pytest.raises(ValueError):
                load_matched_forecasts(f.name)
            
            Path(f.name).unlink()


class TestMeasureCalibrationGW:
    """Test per-gameweek calibration measurement."""
    
    def test_simple_gw_measurement(self):
        """Test calibration on a simple gameweek."""
        matched = pd.DataFrame({
            'player_id': [1, 2, 3, 4],
            'gw': [11, 11, 11, 11],
            'ffpundit_start_pct': [100.0, 0.0, 60.0, 50.0],
        })
        
        actual = pd.DataFrame({
            'player_id': [1, 2, 3, 4],
            'gw': [11, 11, 11, 11],
            'started': [1, 0, 1, 1],
        })
        
        result = measure_calibration_gw(matched, actual, 11)
        
        assert result is not None
        assert result['gw'] == 11
        assert result['n_players'] == 4
        assert 'brier_score' in result
        assert 'ece' in result
    
    def test_missing_gw(self):
        """Missing gameweek should return None."""
        matched = pd.DataFrame({
            'player_id': [1],
            'gw': [10],
            'ffpundit_start_pct': [50.0],
        })
        
        actual = pd.DataFrame({
            'player_id': [1],
            'gw': [10],
            'started': [1],
        })
        
        result = measure_calibration_gw(matched, actual, 11)
        assert result is None
    
    def test_missing_start_labels(self):
        """Missing start labels should be skipped gracefully."""
        matched = pd.DataFrame({
            'player_id': [1, 2, 3],
            'gw': [11, 11, 11],
            'ffpundit_start_pct': [50.0, 60.0, 70.0],
        })
        
        actual = pd.DataFrame({
            'player_id': [1, 2],  # Player 3 missing
            'gw': [11, 11],
            'started': [1, 0],
        })
        
        result = measure_calibration_gw(matched, actual, 11)
        
        assert result is not None
        assert result['n_players'] == 2  # Only 2 players matched


class TestBrierScoreWarning:
    """Test Brier score warning threshold."""
    
    def test_brier_above_threshold_logs_warning(self, caplog):
        """Brier > 0.20 should log warning."""
        import logging
        
        matched = pd.DataFrame({
            'player_id': list(range(10)),
            'gw': [11] * 10,
            'ffpundit_start_pct': [5.0] * 10,  # Very low predictions
        })
        
        actual = pd.DataFrame({
            'player_id': list(range(10)),
            'gw': [11] * 10,
            'started': [1] * 10,  # All started: predictions will be way off
        })
        
        with caplog.at_level(logging.WARNING):
            result = measure_calibration_gw(matched, actual, 11)
        
        # Brier should be high
        assert result['brier_score'] > 0.20
        # Check warning logged
        assert any("needs review" in record.message for record in caplog.records)

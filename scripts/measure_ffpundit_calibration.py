"""Measure FFPundit forecast calibration: Brier score and ECE per gameweek."""

import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _build_element_to_code_map(players_csv: str) -> dict:
    """Build mapping from FPL element_id to player code (used in player_gameweeks).
    
    Args:
        players_csv: Path to players.csv (has id and code columns)
        
    Returns:
        Dict mapping element_id (players.id) to player code (players.code)
    """
    df = pd.read_csv(players_csv)
    return dict(zip(df['id'], df['code']))

if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_matched_forecasts(csv_path: str) -> pd.DataFrame:
    """Load matched FFPundit forecasts CSV.
    
    Args:
        csv_path: Path to matched forecasts CSV
        
    Returns:
        DataFrame with columns: player_id, gw, ffpundit_start_pct, etc.
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required columns missing
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required = {"player_id", "gw", "ffpundit_start_pct"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns. Required: {required}, Got: {set(df.columns)}")
    
    return df


def load_actual_starts(parquet_path: str, season: str, gameweeks: list[int]) -> pd.DataFrame:
    """Load actual start labels from player_gameweeks.parquet.
    
    Args:
        parquet_path: Path to player_gameweeks.parquet
        season: Season string (e.g., "2026-27")
        gameweeks: List of gameweek numbers to load
        
    Returns:
        DataFrame with columns: player_id, gw, started (1 if minutes > 0, else 0)
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    # Filter to season and gameweeks
    df = df[(df['season'] == season) & (df['gw'].isin(gameweeks))]
    
    # Create start label: 1 if minutes > 0, else 0
    df['started'] = (df['minutes'] > 0).astype(int)
    
    return df[['player_id', 'gw', 'started']].copy()


def compute_brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Compute Brier score: mean((P - actual)^2).
    
    Args:
        predictions: Array of predicted probabilities [0, 1]
        actuals: Array of actual outcomes {0, 1}
        
    Returns:
        Brier score (lower is better, range [0, 2])
    """
    if len(predictions) == 0:
        return np.nan
    return float(np.mean((predictions - actuals) ** 2))


def compute_ece(predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Bin predictions into [0-0.1), [0.1-0.2), ..., [0.9-1.0].
    For each bin, compute accuracy and weight by bin frequency.
    ECE = sum(|accuracy - mean_pred| * bin_freq)
    
    Args:
        predictions: Array of predicted probabilities [0, 1]
        actuals: Array of actual outcomes {0, 1}
        n_bins: Number of bins (default 10 for deciles)
        
    Returns:
        ECE (lower is better, range [0, 1])
    """
    if len(predictions) == 0:
        return np.nan
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # Include upper edge only in last bin to avoid double-counting
        mask = (predictions >= lower) & (predictions < upper) if i < n_bins - 1 else (predictions >= lower) & (predictions <= upper)
        
        if mask.sum() == 0:
            continue
        
        bin_preds = predictions[mask]
        bin_actuals = actuals[mask]
        
        mean_pred = bin_preds.mean()
        accuracy = bin_actuals.mean()
        bin_freq = mask.sum() / len(predictions)
        
        ece += abs(accuracy - mean_pred) * bin_freq
    
    return float(ece)


def measure_calibration_gw(
    matched_forecasts: pd.DataFrame,
    actual_starts: pd.DataFrame,
    gameweek: int,
) -> dict:
    """Measure calibration metrics for a single gameweek.
    
    Args:
        matched_forecasts: DataFrame from load_matched_forecasts()
        actual_starts: DataFrame from load_actual_starts()
        gameweek: Gameweek number
        
    Returns:
        Dict with keys: gw, brier_score, ece, n_players, avg_pred_start_pct, 
        min_pred_start_pct, max_pred_start_pct
    """
    # Filter forecasts to gameweek
    gw_forecasts = matched_forecasts[matched_forecasts['gw'] == gameweek].copy()
    
    if len(gw_forecasts) == 0:
        logger.warning(f"No forecasts for GW {gameweek}")
        return None
    
    # Filter actual starts to gameweek
    gw_starts = actual_starts[actual_starts['gw'] == gameweek].copy()
    
    # Merge on player_id
    merged = gw_forecasts.merge(gw_starts, on='player_id', how='left', suffixes=('', '_actual'))
    
    # Drop rows with missing start label
    if merged['started'].isna().any():
        n_missing = merged['started'].isna().sum()
        logger.warning(f"GW {gameweek}: {n_missing} players missing start label, skipping them")
        merged = merged.dropna(subset=['started'])
    
    if len(merged) == 0:
        logger.warning(f"GW {gameweek}: No matched players with start labels")
        return None
    
    # Convert predictions from percentage to probability [0, 1]
    predictions = merged['ffpundit_start_pct'].values / 100.0
    actuals = merged['started'].values
    
    # Compute metrics
    brier = compute_brier_score(predictions, actuals)
    ece = compute_ece(predictions, actuals)
    
    result = {
        'gw': gameweek,
        'brier_score': brier,
        'ece': ece,
        'n_players': len(merged),
        'avg_pred_start_pct': merged['ffpundit_start_pct'].mean(),
        'min_pred_start_pct': merged['ffpundit_start_pct'].min(),
        'max_pred_start_pct': merged['ffpundit_start_pct'].max(),
    }
    
    # Flag if Brier > 0.20 (< 80% accuracy equivalent)
    if brier > 0.20:
        logger.warning(f"GW {gameweek}: Brier score {brier:.4f} > 0.20 (< 80% accuracy) — needs review")
    
    return result


def measure_calibration_batch(
    matched_forecasts: pd.DataFrame,
    actual_starts: pd.DataFrame,
    gameweeks: list[int],
    output_csv: str,
) -> pd.DataFrame:
    """Measure calibration for a batch of gameweeks and save to CSV.
    
    Args:
        matched_forecasts: DataFrame from load_matched_forecasts()
        actual_starts: DataFrame from load_actual_starts()
        gameweeks: List of gameweek numbers
        output_csv: Path to output CSV
        
    Returns:
        DataFrame with per-GW metrics
    """
    results = []
    
    for gw in gameweeks:
        result = measure_calibration_gw(matched_forecasts, actual_starts, gw)
        if result is not None:
            results.append(result)
    
    if not results:
        logger.error("No results computed")
        return pd.DataFrame()
    
    output_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {len(output_df)} gameweeks to {output_csv}")
    
    return output_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure FFPundit forecast calibration (Brier score + ECE)")
    parser.add_argument(
        "--matched-forecasts",
        required=True,
        help="Path to matched FFPundit forecasts CSV",
    )
    parser.add_argument(
        "--actual-starts",
        required=True,
        help="Path to player_gameweeks.parquet (for actual start labels)",
    )
    parser.add_argument(
        "--players-csv",
        required=True,
        help="Path to players.csv (for element_id → code mapping)",
    )
    parser.add_argument(
        "--season",
        default="2026-27",
        help="Season string (default: 2026-27)",
    )
    parser.add_argument(
        "--gameweeks",
        required=True,
        help="Gameweek range: either single (e.g., '11') or range (e.g., '11-14')",
    )
    parser.add_argument(
        "--output-csv",
        default="data/validation/ffpundit_calibration.csv",
        help="Output CSV path (default: data/validation/ffpundit_calibration.csv)",
    )
    
    args = parser.parse_args()
    
    # Parse gameweeks
    if "-" in args.gameweeks:
        start, end = map(int, args.gameweeks.split("-"))
        gameweeks = list(range(start, end + 1))
    else:
        gameweeks = [int(args.gameweeks)]
    
    logger.info(f"Loading matched forecasts from {args.matched_forecasts}")
    matched = load_matched_forecasts(args.matched_forecasts)
    logger.info(f"Loaded {len(matched)} forecasts")
    
    # Map element_id (from matcher) to code (used in player_gameweeks)
    logger.info(f"Building element_id → code mapping from {args.players_csv}")
    element_to_code = _build_element_to_code_map(args.players_csv)
    matched['player_id'] = matched['player_id'].map(element_to_code)
    matched = matched.dropna(subset=['player_id'])
    matched['player_id'] = matched['player_id'].astype(int)
    logger.info(f"Mapped {len(matched)} forecasts to player codes")
    
    logger.info(f"Loading actual starts from {args.actual_starts} (season {args.season})")
    actual = load_actual_starts(args.actual_starts, args.season, gameweeks)
    logger.info(f"Loaded {len(actual)} player-GW records with start labels")
    
    logger.info(f"Computing calibration for GW {gameweeks}")
    output_df = measure_calibration_batch(matched, actual, gameweeks, args.output_csv)
    
    logger.info(f"Results:\n{output_df.to_string(index=False)}")

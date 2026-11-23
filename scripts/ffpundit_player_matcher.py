"""Match FFPundit player names to FPL player IDs using fuzzy/exact matching."""

import logging
import re
import sys
from pathlib import Path
from typing import Optional
import pandas as pd


logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_ffpundit_forecasts(csv_path: str) -> pd.DataFrame:
    """Load FFPundit forecasts CSV.
    
    Args:
        csv_path: Path to gw{gw}_forecasts.csv
        
    Returns:
        DataFrame with columns: team, player_name, starting_pct, scraped_at
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required columns missing
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required = {"team", "player_name", "starting_pct"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns. Required: {required}, Got: {set(df.columns)}")
    
    return df


def load_fpl_players(csv_path: str) -> pd.DataFrame:
    """Load FPL players CSV.
    
    Args:
        csv_path: Path to players.csv
        
    Returns:
        DataFrame with columns: id, web_name, team_code, etc.
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required columns missing
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required = {"id", "web_name", "team_code"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns. Required: {required}, Got: {set(df.columns)}")
    
    return df


def _normalize_name(name: str) -> str:
    """Normalize player name: strip, lowercase, collapse whitespace."""
    return " ".join(name.strip().lower().split())


def _tokenize_name(name: str) -> set:
    """Tokenize name: normalize, remove periods/hyphens, split on whitespace."""
    normalized = name.lower().strip()
    normalized = re.sub(r'[.\-_]', ' ', normalized)
    return set(normalized.split())


def match_player_names(
    ffpundit_df: pd.DataFrame,
    fpl_df: pd.DataFrame,
    teams_df: Optional[pd.DataFrame] = None,
    fuzzy_threshold: float = 0.50,
) -> pd.DataFrame:
    """Match FFPundit player names to FPL player IDs.
    
    Strategy:
    1. Try exact match (case-insensitive, whitespace-normalized)
    2. Fall back to fuzzy match (token-based, threshold 0.50)
    3. If no match, log warning and return row with player_id=NaN, match_type="skipped"
    
    Args:
        ffpundit_df: DataFrame with columns: team, player_name, starting_pct
        fpl_df: DataFrame with columns: id, web_name, team_code, ...
        teams_df: DataFrame with columns: id, name (unused, kept for API compatibility)
        fuzzy_threshold: Minimum similarity ratio for fuzzy match (0.50)
        
    Returns:
        DataFrame with columns: player_id, player_name, starting_pct, 
        matched_fpl_name, match_type (exact|fuzzy|skipped), team
    """
    results = []
    
    for _, row in ffpundit_df.iterrows():
        ffpundit_name = row["player_name"]
        team_name = row["team"]
        start_pct = row["starting_pct"]
        
        normalized_ffpundit = _normalize_name(ffpundit_name)
        
        # Try exact match (full string)
        exact_matches = fpl_df[
            fpl_df["web_name"].apply(lambda x: _normalize_name(x) == normalized_ffpundit)
        ]
        
        if len(exact_matches) > 0:
            matched_row = exact_matches.iloc[0]
            results.append({
                "player_id": matched_row["id"],
                "player_name": ffpundit_name,
                "starting_pct": start_pct,
                "matched_fpl_name": matched_row["web_name"],
                "match_type": "exact",
                "team": team_name,
            })
            continue
        
        # Try fuzzy match: find best token overlap
        # Score bidirectionally: how many tokens match from either side
        ffpundit_tokens = _tokenize_name(normalized_ffpundit)
        best_score = -1
        best_match = None
        
        for _, candidate in fpl_df.iterrows():
            candidate_tokens = _tokenize_name(candidate["web_name"])
            if not ffpundit_tokens or not candidate_tokens:
                continue
            
            # Score: common tokens / min(ffpundit_tokens, candidate_tokens)
            # This handles both "silva" vs "silva joão silva" and vice versa
            common = len(ffpundit_tokens & candidate_tokens)
            denom = min(len(ffpundit_tokens), len(candidate_tokens))
            score = common / denom if denom > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_score >= fuzzy_threshold:
            results.append({
                "player_id": best_match["id"],
                "player_name": ffpundit_name,
                "starting_pct": start_pct,
                "matched_fpl_name": best_match["web_name"],
                "match_type": "fuzzy",
                "team": team_name,
            })
            continue
        
        # No match: skip
        logger.warning(f"No match for {team_name} / {ffpundit_name}")
        results.append({
            "player_id": None,
            "player_name": ffpundit_name,
            "starting_pct": start_pct,
            "matched_fpl_name": None,
            "match_type": "skipped",
            "team": team_name,
        })
    
    return pd.DataFrame(results)


def build_output_dataframe(
    matches: pd.DataFrame,
    season: str,
    gameweek: int,
) -> pd.DataFrame:
    """Build final output DataFrame with schema.
    
    Args:
        matches: Output from match_player_names()
        season: Season string (e.g., "2026-27")
        gameweek: Gameweek number (1-38)
        
    Returns:
        DataFrame with columns:
        - player_id: FPL player ID (or NaN if skipped)
        - season: Season string
        - gw: Gameweek number
        - ffpundit_start_pct: Starting percentage from FFPundit
        - matched_fpl_name: FPL web_name (or NaN if skipped)
        - match_type: "exact", "fuzzy", or "skipped"
    """
    result = pd.DataFrame({
        "player_id": matches["player_id"],
        "season": season,
        "gw": gameweek,
        "ffpundit_start_pct": matches["starting_pct"],
        "matched_fpl_name": matches["matched_fpl_name"],
        "match_type": matches["match_type"],
    })
    
    return result


def main(
    ffpundit_csv: str,
    fpl_players_csv: str,
    season: str,
    gameweek: int,
    teams_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Orchestrate: load, match, output.
    
    Args:
        ffpundit_csv: Path to FFPundit forecasts CSV
        fpl_players_csv: Path to FPL players.csv
        season: Season string (e.g., "2026-27")
        gameweek: Gameweek number (1-38)
        teams_csv: Path to teams.csv (optional, for team name mapping)
        
    Returns:
        Matched output DataFrame, ready for inference
    """
    ffpundit_df = load_ffpundit_forecasts(ffpundit_csv)
    fpl_df = load_fpl_players(fpl_players_csv)
    
    teams_df = None
    if teams_csv:
        teams_df = pd.read_csv(teams_csv)
    
    matches = match_player_names(ffpundit_df, fpl_df, teams_df)
    output = build_output_dataframe(matches, season, gameweek)
    
    return output

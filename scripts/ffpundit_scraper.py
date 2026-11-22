"""FFPundit web scraper for player Starting % forecasts."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)

# Only configure if not already configured (avoid duplicate handlers in tests)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def parse_starting_percentage(text: str) -> Optional[float]:
    """Parse starting percentage from text like '65%' or '42'."""
    if not text:
        return None
    
    text = text.strip().rstrip('%')
    try:
        return float(text)
    except ValueError:
        return None


def get_current_gameweek() -> int:
    """Get current Premier League gameweek (1-38).
    
    For now, returns a placeholder. In production, would call FPL API
    or parse from a reliable source.
    """
    # TODO: Implement actual gameweek detection from FPL API
    # Placeholder: return 1
    return 1


class FFPunditScraper:
    """Scrapes player Starting % from FFPundit team news pages."""

    BASE_URL = "https://www.fantasyfootballpundit.com/fantasy-premier-league-team-news/"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    def __init__(self, cache_dir: Path = None):
        """Initialize scraper with optional cache directory."""
        if cache_dir is None:
            cache_dir = Path("data/raw/ffpundit")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, gw: int) -> Path:
        """Get cache file path for gameweek."""
        return self.cache_dir / f"gw{gw}_forecasts.csv"

    def _fetch_page(self) -> Optional[str]:
        """Fetch FFPundit team news page. Return HTML or None on error."""
        try:
            response = requests.get(
                self.BASE_URL,
                headers={"User-Agent": self.USER_AGENT},
                timeout=10
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"HTTP error fetching {self.BASE_URL}: {e}")
            return None

    def _parse_html(self, html: str) -> pd.DataFrame:
        """Parse FFPundit HTML and extract player Starting % data."""
        if not html:
            return pd.DataFrame()

        soup = BeautifulSoup(html, 'html.parser')
        rows = []
        now = datetime.now()

        # Look for team headings (e.g., "Arsenal Predicted Lineup")
        team_headings = soup.find_all('h2')

        for h2 in team_headings:
            heading_text = h2.get_text(strip=True)
            
            # Extract team name from heading like "Arsenal Predicted Lineup"
            if 'Predicted Lineup' not in heading_text:
                continue
            
            team_name = heading_text.replace('Predicted Lineup', '').strip()
            
            # Find all tables for this team (Starting 11 + Potential Starters)
            # We need to get the next 2 tables but stop if we hit another h2
            current = h2
            tables_for_team = []
            
            for _ in range(2):  # Up to 2 tables per team
                table = current.find_next('table')
                if not table:
                    break
                
                # Check if we've crossed into next team (found another h2)
                next_h2 = table.find_next('h2')
                prev_h2 = table.find_previous('h2')
                if next_h2 and next_h2 != h2 and prev_h2 == h2:
                    # This table belongs to this team
                    tables_for_team.append(table)
                elif not next_h2:
                    # Last team, include this table
                    tables_for_team.append(table)
                else:
                    break
                
                current = table

            # Parse all tables for this team
            for table in tables_for_team:
                table_rows = table.find_all('tr')
                for row in table_rows[1:]:  # Skip header row
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 4:
                        continue

                    # Columns: Player, Pos, GW10 Mins, Start %
                    player_cell = cells[0]
                    
                    # Try to extract from span with nameFull class (clean name)
                    name_span = player_cell.find('span', class_=lambda x: x and 'nameFull' in x if x else False)
                    if name_span:
                        player_name = name_span.get_text(strip=True)
                    else:
                        # Fallback: use first text before any special characters
                        player_text = player_cell.get_text(strip=True)
                        # Remove status markers
                        for suffix in ['Doubt', 'Injury', 'Suspended']:
                            if suffix in player_text:
                                player_text = player_text.replace(suffix, '').strip()
                        player_name = player_text.split('\n')[0].strip()
                    
                    # Start % is in the 4th column (index 3)
                    pct_text = cells[3].get_text(strip=True)
                    pct = parse_starting_percentage(pct_text)

                    if player_name and pct is not None:
                        rows.append({
                            "team": team_name,
                            "player_name": player_name,
                            "starting_pct": pct,
                            "scraped_at": now
                        })

        df = pd.DataFrame(rows)
        
        if not df.empty:
            # Ensure all 20 PL teams represented (or log if incomplete)
            unique_teams = df["team"].nunique()
            if unique_teams < 20:
                logger.warning(f"Incomplete parse: found {unique_teams} teams, expected 20")

        return df

    def _write_cache(self, df: pd.DataFrame, gw: int) -> None:
        """Write dataframe to cache CSV."""
        if df.empty:
            logger.warning(f"Not caching empty dataframe for GW {gw}")
            return

        cache_path = self._get_cache_path(gw)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached {len(df)} rows to {cache_path}")

    def _cleanup_old_caches(self) -> None:
        """Keep only the latest 10 gameweek caches."""
        cache_files = []
        for f in self.cache_dir.glob("gw*_forecasts.csv"):
            try:
                # Extract GW number from filename like "gw5_forecasts.csv"
                name_parts = f.stem.split('_')  # Remove .csv
                gw_str = name_parts[0].replace('gw', '')
                gw_num = int(gw_str)
                cache_files.append((gw_num, f))
            except (ValueError, IndexError):
                continue
        
        cache_files.sort()
        
        if len(cache_files) > 10:
            files_to_remove = cache_files[:-10]
            for _, cache_file in files_to_remove:
                cache_file.unlink()
                logger.info(f"Removed old cache: {cache_file}")

    def scrape(self, gw: Optional[int] = None) -> pd.DataFrame:
        """Scrape FFPundit for player Starting % data.
        
        Args:
            gw: Gameweek to scrape (1-38). If None, uses current gameweek.
        
        Returns:
            DataFrame with columns: team, player_name, starting_pct, scraped_at
            Returns empty DataFrame on error (never raises exception).
        """
        if gw is None:
            gw = get_current_gameweek()

        # Check cache first
        cache_path = self._get_cache_path(gw)
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                logger.info(f"Loaded {len(df)} rows from cache: {cache_path}")
                return df
            except Exception as e:
                logger.error(f"Error reading cache {cache_path}: {e}")

        # Fetch and parse
        try:
            html = self._fetch_page()
            if html is None:
                return pd.DataFrame()

            df = self._parse_html(html)
            
            if not df.empty:
                self._write_cache(df, gw)
                self._cleanup_old_caches()
            
            return df

        except Exception as e:
            logger.error(f"Unexpected error during scrape: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape FFPundit Starting % forecasts.")
    parser.add_argument(
        "--gw",
        type=int,
        help="Gameweek to scrape (1-38). Auto-detects current if not provided."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/ffpundit"),
        help="Cache directory for CSV files."
    )
    
    args = parser.parse_args()
    
    scraper = FFPunditScraper(cache_dir=args.cache_dir)
    df = scraper.scrape(gw=args.gw)
    
    if not df.empty:
        print(f"Scraped {len(df)} rows")
        print(df.head())
    else:
        print("No data scraped")

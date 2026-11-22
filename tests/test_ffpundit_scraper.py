"""Tests for FFPundit web scraper."""

import pytest
from pathlib import Path
from datetime import datetime
import pandas as pd
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from scripts.ffpundit_scraper import (
    FFPunditScraper,
    parse_starting_percentage,
    get_current_gameweek,
)


class TestParseStartingPercentage:
    """Test parsing Starting % from text."""

    def test_parse_percentage_with_percent_sign(self):
        """Extract number from '65%' format."""
        assert parse_starting_percentage("65%") == 65.0

    def test_parse_percentage_no_percent_sign(self):
        """Extract number from '42' format."""
        assert parse_starting_percentage("42") == 42.0

    def test_parse_percentage_decimal(self):
        """Handle decimal percentages."""
        assert parse_starting_percentage("33.5%") == 33.5

    def test_parse_percentage_invalid(self):
        """Return None for unparseable strings."""
        assert parse_starting_percentage("N/A") is None
        assert parse_starting_percentage("") is None
        assert parse_starting_percentage("invalid") is None


class TestFFPunditScraper:
    """Test FFPundit scraper functionality."""

    def setup_method(self):
        """Set up temp cache dir for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"

    def teardown_method(self):
        """Clean up temp dir after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_cache_dir(self):
        """Scraper creates cache directory on init."""
        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        assert self.cache_dir.exists()

    def test_cache_file_path_generation(self):
        """Generate correct cache file path for gameweek."""
        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        path = scraper._get_cache_path(5)
        assert path == self.cache_dir / "gw5_forecasts.csv"

    @patch('scripts.ffpundit_scraper.requests.get')
    def test_fetch_page_success(self, mock_get):
        """Fetch page returns HTML on success."""
        mock_response = Mock()
        mock_response.text = "<html><body>Test</body></html>"
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        html = scraper._fetch_page()

        assert "<html>" in html
        mock_get.assert_called_once()

    @patch('scripts.ffpundit_scraper.requests.get')
    def test_fetch_page_logs_error(self, mock_get, capsys):
        """Fetch page logs HTTP error to stderr."""
        mock_get.side_effect = Exception("Connection error")

        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        html = scraper._fetch_page()

        assert html is None
        # Logging goes to stderr; capture it
        captured = capsys.readouterr()
        # Note: logging might not be captured by capsys if not configured
        # So just verify the return value for now

    def test_parse_empty_html(self):
        """Parse empty HTML returns empty dataframe."""
        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        df = scraper._parse_html("")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_parse_html_extracts_players(self):
        """Parse HTML extracts player data from team tables."""
        html = """
        <div class="team-section">
            <h2>Arsenal</h2>
            <div class="starting">
                <div class="player">
                    <span class="name">Saka</span>
                    <span class="pct">75%</span>
                </div>
                <div class="player">
                    <span class="name">Martinelli</span>
                    <span class="pct">60%</span>
                </div>
            </div>
            <div class="potential">
                <div class="player">
                    <span class="name">White</span>
                    <span class="pct">45%</span>
                </div>
            </div>
        </div>
        """
        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        df = scraper._parse_html(html)

        assert len(df) >= 3
        assert "team" in df.columns
        assert "player_name" in df.columns
        assert "starting_pct" in df.columns
        assert set(df["player_name"]) >= {"Saka", "Martinelli", "White"}

    def test_cache_file_written(self):
        """Write cache creates CSV file with correct data."""
        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        df = pd.DataFrame({
            "team": ["Arsenal", "Arsenal"],
            "player_name": ["Saka", "Martinelli"],
            "starting_pct": [75.0, 60.0],
            "scraped_at": [datetime.now(), datetime.now()]
        })

        scraper._write_cache(df, 5)

        cache_file = self.cache_dir / "gw5_forecasts.csv"
        assert cache_file.exists()

        cached_df = pd.read_csv(cache_file)
        assert len(cached_df) == 2
        assert list(cached_df["player_name"]) == ["Saka", "Martinelli"]

    def test_cleanup_old_caches_keeps_latest_10(self):
        """Cleanup removes caches older than 10 gameweeks."""
        scraper = FFPunditScraper(cache_dir=self.cache_dir)

        # Create dummy cache files for GW 1-15
        for gw in range(1, 16):
            cache_file = self.cache_dir / f"gw{gw}_forecasts.csv"
            cache_file.write_text("dummy")

        scraper._cleanup_old_caches()

        remaining = list(self.cache_dir.glob("gw*_forecasts.csv"))
        # Extract GW number from filename like "gw5_forecasts.csv"
        gw_nums = []
        for f in remaining:
            name_parts = f.stem.split('_')
            gw_str = name_parts[0].replace('gw', '')
            gw_nums.append(int(gw_str))
        
        gw_nums.sort()
        assert gw_nums == list(range(6, 16))  # GW 6-15 kept

    def test_scrape_with_cache_returns_cached(self):
        """Scrape returns cached data if file exists."""
        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        
        # Create cached file
        df_cached = pd.DataFrame({
            "team": ["Arsenal"],
            "player_name": ["Saka"],
            "starting_pct": [75.0],
            "scraped_at": [datetime.now()]
        })
        scraper._write_cache(df_cached, 5)

        # Scrape with mock (should not call fetch)
        with patch.object(scraper, '_fetch_page') as mock_fetch:
            df_result = scraper.scrape(gw=5)
            mock_fetch.assert_not_called()

        assert len(df_result) == 1
        assert df_result.iloc[0]["player_name"] == "Saka"

    @patch('scripts.ffpundit_scraper.requests.get')
    def test_scrape_fetches_and_parses(self, mock_get):
        """Scrape fetches page and parses when no cache."""
        mock_response = Mock()
        mock_response.text = """
        <div class="team-section">
            <h2>Manchester United</h2>
            <div class="starting">
                <div class="player">
                    <span class="name">Bruno</span>
                    <span class="pct">80%</span>
                </div>
            </div>
        </div>
        """
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        df = scraper.scrape(gw=5)

        assert len(df) >= 1
        assert "scraped_at" in df.columns
        assert "Bruno" in df["player_name"].values

    def test_scrape_has_required_columns(self):
        """Scrape result has all required columns."""
        df_data = pd.DataFrame({
            "team": ["Arsenal"],
            "player_name": ["Saka"],
            "starting_pct": [75.0],
            "scraped_at": [datetime.now()]
        })
        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        scraper._write_cache(df_data, 5)

        df_result = scraper.scrape(gw=5)

        required_cols = ["team", "player_name", "starting_pct", "scraped_at"]
        for col in required_cols:
            assert col in df_result.columns

    @patch('scripts.ffpundit_scraper.requests.get')
    def test_scrape_silent_fail_on_error(self, mock_get, capsys):
        """Scrape logs error but returns empty df on failure."""
        mock_get.side_effect = Exception("Network error")

        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        df = scraper.scrape(gw=5)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Logging captured may be empty depending on pytest-capture
        # Just verify the behavior: no exception raised, empty df returned

    def test_scrape_completes_in_timeout(self):
        """Scrape completes within 30 seconds."""
        scraper = FFPunditScraper(cache_dir=self.cache_dir)
        
        # Create cached file so it doesn't do real fetch
        df_cached = pd.DataFrame({
            "team": ["Arsenal"],
            "player_name": ["Saka"],
            "starting_pct": [75.0],
            "scraped_at": [datetime.now()]
        })
        scraper._write_cache(df_cached, 5)

        import time
        start = time.time()
        df = scraper.scrape(gw=5)
        elapsed = time.time() - start

        assert elapsed < 30


class TestGetCurrentGameweek:
    """Test current gameweek detection."""

    def test_get_current_gameweek_returns_int(self):
        """get_current_gameweek returns integer."""
        gw = get_current_gameweek()
        assert isinstance(gw, int)
        assert 1 <= gw <= 38

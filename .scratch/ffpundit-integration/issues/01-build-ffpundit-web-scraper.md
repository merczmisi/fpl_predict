# 01: Build FFPundit web scraper

**What to build:** Script extracts player-level Starting % from FFPundit team tables (fantasyfootballpundit.com/fantasy-premier-league-team-news/), caches locally, fails gracefully with logging.

**Blocked by:** None (can start immediately)

**Status:** done

- [X] Scraper fetches FFPundit team news page, extracts all 20 PL team tables
- [X] Scraper parses "Starting 11" and "Potential Starters" sections, capturing player name and Starting % for each
- [X] Scraper outputs DataFrame (team, player_name, starting_pct, scraped_at) to local cache `data/raw/ffpundit/gw{gw}_forecasts.csv`
- [X] Scraper caches latest 10 gameweeks locally, discards older caches
- [X] Scraper logs HTTP errors, parse errors, incomplete data to stderr; silently skips on failure (never raises exception)
- [X] Scraper can be called with target gameweek argument; auto-detects current GW if not provided
- [X] Scraper can be run 1 day before kickoff and completes within 30 seconds

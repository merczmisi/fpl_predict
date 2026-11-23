# FFPundit External Lineup Forecast Integration Spec

## Problem Statement

V4 model predicts E[points] using P(start) from a trained classifier, achieving break-even accuracy vs. V1 (MAE ~1.056). The classifier was trained on 2022-25 lineup data, with no external lineup forecasts to override it.

FFPundit (fantasyfootballpundit.com) publishes team-level Starting % predictions daily, one day before kickoff, with claimed 85-92% accuracy. These forecasts can replace the classifier output and potentially unlock 15-20% improvement in point calibration, but only if:

1. Scraper reliably extracts player-level Starting % from web tables
2. External probabilities are matched to FPL player IDs (player name → FPL API bridge)
3. Calibration is measured against actual lineups (Brier score + ECE)
4. V4 inference safely falls back to classifier if ffpundit unavailable

## Solution

Build a web scraper for FFPundit, integrate it into the v4 inference pipeline, and validate calibration on live 2026-27 data (GW11+) in shadow mode (parallel execution, hidden from users). After measuring Brier score + ECE on 2-3 gameweeks, present results for human sign-off before making predictions visible to users.

### Integration Points

1. **Scraper** (`scripts/data/fetch/scrape_ffpundit_lineup.py`): Daily 1 day pre-kickoff, extract player-level Starting %, cache locally
2. **Player matching** (`scripts/data/process/match_ffpundit_to_fpl.py`): Map FFPundit names to FPL player IDs using FPL API as bridge + fuzzy match fallback
3. **Inference enhancement** (`scripts/models/v4/inference.py`): Accept optional external_start_probs DataFrame, use in place of classifier output
4. **Calibration measurement** (`scripts/validation/measure_ffpundit_calibration.py`): Compute Brier score + ECE per gameweek against actual lineups
5. **Prediction output** (`scripts/prediction/predict_next_gw.py`): Output separate v4-ffpundit column alongside v4-classifier for A/B tracking

## User Stories

1. As a data engineer, I want to scrape FFPundit team tables one day before kickoff, so that I capture pre-deadline lineup forecasts at stable accuracy.

2. As a data engineer, I want to match FFPundit player names to FPL player IDs using FPL API as the bridge, so that I correctly join forecasts to player models.

3. As a data engineer, I want to implement fuzzy name matching for FFPundit players with no exact FPL API hit, so that I handle spelling/accent variations gracefully.

4. As a data engineer, I want to cache FFPundit scrapes locally (latest 10 gameweeks), so that I can re-validate calibration without re-scraping.

5. As a data engineer, I want to alert if the scrape fails or returns incomplete data for a gameweek, so that I know when ffpundit is unavailable and fallback is active.

6. As a model user, I want the v4 inference pipeline to accept an optional external_start_probs DataFrame, so that I can test ffpundit without modifying core model code.

7. As a model user, I want the pipeline to fall back to the trained classifier if external_start_probs is None or missing for a player, so that I never output NaN predictions.

8. As a model developer, I want to measure Brier score (forecast accuracy) per gameweek, so that I validate FFPundit's claimed 85-92% accuracy.

9. As a model developer, I want to measure Expected Calibration Error (ECE) per gameweek, so that I ensure forecast probabilities match actual starting rates (e.g., 50% predicted = ~50% actual).

10. As a model developer, I want to run v4+ffpundit in shadow mode (parallel to production), so that I collect metrics without affecting user-visible predictions.

11. As a model developer, I want to output v4-ffpundit predictions in a separate column alongside v4-classifier predictions, so that I can audit the difference per prediction.

12. As a model developer, I want to present calibration metrics (Brier score, ECE, MAE vs V1) after 2-3 gameweeks, so that I can seek human sign-off before full rollout.

13. As a data user, I want to understand which predictions used ffpundit vs. classifier, so that I can build trust in the external forecast.

## Implementation Decisions

### Web Scraper

- **Source**: fantasyfootballpundit.com/fantasy-premier-league-team-news/ (20 team tables, updated daily)
- **Timing**: Run 1 day before each weekend/Tuesday/Wednesday/Friday matchday (pre-kickoff but after lineup confirmations)
- **Data extracted**: Team name, player name, Starting % (0-100 for each player in "Starting 11" and "Potential Starters" sections)
- **Storage format**: Local CSV cache `data/raw/ffpundit/gw{gw}_forecasts.csv` with columns (team, player_name, starting_pct, scraped_at)
- **Error handling**: Log failures to stderr, skip on HTTP errors, do not raise exception (infer fallback to classifier)

### Player Name Matching

- **Primary**: Exact match via FPL API player name lookup (case-insensitive, whitespace-normalized)
- **Secondary**: Fuzzy match (difflib.SequenceMatcher, threshold 0.85) if exact miss
- **Fallback**: Skip player if fuzzy also fails; that player gets classifier output
- **Bridge data**: Query FPL API once per season to build a local player name → player_id cache
- **Cache location**: `data/processed/{season}/fpl_players_cache.csv` (player_id, player_name, team, position)

### Inference Enhancement

- **Function signature**: `predict_fixture_level(fixture_data, external_start_probs=None)` (unchanged)
- **New behavior**: If external_start_probs provided and (player_id, season, gw) tuple exists, use external P(start) instead of classifier.predict_proba()
- **Missing data**: If external_start_probs is None or tuple missing, fall back to classifier
- **Formula unchanged**: E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start]

### Calibration Measurement

- **Actual start labels**: Load from fixture data (1 = started, 0 = benched) for GW played
- **Brier score**: mean((P(start)_pred - actual_start)^2) per gameweek
- **ECE (Expected Calibration Error)**: Bin predictions [0-0.1), [0.1-0.2), ... [0.9-1.0), compute accuracy per bin, weight by bin size
- **Output**: Metrics CSV `data/validation/ffpundit_calibration.csv` (gw, brier_score, ece, n_players, avg_pred_start_pct)
- **Threshold for concern**: Brier score > 0.20 (40%+ miscalibration, manual review advised)

### Shadow Mode Rollout

- **Timing**: Scrape + match + inference runs in parallel with v4-classifier on every GW11+ prediction
- **Output**: Separate v4-ffpundit column in predictions CSV, identical format to v4-classifier
- **User visibility**: Off by default; only published to users post-sign-off
- **Tracking**: Log (player_id, gw, classifier_p_start, ffpundit_p_start, diff) for audit
- **Sign-off gate**: Require manual review of Brier score + ECE after 2-3 gameweeks before toggling v4-ffpundit column to user output

### Fallback & Error Handling

- **Scraper fails**: Log warning, skip ffpundit input for that GW, infer from classifier
- **Player name match fails**: Output classifier output for that player (no NaN)
- **Inference fails**: Raise exception only if core formula breaks (missing feature, NaN in E[pts|start]); external_start_probs errors degrade gracefully

## Testing Decisions

**What makes a good test:**
- Tests external behavior: Does scraper extract player names + Starting % correctly? Does matching map to FPL IDs? Does inference formula compute E[pts] correctly?
- Tests fallback behavior: If ffpundit unavailable, does classifier output appear? If player name match fails, does classifier output appear?
- Tests calibration: Are Brier score and ECE computed correctly against known test lineups?
- Tests do not inspect implementation details: Test "scrape returns valid DataFrame" not "BeautifulSoup parses <div> tag"; test "inference returns non-NaN points" not "line 42 multiplies correctly"

**Modules to test:**
1. **Scraper** (`scripts/data/fetch/scrape_ffpundit_lineup.py`): Does it return DataFrame with expected columns? Does it handle network errors gracefully?
2. **Matcher** (`scripts/data/process/match_ffpundit_to_fpl.py`): Does it map FFPundit names to FPL IDs correctly? Does fuzzy match handle typos? Does it skip on failure?
3. **Inference** (`scripts/models/v4/inference.py`): Does formula E[pts] = P(start) × E[pts|start] + (1−P(start)) × E[pts|¬start] hold? Does fallback to classifier work?
4. **Calibration measurement** (`scripts/validation/measure_ffpundit_calibration.py`): Does Brier score match manual calculation? Does ECE binning work?

**Prior art:**
- Existing test `test_v4_inference_pipeline.py`: Tests formula, feature handling, model loading
- Existing test `test_enrich_starts.py`: Tests labeling logic, NaN handling

## Out of Scope

- Retraining v4 classifier on 2026-27 data (Phase 2 future work)
- Building a tactical/rotation prediction model (Phase 2)
- Ensemble of multiple external sources (Phase 3)
- Integration with FPL official team news API (Phase 2)
- Modifying v4 feature engineering based on ffpundit insights
- Changing the v4 architecture (three-model decomposition stays)

## Further Notes

**Data freshness window:** FFPundit updates at deadline (11:30 UTC Fri, Tue, Wed, etc.). Scraping 1 day pre-kickoff captures lineups ~24h before fixtures. This is later than FPL official news (Phase 2) but earlier than post-match data, making it ideal for pre-deadline ML predictions.

**Accuracy validation**: Cannot validate ffpundit's claimed 85-92% historically (no past predictions available). Only live GW11+ validation possible. If live Brier score falls below 0.18 (82%+ accuracy), proceed; if above 0.20 (80%- accuracy), pause and investigate.

**Production risk:** External scrape is a single point of failure. Keep trained classifier always loaded as fallback. If scraper breaks, silence it (log, skip) rather than crashing v4 inference.

**A/B success criteria**: 
- Conservative win: MAE within 2% of v1 (target 1.078 ≤ MAE ≤ 1.075)
- Strong win: MAE 5% better than v1 (target MAE ≤ 1.003)
- If Brier score > 0.20 or ECE > 0.15 on first 2 GW, pause and audit before continuing.

# Fields

## Fields in parquet after process

| Field | Description | Type | Example value |
|---|---|---|---|
| `season` | Season identifier | string | `2025-26` |
| `gw` | Gameweek number | integer | `16` |
| `fixture_id` | Unique fixture identifier | integer | `1234` |
| `player_id` | Unique player identifier | integer | `456` |
| `position` | Player's position | string | `MID` |
| `team_id` | Team identifier | integer | `10` |
| `team_code` | Team code | string | `ARS` |
| `team_strength` | Team strength rating (aka FDR) | integer | `4` |
| `opponent_id` | Opponent team identifier | integer | `20` |
| `opponent_team_code` | Opponent team name | string | `CHE` |
| `opponent_team_strength` | Opponent team strength rating (aka FDR) | integer | `3` |
| `opponent_team_promoted` | Whether opponent was promoted this season | boolean | `true` |
| `is_home` | Whether the player's team was playing at home | boolean | `true` |
| `team_goals` | Goals scored by player's team | integer | `2` |
| `opponent_goals` | Goals scored by opponent | integer | `1` |
| `minutes` | Minutes played | integer | `90` |
| `goals_scored` | Goals scored by the player | integer | `1` |
| `goals_conceded` | Goals conceded while playing | integer | `1` |
| `xG` | Expected goals | float | `0.72` |
| `xGC` | Expected goals conceded | float | `1.35` |
| `xA` | Expected assists | float | `0.31` |
| `assists` | Assists recorded | integer | `1` |
| `clean_sheets` | Clean sheets recorded (0 or 1) | integer | `1` |
| `yellow_cards` | Yellow cards received | integer | `0` |
| `red_cards` | Red cards received | integer | `0` |
| `own_goals` | Own goals scored | integer | `0` |
| `penalties_missed` | Penalties missed | integer | `0` |
| `penalties_saved` | Penalties saved | integer | `0` |
| `bonus` | FPL bonus points | integer | `3` |
| `bps` | Bonus Points System score | integer | `42` |
| `creativity` | FPL creativity score | float | `68.4` |
| `ict_index` | FPL ICT index | float | `12.7` |
| `influence` | FPL influence score | float | `45.2` |
| `threat` | FPL threat score | float | `71.8` |
| `total_points` | Total FPL points earned | integer | `12` |



## Fields not available for every season

- **xG**
- **xA**
- **xC**
- **starts**

### Data availability by season

| Season | Availability |
|---|---|
| 2020-21 | Missing |
| 2021-22 | Missing |
| 2022-23 | Exists from GW16 |
| 2023-24 | Exists |
| 2024-25 | Exists |
| 2025-26 | Exists |

# Models

## v1:

- is_home
- team_strength
- opponent_strength

- minutes history
- points history
- goals history
- assists history
- bonus history
- bps history

- player position

# Publishing predictions

Predictions for the next unplayed gameweek are published as static JSON for other apps to consume.

## Weekly routine

1. `python fetch_player_gameweeks.py` – fetch played results and upcoming fixtures.
2. Rebuild the processed parquet (`create_player_gameweeks.py`) so the new gameweek is included.
3. `python publish_predictions.py` – writes `predictions/gw_{N}.json` and `predictions/latest.json`.

The target gameweek `N` is detected automatically: the earliest gameweek in the current season with no recorded minutes yet.

## JSON schema

```json
{
  "season": "2026-27",
  "gw": 7,
  "generated_at": "2026-10-16T13:30:28+00:00",
  "model_version": "20261009",
  "players": [
    {
      "id": 411,
      "web_name": "Haaland",
      "position": "FWD",
      "team_code": 43,
      "has_fixture": true,
      "pred_total_points": 6.7
    }
  ]
}
```

| Field | Description |
|---|---|
| `id` | FPL draft element id (same as `id` in the draft API and `players.csv`) |
| `position` | `GKP`, `DEF`, `MID` or `FWD` |
| `team_code` | FPL team `code` |
| `has_fixture` | `false` if the player's team has no fixture in the gameweek (blank gameweek) |
| `pred_total_points` | Predicted FPL points; `0` when `has_fixture` is `false` |
| `model_version` | Modified date (`YYYYMMDD`) of the model file used |

Players are sorted by `pred_total_points`, descending.

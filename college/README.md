# NCAAB College Basketball Betting Model

A machine learning betting model for NCAA Men's Basketball (NCAAB), integrated into the NBA Betting Brain repository. Uses ESPN's public API for historical game data, sklearn for modeling, and The Odds API for daily lines.

---

## Architecture

```
college/
├── __init__.py          # Module init
├── config.py            # Constants, paths, API keys
├── espn_loader.py       # ESPN API data ingestion → SQLite
├── features.py          # Feature engineering (rolling stats)
├── train.py             # Model training (Ridge + GBR)
├── daily_picks.py       # Daily picks CLI
└── README.md            # This file
```

---

## Data Sources

| Source | URL | Notes |
|--------|-----|-------|
| ESPN Scoreboard | `site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard` | No API key |
| ESPN Team Schedule | `…/teams/{id}/schedule?season={year}` | Per-team, per-season |
| ESPN Teams List | `…/teams?limit=400` | ~362 D1 teams |
| The Odds API | `api.the-odds-api.com/v4/sports/basketball_ncaab/odds/` | DraftKings spreads + totals |

Data is stored in **SQLite** (`data/ncaab_betting.db`, table `ncaab_games`).

---

## Setup

Uses the same Python virtual environment as the parent repo:

```bash
cd nba-betting-brain
source venv/bin/activate
```

---

## Step 1: Load Historical Data

```bash
# Load 2026 season only (fastest)
venv/bin/python3 college/espn_loader.py --season 2026

# Load all seasons (2022–2026)
venv/bin/python3 college/espn_loader.py --all
```

Or from Python:

```python
from college.espn_loader import load_all_seasons
load_all_seasons([2026])
```

Seasons: ESPN uses the **ending year** (e.g., `2026` = 2025–26 season).

---

## Step 2: Build Features

```bash
venv/bin/python3 college/features.py
```

Generates `data/ncaab_training_features.csv` with rolling 10-game team stats.

**Features per game:**
- `home_avg_pts_for/against` — rolling scoring averages
- `home_last5_wins` — wins in last 5 games
- `home_home_wins_pct`, `home_away_wins_pct` — split win rates
- `home_avg_margin` — avg point differential
- Same set for away team
- `home_court_flag` — 1 if home team plays at home (0 = neutral)
- `implied_total` — sum of avg scoring rates
- `efficiency_diff` — net efficiency gap (home - away)

**Targets:**
- `target_total` — actual combined score
- `target_margin` — actual home margin (positive = home win)

---

## Step 3: Train Models

```bash
venv/bin/python3 college/train.py
```

Trains two models via walk-forward validation (80% train, 20% test):

| Model | Predicts | Algorithm |
|-------|----------|-----------|
| `total_latest.pkl` | Total combined score | Best of Ridge / GBR |
| `spread_latest.pkl` | Home team margin | Best of Ridge / GBR |

Models saved to `models/saved/ncaab/`.

---

## Step 4: Daily Picks

```bash
# Today's picks (plain text)
venv/bin/python3 college/daily_picks.py

# Specific date
venv/bin/python3 college/daily_picks.py --date 2026-03-15

# Discord-formatted output (with emoji)
venv/bin/python3 college/daily_picks.py --discord
```

**Pick logic:**
1. Fetch today's games from ESPN scoreboard
2. Build rolling features from DB history
3. Run total + spread models
4. Fetch DraftKings lines from Odds API
5. Compute EV using `analyzers/ev_calculator.py`
6. Output picks sorted by EV, filter to EV ≥ 3%

---

## Notes

- **Minimum games:** Teams need ≥5 historical games before being included in features
- **NCAAB sigma:** Wider uncertainty than NBA — 14 pts for spread, 20 pts for totals
- **Home court advantage:** 3.5 pts (larger than NBA ~2.5 pts)
- **No external deps beyond repo venv** — uses sklearn only (no XGBoost/LightGBM)
- **Storage:** SQLite + CSV only (no parquet/pyarrow)

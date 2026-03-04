"""
config.py — Central configuration for NBA Betting Brain.

All tunable parameters, paths, and thresholds live here.
Environment variables override defaults where applicable.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "nba_brain.db"
MODEL_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "reports"
PERFORMANCE_FILE = BASE_DIR / "PERFORMANCE.md"

# Allow override via env var
DB_PATH = Path(os.getenv("NBA_DB_PATH", str(DB_PATH)))
MODEL_DIR = Path(os.getenv("NBA_MODEL_DIR", str(MODEL_DIR)))

# ─── Model versioning ─────────────────────────────────────────────────────────

MODEL_VERSION = "v2.0.0"
CURRENT_MODEL_NAME = f"ensemble_{MODEL_VERSION}"

# ─── Data / Training ──────────────────────────────────────────────────────────

# NBA seasons to load (format: "YYYY-YY", e.g. "2024-25")
SEASONS = ["2024-25", "2025-26"]

# Season start dates (approximate; used to bound data fetching)
SEASON_START_DATES = {
    "2024-25": "2024-10-22",
    "2025-26": "2025-10-28",
}

# Rolling window sizes
TEAM_ROLLING_WINDOW = 20      # games for rolling team ORTG/DRTG
PLAYER_ROLLING_WINDOW = 10    # games for rolling player stats
PLAYER_SHORT_WINDOW = 5       # short-form player window (last 5)

# Retrain window (for daily_retrain.py)
DAILY_RETRAIN_WINDOW_DAYS = 60

# Walk-forward backtest settings
WALK_FORWARD_TRAIN_SIZE = 300   # initial training set size
WALK_FORWARD_STEP_SIZE = 50     # games per step

# ─── Prediction / Confidence ──────────────────────────────────────────────────

# Confidence thresholds for bet recommendations
HIGH_CONFIDENCE_THRESHOLD = 0.70   # ≥70%: strong bet signal
MEDIUM_CONFIDENCE_THRESHOLD = 0.60 # 60-70%: moderate signal
MIN_BET_CONFIDENCE = 0.55          # <55%: no bet

# Auto-promote new model if ROI improves by this much over previous
AUTO_PROMOTE_ROI_DELTA = 0.02  # 2%

# League-average ORTG baseline (used in rule-based normalisation)
LEAGUE_AVG_ORTG = 113.5

# Estimated points impact per key absence (starter out)
KEY_ABSENCE_PTS_IMPACT = 4.5   # pts/game lost per key player absent

# ─── Betting / P&L Simulation ─────────────────────────────────────────────────

DEFAULT_BET_SIZE = 100.0       # unit size in dollars (or abstract units)
DEFAULT_ODDS = -110            # standard American odds for totals

# Kelly Criterion max fraction of bankroll per bet
KELLY_MAX_FRACTION = 0.25

# ─── API ──────────────────────────────────────────────────────────────────────

API_HOST = os.getenv("NBA_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("NBA_API_PORT", "8000"))

# ─── Data Sources ─────────────────────────────────────────────────────────────

NBA_CDN_BASE = "https://cdn.nba.com/static/json"
NBA_SCOREBOARD_URL = (
    NBA_CDN_BASE + "/liveData/scoreboard/todaysScoreboard_00.json"
)
NBA_SCHEDULE_URL = (
    "https://data.nba.com/data/10s/v2015/json/mobile_teams"
    "/nba/{season}/league/00_full_schedule.json"
)
NBA_BOXSCORE_BASE = (
    "https://stats.nba.com/stats/boxscoresummaryv2?GameID={game_id}"
)
BBALL_REF_BASE = "https://www.basketball-reference.com"

# Request headers (Basketball Reference rate-limits bots)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NBA_STATS_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("NBA_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

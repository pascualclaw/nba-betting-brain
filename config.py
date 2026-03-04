"""Central config for NBA Betting Brain."""
from pathlib import Path

# Paths
ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "nba_betting.db"
MODEL_SAVE_DIR = ROOT / "models" / "saved"
DATA_DIR = ROOT / "data"

# NBA seasons to load (most recent first)
ALL_SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

# Current season
CURRENT_SEASON = "2025-26"

# Seasons for training (default)
SEASONS = ALL_SEASONS[-5:]  # last 5 seasons

# API settings
NBA_API_DELAY = 0.6  # seconds between requests (rate limit)

# Model settings
ROLLING_WINDOW = 20       # games for rolling team stats
PROP_ROLLING_WINDOW = 10  # games for rolling player stats
MIN_GAMES_FOR_PREDICTION = 5  # minimum games played before we trust stats

# Betting thresholds
MIN_EDGE_TO_BET = 2.0     # minimum projected edge in pts before recommending
HIGH_PACE_WARNING_THRESHOLD = 52  # Q1 combined pts that triggers Under warning

# Database
GAMES_TABLE = "games"
SNAPSHOTS_TABLE = "team_snapshots"
PLAYER_SNAPSHOTS_TABLE = "player_snapshots"
PREDICTIONS_TABLE = "predictions"
OUTCOMES_TABLE = "outcomes"

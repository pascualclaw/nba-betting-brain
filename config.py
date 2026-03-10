"""Central config for NBA Betting Brain."""
from pathlib import Path

# Paths
ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "nba_betting.db"
MODEL_SAVE_DIR = ROOT / "models" / "saved"
DATA_DIR = ROOT / "data"

# NBA seasons to load (most recent first)
ALL_SEASONS = [
    "2010-11",
    "2011-12",
    "2012-13",
    "2013-14",
    "2014-15",
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
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
SEASONS = ALL_SEASONS  # all 15 seasons (2010-11 through 2025-26)

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

# ── Feature Flags (Multimodal Pipeline — arXiv:2410.21484) ────────────────
# Phase 1 (tonight): travel_fatigue + referee_pace enabled
# Phase 2 (future): public_betting_pct + lineup_efficiency

ENABLE_TRAVEL_FATIGUE    = True   # B2B and 3-in-4 detection ✅ built
ENABLE_REFEREE_PACE      = True   # crew foul tendency wired into MC ✅ built
ENABLE_LEVERAGE_WEIGHTS  = True   # leverage-weighted player distributions ✅ built
ENABLE_CALIBRATION       = True   # isotonic regression probability calibration ✅ built
ENABLE_LINE_MOVEMENT     = True   # opening line logging + sharp action detection ✅ built

ENABLE_PUBLIC_BETTING_PCT = False  # placeholder — needs data source (Phase 2)
ENABLE_LINEUP_EFFICIENCY  = False  # on/off lineup splits (Phase 2)
ENABLE_LIVE_RERUN         = False  # live possession-by-possession (Phase 2)

# Calibration settings
CALIBRATION_MIN_SAMPLES  = 200    # minimum games to fit isotonic regression
CALIBRATION_WINDOW       = 5000   # max historical games for calibration fit

# Evaluation metric
PRIMARY_EVAL_METRIC = "brier_score"  # NOT MAE, NOT accuracy — Brier score

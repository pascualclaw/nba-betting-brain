"""
database/schema.py — SQLite schema definitions for NBA Betting Brain.

Tables:
  games                — completed game results
  team_stats_snapshot  — rolling team stats AS OF a given date (no look-ahead)
  player_stats_snapshot— rolling player stats AS OF a given date
  predictions          — model predictions before each game
  outcomes             — actual results + P&L for each prediction
  model_runs           — training run history
  injuries_snapshot    — player injury status AS OF a given date
"""

# ─── DDL statements ───────────────────────────────────────────────────────────

CREATE_GAMES = """
CREATE TABLE IF NOT EXISTS games (
    game_id         TEXT PRIMARY KEY,
    date            TEXT NOT NULL,          -- YYYY-MM-DD
    home            TEXT NOT NULL,          -- 3-letter team abbrev
    away            TEXT NOT NULL,
    home_score      INTEGER,
    away_score      INTEGER,
    total           REAL,                   -- home_score + away_score
    q1_home         INTEGER,
    q1_away         INTEGER,
    q2_home         INTEGER,
    q2_away         INTEGER,
    q3_home         INTEGER,
    q3_away         INTEGER,
    q4_home         INTEGER,
    q4_away         INTEGER,
    ot_home         INTEGER DEFAULT 0,
    ot_away         INTEGER DEFAULT 0,
    q1_total        REAL,                   -- q1_home + q1_away (pace signal)
    pace            REAL,                   -- possessions per 48 min (if available)
    season          TEXT NOT NULL,          -- e.g. "2025-26"
    status          TEXT DEFAULT 'FINAL',   -- FINAL | LIVE | SCHEDULED
    arena           TEXT,
    is_neutral      INTEGER DEFAULT 0,      -- 1 = neutral site
    loaded_at       TEXT DEFAULT (datetime('now'))
);
"""

CREATE_TEAM_STATS_SNAPSHOT = """
CREATE TABLE IF NOT EXISTS team_stats_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team            TEXT NOT NULL,
    date            TEXT NOT NULL,          -- snapshot as-of date (YYYY-MM-DD)
    game_id         TEXT,                   -- the game this was built for
    ortg            REAL,                   -- offensive rating (last 20 games)
    drtg            REAL,                   -- defensive rating (last 20 games)
    pace            REAL,                   -- pace (last 20 games)
    net_rtg         REAL,                   -- ortg - drtg
    last_10_ortg    REAL,                   -- ortg last 10 games
    last_10_drtg    REAL,                   -- drtg last 10 games
    games_played    INTEGER,                -- how many games in rolling window
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(team, date)
);
"""

CREATE_PLAYER_STATS_SNAPSHOT = """
CREATE TABLE IF NOT EXISTS player_stats_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player          TEXT NOT NULL,          -- player full name or ID
    player_id       TEXT,                   -- NBA player ID
    team            TEXT NOT NULL,
    date            TEXT NOT NULL,          -- snapshot as-of date
    pts_avg         REAL,                   -- pts/game (last 10)
    reb_avg         REAL,                   -- reb/game (last 10)
    ast_avg         REAL,                   -- ast/game (last 10)
    pra_avg         REAL,                   -- pts+reb+ast avg (last 10)
    minutes_avg     REAL,                   -- min/game (last 10)
    last_5_avg      REAL,                   -- PRA avg last 5 games
    last_10_avg     REAL,                   -- PRA avg last 10 games
    games_played    INTEGER,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(player, date)
);
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    pred_id         TEXT PRIMARY KEY,       -- uuid
    game_id         TEXT NOT NULL,
    pred_type       TEXT NOT NULL,          -- 'total' | 'spread' | 'prop'
    predicted_value REAL,                   -- e.g. 224.5 for total
    direction       TEXT,                   -- 'over' | 'under' | 'home' | 'away'
    line            REAL,                   -- the posted line at time of prediction
    confidence      REAL,                   -- 0.0-1.0
    model_version   TEXT NOT NULL,
    features_json   TEXT,                   -- JSON blob of features used
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""

CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    pred_id         TEXT PRIMARY KEY,
    actual_value    REAL,                   -- e.g. 228 total points scored
    hit             INTEGER,                -- 1 = correct, 0 = wrong
    pnl             REAL,                   -- profit/loss in units
    odds            REAL,                   -- actual odds taken (American)
    scored_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (pred_id) REFERENCES predictions(pred_id)
);
"""

CREATE_MODEL_RUNS = """
CREATE TABLE IF NOT EXISTS model_runs (
    run_id          TEXT PRIMARY KEY,       -- uuid
    date            TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    training_games  INTEGER,
    val_accuracy    REAL,                   -- fraction correct on val set
    roi             REAL,                   -- return on investment
    sharpe          REAL,                   -- Sharpe ratio
    max_drawdown    REAL,
    notes           TEXT,
    promoted        INTEGER DEFAULT 0,      -- 1 if this version was auto-promoted
    created_at      TEXT DEFAULT (datetime('now'))
);
"""

CREATE_INJURIES_SNAPSHOT = """
CREATE TABLE IF NOT EXISTS injuries_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team            TEXT NOT NULL,
    player          TEXT NOT NULL,
    date            TEXT NOT NULL,          -- snapshot date
    status          TEXT NOT NULL,          -- 'Out' | 'Questionable' | 'Doubtful' | 'Available'
    reason          TEXT,                   -- injury description
    key_absence     INTEGER DEFAULT 0,      -- 1 = starter / key player
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(team, player, date)
);
"""

# ─── Index DDL ────────────────────────────────────────────────────────────────

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);",
    "CREATE INDEX IF NOT EXISTS idx_games_home ON games(home);",
    "CREATE INDEX IF NOT EXISTS idx_games_away ON games(away);",
    "CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);",
    "CREATE INDEX IF NOT EXISTS idx_team_stats_team_date ON team_stats_snapshot(team, date);",
    "CREATE INDEX IF NOT EXISTS idx_player_stats_player_date ON player_stats_snapshot(player, date);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(pred_type);",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_hit ON outcomes(hit);",
    "CREATE INDEX IF NOT EXISTS idx_model_runs_date ON model_runs(date);",
    "CREATE INDEX IF NOT EXISTS idx_injuries_team_date ON injuries_snapshot(team, date);",
]

# ─── All DDL in order ─────────────────────────────────────────────────────────

ALL_SCHEMA = [
    CREATE_GAMES,
    CREATE_TEAM_STATS_SNAPSHOT,
    CREATE_PLAYER_STATS_SNAPSHOT,
    CREATE_PREDICTIONS,
    CREATE_OUTCOMES,
    CREATE_MODEL_RUNS,
    CREATE_INJURIES_SNAPSHOT,
    *INDEXES,
]

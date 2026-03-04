"""SQLite schema — creates all tables."""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,
    date        TEXT NOT NULL,
    season      TEXT NOT NULL,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    home_score  INTEGER,
    away_score  INTEGER,
    total       INTEGER,
    winner      TEXT,
    home_margin INTEGER,
    q1_total    INTEGER,
    q2_total    INTEGER,
    q3_total    INTEGER,
    q4_total    INTEGER,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS team_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT NOT NULL,
    date            TEXT NOT NULL,
    team            TEXT NOT NULL,
    is_home         INTEGER NOT NULL,
    pts_for_avg     REAL,
    pts_against_avg REAL,
    total_avg       REAL,
    net_rating      REAL,
    pace_proxy      REAL,
    win_pct         REAL,
    games           INTEGER,
    last5_pts_for   REAL,
    last5_pts_against REAL,
    h2h_total_avg   REAL,
    h2h_games       INTEGER,
    h2h_over220_rate REAL,
    UNIQUE(game_id, team)
);

CREATE TABLE IF NOT EXISTS player_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT NOT NULL,
    date            TEXT NOT NULL,
    player          TEXT NOT NULL,
    team            TEXT NOT NULL,
    pts_avg_10      REAL,
    reb_avg_10      REAL,
    ast_avg_10      REAL,
    pra_avg_10      REAL,
    min_avg_10      REAL,
    pts_avg_5       REAL,
    reb_avg_5       REAL,
    pra_avg_5       REAL,
    is_starting     INTEGER,
    games_played    INTEGER,
    UNIQUE(game_id, player)
);

CREATE TABLE IF NOT EXISTS predictions (
    pred_id         TEXT PRIMARY KEY,
    game_id         TEXT NOT NULL,
    date            TEXT NOT NULL,
    pred_type       TEXT NOT NULL,
    player          TEXT,
    predicted_value REAL,
    direction       TEXT,
    line            REAL,
    confidence      REAL,
    model_version   TEXT,
    features_json   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS outcomes (
    pred_id         TEXT PRIMARY KEY,
    actual_value    REAL,
    hit             INTEGER,
    pnl             REAL,
    odds            INTEGER,
    notes           TEXT,
    scored_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(pred_id) REFERENCES predictions(pred_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id          TEXT PRIMARY KEY,
    date            TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    training_games  INTEGER,
    val_mae         REAL,
    val_direction_acc REAL,
    betting_roi     REAL,
    top_features    TEXT,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_snapshots_game ON team_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_pred ON outcomes(pred_id);
"""


def create_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


if __name__ == "__main__":
    from config import DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    print(f"Schema created at {DB_PATH}")
    conn.close()

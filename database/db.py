"""Database connection manager and helper functions."""
import sqlite3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from database.schema import create_schema


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    create_schema(conn)
    return conn


def upsert_game(conn: sqlite3.Connection, game: dict):
    conn.execute("""
        INSERT OR REPLACE INTO games
        (game_id, date, season, home, away, home_score, away_score, total, winner, home_margin)
        VALUES (:game_id, :date, :season, :home, :away, :home_score, :away_score,
                :total, :winner, :home_margin)
    """, {**game, "home_margin": game.get("home_margin", game.get("home_score", 0) - game.get("away_score", 0))})


def upsert_team_snapshot(conn: sqlite3.Connection, snap: dict):
    conn.execute("""
        INSERT OR REPLACE INTO team_snapshots
        (game_id, date, team, is_home, pts_for_avg, pts_against_avg, total_avg,
         net_rating, pace_proxy, win_pct, games, last5_pts_for, last5_pts_against,
         h2h_total_avg, h2h_games, h2h_over220_rate)
        VALUES (:game_id, :date, :team, :is_home, :pts_for_avg, :pts_against_avg, :total_avg,
                :net_rating, :pace_proxy, :win_pct, :games, :last5_pts_for, :last5_pts_against,
                :h2h_total_avg, :h2h_games, :h2h_over220_rate)
    """, snap)


def get_games_for_season(conn: sqlite3.Connection, season: str) -> list:
    rows = conn.execute("SELECT * FROM games WHERE season=? ORDER BY date", (season,)).fetchall()
    return [dict(r) for r in rows]


def get_all_games(conn: sqlite3.Connection) -> list:
    rows = conn.execute("SELECT * FROM games ORDER BY date").fetchall()
    return [dict(r) for r in rows]


def count_games(conn: sqlite3.Connection, season: str = None) -> int:
    if season:
        return conn.execute("SELECT COUNT(*) FROM games WHERE season=?", (season,)).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]

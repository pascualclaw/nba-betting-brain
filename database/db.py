"""
database/db.py — SQLite connection manager, migration runner, and upsert helpers.

Usage:
    from database.db import DB

    with DB() as db:
        db.upsert_game(game_dict)
        rows = db.query("SELECT * FROM games WHERE date = ?", ["2025-12-01"])
"""

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from database.schema import ALL_SCHEMA

logger = logging.getLogger(__name__)


# ─── DB class ─────────────────────────────────────────────────────────────────

class DB:
    """
    Thread-safe SQLite database wrapper with migration support and upsert helpers.

    Can be used as a context manager:
        with DB() as db:
            db.execute(...)

    Or instantiated directly:
        db = DB()
        db.connect()
        ...
        db.close()
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        if path is None:
            from config import DB_PATH
            path = DB_PATH

        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> "DB":
        """Open the SQLite connection and apply WAL mode for concurrency."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        logger.debug("Connected to SQLite at %s", self.path)
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DB":
        self.connect()
        self.migrate()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Migration ─────────────────────────────────────────────────────────────

    def migrate(self) -> None:
        """Apply all schema DDL statements (idempotent — uses IF NOT EXISTS)."""
        assert self._conn, "Not connected"
        with self._conn:
            for ddl in ALL_SCHEMA:
                self._conn.execute(ddl)
        logger.debug("Schema migration complete.")

    # ── Query helpers ─────────────────────────────────────────────────────────

    def execute(self, sql: str, params: list[Any] | None = None) -> sqlite3.Cursor:
        """Execute a single SQL statement."""
        assert self._conn, "Not connected"
        params = params or []
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, param_list: list[list[Any]]) -> sqlite3.Cursor:
        assert self._conn, "Not connected"
        return self._conn.executemany(sql, param_list)

    def query(self, sql: str, params: list[Any] | None = None) -> list[dict]:
        """Execute a SELECT and return list of dicts."""
        cursor = self.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def query_one(self, sql: str, params: list[Any] | None = None) -> Optional[dict]:
        """Execute a SELECT and return first row as dict, or None."""
        cursor = self.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager for an explicit transaction."""
        assert self._conn, "Not connected"
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── Upsert helpers ────────────────────────────────────────────────────────

    def upsert_game(self, game: dict) -> None:
        """Insert or replace a game record."""
        sql = """
        INSERT INTO games (
            game_id, date, home, away, home_score, away_score, total,
            q1_home, q1_away, q2_home, q2_away, q3_home, q3_away,
            q4_home, q4_away, ot_home, ot_away, q1_total,
            pace, season, status, arena, is_neutral
        ) VALUES (
            :game_id, :date, :home, :away, :home_score, :away_score, :total,
            :q1_home, :q1_away, :q2_home, :q2_away, :q3_home, :q3_away,
            :q4_home, :q4_away, :ot_home, :ot_away, :q1_total,
            :pace, :season, :status, :arena, :is_neutral
        )
        ON CONFLICT(game_id) DO UPDATE SET
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            total = excluded.total,
            q1_home = excluded.q1_home,
            q1_away = excluded.q1_away,
            q2_home = excluded.q2_home,
            q2_away = excluded.q2_away,
            q3_home = excluded.q3_home,
            q3_away = excluded.q3_away,
            q4_home = excluded.q4_home,
            q4_away = excluded.q4_away,
            ot_home = excluded.ot_home,
            ot_away = excluded.ot_away,
            q1_total = excluded.q1_total,
            pace = excluded.pace,
            status = excluded.status;
        """
        _defaults = {
            "q1_home": None, "q1_away": None,
            "q2_home": None, "q2_away": None,
            "q3_home": None, "q3_away": None,
            "q4_home": None, "q4_away": None,
            "ot_home": 0, "ot_away": 0,
            "q1_total": None, "pace": None,
            "status": "FINAL", "arena": None, "is_neutral": 0,
        }
        record = {**_defaults, **game}
        with self.transaction():
            self.execute(sql, record)  # type: ignore[arg-type]

    def upsert_team_snapshot(self, snap: dict) -> None:
        """Insert or update a team stats snapshot."""
        sql = """
        INSERT INTO team_stats_snapshot (
            team, date, game_id, ortg, drtg, pace, net_rtg,
            last_10_ortg, last_10_drtg, games_played
        ) VALUES (
            :team, :date, :game_id, :ortg, :drtg, :pace, :net_rtg,
            :last_10_ortg, :last_10_drtg, :games_played
        )
        ON CONFLICT(team, date) DO UPDATE SET
            ortg = excluded.ortg,
            drtg = excluded.drtg,
            pace = excluded.pace,
            net_rtg = excluded.net_rtg,
            last_10_ortg = excluded.last_10_ortg,
            last_10_drtg = excluded.last_10_drtg,
            games_played = excluded.games_played;
        """
        _defaults = {"game_id": None, "net_rtg": None, "games_played": 0}
        record = {**_defaults, **snap}
        with self.transaction():
            self.execute(sql, record)  # type: ignore[arg-type]

    def upsert_player_snapshot(self, snap: dict) -> None:
        """Insert or update a player stats snapshot."""
        sql = """
        INSERT INTO player_stats_snapshot (
            player, player_id, team, date,
            pts_avg, reb_avg, ast_avg, pra_avg, minutes_avg,
            last_5_avg, last_10_avg, games_played
        ) VALUES (
            :player, :player_id, :team, :date,
            :pts_avg, :reb_avg, :ast_avg, :pra_avg, :minutes_avg,
            :last_5_avg, :last_10_avg, :games_played
        )
        ON CONFLICT(player, date) DO UPDATE SET
            pts_avg = excluded.pts_avg,
            reb_avg = excluded.reb_avg,
            ast_avg = excluded.ast_avg,
            pra_avg = excluded.pra_avg,
            minutes_avg = excluded.minutes_avg,
            last_5_avg = excluded.last_5_avg,
            last_10_avg = excluded.last_10_avg,
            games_played = excluded.games_played;
        """
        _defaults = {"player_id": None, "last_5_avg": None, "last_10_avg": None}
        record = {**_defaults, **snap}
        with self.transaction():
            self.execute(sql, record)  # type: ignore[arg-type]

    def insert_prediction(self, pred: dict) -> str:
        """Insert a new prediction record. Returns pred_id."""
        pred_id = pred.get("pred_id") or str(uuid.uuid4())
        sql = """
        INSERT INTO predictions (
            pred_id, game_id, pred_type, predicted_value, direction,
            line, confidence, model_version, features_json
        ) VALUES (
            :pred_id, :game_id, :pred_type, :predicted_value, :direction,
            :line, :confidence, :model_version, :features_json
        )
        ON CONFLICT(pred_id) DO NOTHING;
        """
        record = {**pred, "pred_id": pred_id}
        if "features_json" in record and not isinstance(record["features_json"], str):
            record["features_json"] = json.dumps(record["features_json"])
        with self.transaction():
            self.execute(sql, record)  # type: ignore[arg-type]
        return pred_id

    def upsert_outcome(self, outcome: dict) -> None:
        """Insert or update an outcome record."""
        sql = """
        INSERT INTO outcomes (pred_id, actual_value, hit, pnl, odds)
        VALUES (:pred_id, :actual_value, :hit, :pnl, :odds)
        ON CONFLICT(pred_id) DO UPDATE SET
            actual_value = excluded.actual_value,
            hit = excluded.hit,
            pnl = excluded.pnl,
            odds = excluded.odds;
        """
        with self.transaction():
            self.execute(sql, outcome)  # type: ignore[arg-type]

    def insert_model_run(self, run: dict) -> str:
        """Log a training run. Returns run_id."""
        run_id = run.get("run_id") or str(uuid.uuid4())
        sql = """
        INSERT INTO model_runs (
            run_id, date, model_version, training_games,
            val_accuracy, roi, sharpe, max_drawdown, notes, promoted
        ) VALUES (
            :run_id, :date, :model_version, :training_games,
            :val_accuracy, :roi, :sharpe, :max_drawdown, :notes, :promoted
        );
        """
        _defaults = {
            "sharpe": None, "max_drawdown": None,
            "notes": None, "promoted": 0,
        }
        record = {**_defaults, **run, "run_id": run_id}
        with self.transaction():
            self.execute(sql, record)  # type: ignore[arg-type]
        return run_id

    def upsert_injury(self, injury: dict) -> None:
        """Insert or update an injury snapshot record."""
        sql = """
        INSERT INTO injuries_snapshot (team, player, date, status, reason, key_absence)
        VALUES (:team, :player, :date, :status, :reason, :key_absence)
        ON CONFLICT(team, player, date) DO UPDATE SET
            status = excluded.status,
            reason = excluded.reason,
            key_absence = excluded.key_absence;
        """
        _defaults = {"reason": None, "key_absence": 0}
        record = {**_defaults, **injury}
        with self.transaction():
            self.execute(sql, record)  # type: ignore[arg-type]

    # ── Convenience queries ───────────────────────────────────────────────────

    def get_games_for_training(self, season: Optional[str] = None) -> list[dict]:
        """Return all FINAL games, optionally filtered by season."""
        if season:
            return self.query(
                "SELECT * FROM games WHERE status = 'FINAL' AND season = ? ORDER BY date ASC",
                [season],
            )
        return self.query(
            "SELECT * FROM games WHERE status = 'FINAL' ORDER BY date ASC"
        )

    def get_team_snapshot(self, team: str, date: str) -> Optional[dict]:
        """Get the most recent team snapshot on or before `date`."""
        return self.query_one(
            """SELECT * FROM team_stats_snapshot
               WHERE team = ? AND date <= ?
               ORDER BY date DESC LIMIT 1""",
            [team, date],
        )

    def get_h2h_games(
        self, home: str, away: str, season: str, before_date: str
    ) -> list[dict]:
        """Return all H2H games between two teams in a season before a date."""
        return self.query(
            """SELECT * FROM games
               WHERE season = ? AND date < ? AND status = 'FINAL'
               AND ((home = ? AND away = ?) OR (home = ? AND away = ?))
               ORDER BY date ASC""",
            [season, before_date, home, away, away, home],
        )

    def get_injuries_on_date(self, team: str, date: str) -> list[dict]:
        """Get injury snapshot for a team on or before `date`."""
        return self.query(
            """SELECT * FROM injuries_snapshot
               WHERE team = ? AND date <= ?
               ORDER BY date DESC""",
            [team, date],
        )

    def game_exists(self, game_id: str) -> bool:
        """Check if a game_id is already in the database."""
        row = self.query_one(
            "SELECT 1 FROM games WHERE game_id = ?", [game_id]
        )
        return row is not None

    def date_loaded(self, date: str, season: str) -> bool:
        """Check if we've already loaded games for a given date+season."""
        row = self.query_one(
            "SELECT 1 FROM games WHERE date = ? AND season = ? LIMIT 1",
            [date, season],
        )
        return row is not None

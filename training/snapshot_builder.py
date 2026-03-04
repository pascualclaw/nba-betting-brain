"""
training/snapshot_builder.py — Reconstruct pre-game feature snapshots.

For each historical game, this module rebuilds what stats were available
BEFORE the game tip-off — using only rolling windows of prior games.

No look-ahead. Ever.

Usage:
    from training.snapshot_builder import SnapshotBuilder
    builder = SnapshotBuilder(db)
    snap = builder.build_pre_game_snapshot(game_id="0022500312")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from config import (
    KEY_ABSENCE_PTS_IMPACT,
    PLAYER_ROLLING_WINDOW,
    PLAYER_SHORT_WINDOW,
    TEAM_ROLLING_WINDOW,
)
from database.db import DB

logger = logging.getLogger(__name__)


class SnapshotBuilder:
    """
    Builds pre-game snapshots for a given game using only historically
    available data (rolling windows, no look-ahead).
    """

    def __init__(self, db: DB) -> None:
        self.db = db

    # ── Team snapshots ────────────────────────────────────────────────────────

    def get_team_stats(self, team: str, before_date: str) -> dict:
        """
        Return the team's rolling ORTG/DRTG as of `before_date`.
        Falls back to league-average defaults if insufficient data.
        """
        snap = self.db.get_team_snapshot(team, before_date)
        if snap:
            return snap

        # No data yet — return league average defaults
        logger.debug(
            "No team snapshot for %s before %s; using defaults.", team, before_date
        )
        return {
            "team": team,
            "date": before_date,
            "ortg": 110.0,
            "drtg": 110.0,
            "pace": 98.0,
            "net_rtg": 0.0,
            "last_10_ortg": 110.0,
            "last_10_drtg": 110.0,
            "games_played": 0,
        }

    # ── Player snapshots ──────────────────────────────────────────────────────

    def get_player_stats(self, player: str, before_date: str) -> Optional[dict]:
        """
        Return the player's rolling stats as of `before_date`.
        """
        return self.db.query_one(
            """SELECT * FROM player_stats_snapshot
               WHERE player = ? AND date <= ?
               ORDER BY date DESC LIMIT 1""",
            [player, before_date],
        )

    def get_team_players(self, team: str, before_date: str) -> list[dict]:
        """
        Return all players with a recent snapshot for the team.
        Uses the most recent snapshot per player before `before_date`.
        """
        rows = self.db.query(
            """SELECT p.*
               FROM player_stats_snapshot p
               INNER JOIN (
                   SELECT player, MAX(date) AS max_date
                   FROM player_stats_snapshot
                   WHERE team = ? AND date <= ?
                   GROUP BY player
               ) latest ON p.player = latest.player AND p.date = latest.max_date
               WHERE p.team = ?""",
            [team, before_date, team],
        )
        return rows

    # ── Injury snapshots ──────────────────────────────────────────────────────

    def get_injuries(self, team: str, before_date: str) -> list[dict]:
        """
        Return active injury records for the team on or before `before_date`.
        De-duplicates to the most recent status per player.
        """
        rows = self.db.query(
            """SELECT i.*
               FROM injuries_snapshot i
               INNER JOIN (
                   SELECT team, player, MAX(date) AS max_date
                   FROM injuries_snapshot
                   WHERE team = ? AND date <= ?
                   GROUP BY team, player
               ) latest ON i.team = latest.team
                        AND i.player = latest.player
                        AND i.date = latest.max_date
               WHERE i.team = ?
               AND i.status IN ('Out', 'Doubtful')""",
            [team, before_date, team],
        )
        return rows

    def compute_absence_impact(self, team: str, before_date: str) -> tuple[int, float]:
        """
        Compute (key_absences_count, estimated_pts_impact) for a team.

        Returns:
            key_absences: number of starters/key players out
            absence_impact: estimated pts/game lost
        """
        injuries = self.get_injuries(team, before_date)
        key_out = [i for i in injuries if i.get("key_absence", 0) == 1]
        key_absences = len(key_out)
        absence_impact = key_absences * KEY_ABSENCE_PTS_IMPACT
        return key_absences, round(absence_impact, 2)

    # ── Rest / B2B ────────────────────────────────────────────────────────────

    def get_rest_days(self, team: str, before_date: str) -> int:
        """
        Return number of rest days for a team before `before_date`.
        Calculated from the team's most recent game date.
        """
        row = self.db.query_one(
            """SELECT MAX(date) AS last_game
               FROM games
               WHERE (home = ? OR away = ?)
               AND date < ?
               AND status = 'FINAL'""",
            [team, team, before_date],
        )
        if not row or not row.get("last_game"):
            return 7  # Assume a full week rest if no prior game

        last_date = datetime.strptime(row["last_game"], "%Y-%m-%d").date()
        today = datetime.strptime(before_date, "%Y-%m-%d").date()
        return (today - last_date).days

    def is_b2b(self, team: str, before_date: str) -> bool:
        """Return True if the team played yesterday (back-to-back)."""
        return self.get_rest_days(team, before_date) == 1

    # ── H2H history ───────────────────────────────────────────────────────────

    def get_h2h_stats(
        self, home: str, away: str, season: str, before_date: str
    ) -> dict:
        """
        Compute H2H stats between home and away teams before `before_date`.

        Returns a dict with avg_total, q1_avg, games count.
        """
        games = self.db.get_h2h_games(home, away, season, before_date)

        if not games:
            return {
                "h2h_avg_total": 220.0,  # league average fallback
                "h2h_q1_avg": 55.0,
                "h2h_games": 0,
            }

        totals = [g["total"] for g in games if g.get("total") is not None]
        q1_totals = [g["q1_total"] for g in games if g.get("q1_total") is not None]

        return {
            "h2h_avg_total": round(sum(totals) / len(totals), 2) if totals else 220.0,
            "h2h_q1_avg": (
                round(sum(q1_totals) / len(q1_totals), 2) if q1_totals else 55.0
            ),
            "h2h_games": len(games),
        }

    # ── Full pre-game snapshot ────────────────────────────────────────────────

    def build_pre_game_snapshot(self, game: dict) -> dict:
        """
        Build a complete pre-game snapshot for a given game record.

        Args:
            game: dict from games table (must include game_id, date, home, away, season)

        Returns:
            snapshot dict containing all pre-game context (no look-ahead)
        """
        game_id = game["game_id"]
        game_date = game["date"]
        home = game["home"]
        away = game["away"]
        season = game["season"]

        # Team stats
        home_stats = self.get_team_stats(home, game_date)
        away_stats = self.get_team_stats(away, game_date)

        # Rest & schedule
        home_rest = self.get_rest_days(home, game_date)
        away_rest = self.get_rest_days(away, game_date)

        # Injuries
        home_key_absences, home_absence_impact = self.compute_absence_impact(
            home, game_date
        )
        away_key_absences, away_absence_impact = self.compute_absence_impact(
            away, game_date
        )

        # H2H
        h2h = self.get_h2h_stats(home, away, season, game_date)

        snapshot = {
            "game_id": game_id,
            "date": game_date,
            "home": home,
            "away": away,
            "season": season,
            # Home team stats
            "home_ortg": home_stats.get("ortg", 110.0),
            "home_drtg": home_stats.get("drtg", 110.0),
            "home_pace": home_stats.get("pace", 98.0),
            "home_net_rtg": home_stats.get("net_rtg", 0.0),
            "home_last10_ortg": home_stats.get("last_10_ortg", 110.0),
            "home_last10_drtg": home_stats.get("last_10_drtg", 110.0),
            "home_games_played": home_stats.get("games_played", 0),
            # Away team stats
            "away_ortg": away_stats.get("ortg", 110.0),
            "away_drtg": away_stats.get("drtg", 110.0),
            "away_pace": away_stats.get("pace", 98.0),
            "away_net_rtg": away_stats.get("net_rtg", 0.0),
            "away_last10_ortg": away_stats.get("last_10_ortg", 110.0),
            "away_last10_drtg": away_stats.get("last_10_drtg", 110.0),
            "away_games_played": away_stats.get("games_played", 0),
            # Schedule
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "home_b2b": int(home_rest == 1),
            "away_b2b": int(away_rest == 1),
            # Injuries
            "home_key_absences": home_key_absences,
            "away_key_absences": away_key_absences,
            "home_absence_impact": home_absence_impact,
            "away_absence_impact": away_absence_impact,
            # H2H
            "h2h_avg_total": h2h["h2h_avg_total"],
            "h2h_q1_avg": h2h["h2h_q1_avg"],
            "h2h_games": h2h["h2h_games"],
            # Location
            "is_neutral": game.get("is_neutral", 0),
        }

        logger.debug("Built snapshot for game %s (%s @ %s)", game_id, away, home)
        return snapshot

    def build_all_snapshots(
        self, games: list[dict], show_progress: bool = True
    ) -> list[dict]:
        """
        Build pre-game snapshots for all games in chronological order.

        Args:
            games: list of game dicts (should be sorted by date)
            show_progress: whether to show rich progress bar

        Returns:
            list of snapshot dicts, parallel to games list
        """
        snapshots = []

        if show_progress:
            try:
                from rich.progress import track
                iterable = track(
                    games, description="Building snapshots…", transient=True
                )
            except ImportError:
                iterable = games
        else:
            iterable = games

        for game in iterable:
            snap = self.build_pre_game_snapshot(game)
            snapshots.append(snap)

        return snapshots

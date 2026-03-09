"""
Team Over/Under Rate Features

Calculates per-team O/U performance metrics from historical game data:
  - home_ou_rate: fraction of home games that went OVER the game total average
  - away_ou_rate: fraction of away games that went OVER the game total average
  - recent_ou_rate: O/U rate over the last 15 games (home + away combined)
  - avg_game_total: average total points scored in team's games
  - defensive_suppression: team's avg total vs league average (negative = team holds totals down)

Exposed API:
  get_team_ou_features(team_abbr, date, conn, n=30) -> dict
  get_matchup_ou_features(home, away, date, conn) -> dict

Usage:
    import sqlite3
    conn = sqlite3.connect("data/nba_betting.db")
    feats = get_matchup_ou_features("LAL", "BOS", "2025-03-01", conn)
"""

import sqlite3
import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)

# League-average total for fallback/normalization (updated seasonally)
_LEAGUE_AVG_TOTAL = 224.0
_LEAGUE_AVG_TOTAL_CACHE: Optional[float] = None


def _get_league_avg_total(conn: sqlite3.Connection) -> float:
    """Compute league-wide average game total from all games in DB."""
    global _LEAGUE_AVG_TOTAL_CACHE
    if _LEAGUE_AVG_TOTAL_CACHE is not None:
        return _LEAGUE_AVG_TOTAL_CACHE
    try:
        row = conn.execute(
            "SELECT AVG(total) FROM games WHERE total IS NOT NULL AND total > 150"
        ).fetchone()
        if row and row[0]:
            _LEAGUE_AVG_TOTAL_CACHE = float(row[0])
            return _LEAGUE_AVG_TOTAL_CACHE
    except Exception as e:
        log.debug(f"Could not compute league avg total: {e}")
    _LEAGUE_AVG_TOTAL_CACHE = _LEAGUE_AVG_TOTAL
    return _LEAGUE_AVG_TOTAL_CACHE


def _fetch_team_games(
    team_abbr: str,
    before_date: str,
    conn: sqlite3.Connection,
    n: int = 30,
    location: Optional[str] = None,  # "home", "away", or None for all
) -> list:
    """
    Fetch up to `n` games for a team before `before_date`.
    Returns list of dicts with keys: date, total, is_home
    """
    if location == "home":
        query = """
            SELECT date, total, 1 as is_home
            FROM games
            WHERE home = ? AND date < ? AND total IS NOT NULL AND total > 150
            ORDER BY date DESC
            LIMIT ?
        """
        rows = conn.execute(query, (team_abbr, before_date, n)).fetchall()
    elif location == "away":
        query = """
            SELECT date, total, 0 as is_home
            FROM games
            WHERE away = ? AND date < ? AND total IS NOT NULL AND total > 150
            ORDER BY date DESC
            LIMIT ?
        """
        rows = conn.execute(query, (team_abbr, before_date, n)).fetchall()
    else:
        # All games (home + away), combined, most recent first
        query = """
            SELECT date, total,
                   CASE WHEN home = ? THEN 1 ELSE 0 END as is_home
            FROM games
            WHERE (home = ? OR away = ?)
              AND date < ?
              AND total IS NOT NULL AND total > 150
            ORDER BY date DESC
            LIMIT ?
        """
        rows = conn.execute(query, (team_abbr, team_abbr, team_abbr, before_date, n)).fetchall()

    return [{"date": r[0], "total": r[1], "is_home": r[2]} for r in rows]


def _compute_ou_rate(totals: list, threshold: float) -> float:
    """Fraction of games where total > threshold."""
    if not totals:
        return 0.5
    return round(sum(1 for t in totals if t > threshold) / len(totals), 3)


def get_team_ou_features(
    team_abbr: str,
    date: str,
    conn: sqlite3.Connection,
    n: int = 30,
) -> Dict:
    """
    Calculate team-level O/U features using only games before `date`.

    Args:
        team_abbr: NBA team abbreviation (e.g. "LAL")
        date: Cutoff date string "YYYY-MM-DD" — uses only prior games
        conn: SQLite connection to nba_betting.db
        n: Rolling window size for home/away rates

    Returns:
        dict with keys:
          home_ou_rate, away_ou_rate, recent_ou_rate,
          avg_game_total, defensive_suppression,
          home_games, away_games, recent_games
    """
    league_avg = _get_league_avg_total(conn)

    home_games = _fetch_team_games(team_abbr, date, conn, n=n, location="home")
    away_games = _fetch_team_games(team_abbr, date, conn, n=n, location="away")
    recent_games = _fetch_team_games(team_abbr, date, conn, n=15, location=None)

    home_totals = [g["total"] for g in home_games]
    away_totals = [g["total"] for g in away_games]
    recent_totals = [g["total"] for g in recent_games]
    all_totals = home_totals + away_totals

    # Use team's own avg total as the O/U threshold (more relevant than league avg for rate calc)
    team_avg = (sum(all_totals) / len(all_totals)) if all_totals else league_avg

    home_ou_rate = _compute_ou_rate(home_totals, team_avg)
    away_ou_rate = _compute_ou_rate(away_totals, team_avg)
    recent_ou_rate = _compute_ou_rate(recent_totals, league_avg)

    avg_game_total = round(team_avg, 1)

    # Defensive suppression: negative means team holds totals below league avg
    defensive_suppression = round(avg_game_total - league_avg, 2)

    return {
        "home_ou_rate": home_ou_rate,
        "away_ou_rate": away_ou_rate,
        "recent_ou_rate": recent_ou_rate,
        "avg_game_total": avg_game_total,
        "defensive_suppression": defensive_suppression,
        "home_games": len(home_games),
        "away_games": len(away_games),
        "recent_games": len(recent_games),
    }


def get_matchup_ou_features(
    home: str,
    away: str,
    date: str,
    conn: sqlite3.Connection,
    n: int = 30,
) -> Dict:
    """
    Compute combined O/U features for a home vs away matchup.

    Args:
        home: Home team abbreviation
        away: Away team abbreviation
        date: Game date cutoff "YYYY-MM-DD"
        conn: SQLite connection

    Returns:
        dict with prefixed home_* and away_* features plus combined metrics.
    """
    home_feats = get_team_ou_features(home, date, conn, n=n)
    away_feats = get_team_ou_features(away, date, conn, n=n)

    combined = {}

    for k, v in home_feats.items():
        combined[f"home_{k}"] = v

    for k, v in away_feats.items():
        combined[f"away_{k}"] = v

    # Combined/derived features
    avg_ou_rate = round(
        (home_feats["home_ou_rate"] + away_feats["away_ou_rate"]) / 2, 3
    )
    projected_total_ou = round(
        (home_feats["avg_game_total"] + away_feats["avg_game_total"]) / 2, 1
    )
    combined_suppression = round(
        home_feats["defensive_suppression"] + away_feats["defensive_suppression"], 2
    )

    combined["matchup_avg_ou_rate"] = avg_ou_rate
    combined["matchup_projected_total"] = projected_total_ou
    combined["matchup_combined_suppression"] = combined_suppression

    return combined


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    feats = get_team_ou_features("LAL", "2025-03-01", conn, n=30)
    print("LAL O/U features:", feats)

    matchup = get_matchup_ou_features("LAL", "BOS", "2025-03-01", conn)
    print("\nLAL vs BOS matchup features:")
    for k, v in matchup.items():
        print(f"  {k}: {v}")
    conn.close()

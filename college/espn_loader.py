"""
ESPN NCAAB Data Loader

Loads college basketball game data from ESPN's public API.
Stores to SQLite for feature engineering and model training.

Usage:
    from college.espn_loader import load_all_seasons, load_season_games
    load_all_seasons([2026])
"""

import sys
import time
import sqlite3
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, '.')

from college.config import (
    NCAAB_DB_PATH, NCAAB_SEASONS, ESPN_BASE, ESPN_SCOREBOARD,
    ESPN_TEAMS, REQUEST_DELAY
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ── Database Setup ─────────────────────────────────────────────────────────

def get_db_conn() -> sqlite3.Connection:
    """Get SQLite connection with WAL mode for better concurrency."""
    import os
    os.makedirs(os.path.dirname(NCAAB_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(NCAAB_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ncaab_games (
            game_id     TEXT PRIMARY KEY,
            date        TEXT NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            home_score  INTEGER,
            away_score  INTEGER,
            neutral_site INTEGER DEFAULT 0,
            season      INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ncaab_games_date ON ncaab_games(date)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ncaab_games_season ON ncaab_games(season)
    """)
    conn.commit()
    conn.close()


def get_loaded_game_ids() -> set:
    """Return set of game_ids already in the DB."""
    conn = get_db_conn()
    rows = conn.execute("SELECT game_id FROM ncaab_games").fetchall()
    conn.close()
    return {r[0] for r in rows}


def insert_games(games: List[Dict]) -> int:
    """Insert games into DB, skip duplicates. Returns count inserted."""
    if not games:
        return 0
    conn = get_db_conn()
    inserted = 0
    for g in games:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO ncaab_games
                   (game_id, date, home_team_id, away_team_id,
                    home_score, away_score, neutral_site, season)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (g['game_id'], g['date'], g['home_team_id'], g['away_team_id'],
                 g['home_score'], g['away_score'], g.get('neutral_site', 0), g['season'])
            )
            if conn.execute("SELECT changes()").fetchone()[0] > 0:
                inserted += 1
        except Exception as e:
            log.warning(f"Failed to insert game {g.get('game_id')}: {e}")
    conn.commit()
    conn.close()
    return inserted


# ── ESPN API Helpers ────────────────────────────────────────────────────────

def _get(url: str, params: Dict = None, retries: int = 3) -> Optional[Dict]:
    """GET request with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                log.warning(f"Rate limited, sleeping 5s...")
                time.sleep(5)
            else:
                log.warning(f"HTTP {resp.status_code} for {url}")
        except Exception as e:
            log.warning(f"Request failed (attempt {attempt+1}): {e}")
            time.sleep(1)
    return None


# ── Public API Functions ────────────────────────────────────────────────────

def get_all_teams() -> Dict[str, Dict]:
    """
    Fetch all NCAAB teams from ESPN.
    Returns: {team_id: {name, abbreviation}}
    """
    data = _get(ESPN_TEAMS, params={"limit": 400})
    if not data:
        log.error("Failed to fetch teams")
        return {}

    teams = {}
    sports = data.get('sports', [])
    for sport in sports:
        for league in sport.get('leagues', []):
            for team in league.get('teams', []):
                t = team.get('team', {})
                tid = str(t.get('id', ''))
                if tid:
                    teams[tid] = {
                        'name': t.get('displayName', t.get('name', '')),
                        'abbreviation': t.get('abbreviation', ''),
                        'location': t.get('location', ''),
                    }

    # Try alternate response structure
    if not teams:
        for team in data.get('teams', []):
            t = team.get('team', team)
            tid = str(t.get('id', ''))
            if tid:
                teams[tid] = {
                    'name': t.get('displayName', t.get('name', '')),
                    'abbreviation': t.get('abbreviation', ''),
                    'location': t.get('location', ''),
                }

    log.info(f"Loaded {len(teams)} teams")
    return teams


def get_team_schedule(team_id: str, season: int) -> List[Dict]:
    """
    Fetch completed games for a team in a given season.
    Returns list of game dicts with scores, home/away, date.
    """
    url = f"{ESPN_BASE}/teams/{team_id}/schedule"
    data = _get(url, params={"season": season})
    if not data:
        return []

    games = []
    events = data.get('events', [])

    for event in events:
        try:
            # Only completed games
            status = event.get('competitions', [{}])[0].get('status', {})
            status_type = status.get('type', {}).get('name', '')
            is_completed = status.get('type', {}).get('completed', False)
            if not is_completed and status_type not in ('STATUS_FINAL', 'STATUS_FINAL_OT'):
                continue

            competition = event['competitions'][0]
            competitors = competition.get('competitors', [])
            if len(competitors) < 2:
                continue

            game_id = str(event.get('id', ''))
            game_date = event.get('date', '')[:10]  # YYYY-MM-DD

            # Determine home/away
            home_comp = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_comp = next((c for c in competitors if c.get('homeAway') == 'away'), None)

            if not home_comp or not away_comp:
                continue

            home_team_id = str(home_comp['team']['id'])
            away_team_id = str(away_comp['team']['id'])

            try:
                # Score can be plain int/str OR {"value": N, "displayValue": "N"}
                raw_home = home_comp.get('score', 0)
                raw_away = away_comp.get('score', 0)
                if isinstance(raw_home, dict):
                    home_score = int(float(raw_home.get('value', raw_home.get('displayValue', 0))))
                else:
                    home_score = int(float(raw_home))
                if isinstance(raw_away, dict):
                    away_score = int(float(raw_away.get('value', raw_away.get('displayValue', 0))))
                else:
                    away_score = int(float(raw_away))
            except (ValueError, TypeError):
                continue

            if home_score == 0 and away_score == 0:
                continue  # Skip games with no scores

            neutral_site = 1 if competition.get('neutralSite', False) else 0

            games.append({
                'game_id': game_id,
                'date': game_date,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_score': home_score,
                'away_score': away_score,
                'neutral_site': neutral_site,
                'season': season,
            })

        except Exception as e:
            log.debug(f"Error parsing event: {e}")
            continue

    return games


def load_season_games(season: int) -> List[Dict]:
    """
    Load all unique completed games for a season by iterating all team schedules.
    Deduplicates by game_id. Stores to SQLite.
    Returns list of new game dicts loaded.
    """
    init_db()
    existing_ids = get_loaded_game_ids()

    log.info(f"Loading season {season}. Existing games in DB: {len(existing_ids)}")

    teams = get_all_teams()
    if not teams:
        log.error("No teams loaded, aborting")
        return []

    seen_game_ids = set()
    all_games = []

    for i, team_id in enumerate(teams.keys()):
        if i % 50 == 0:
            log.info(f"  Processing team {i+1}/{len(teams)}...")

        time.sleep(REQUEST_DELAY)
        games = get_team_schedule(team_id, season)

        for g in games:
            gid = g['game_id']
            if gid not in seen_game_ids and gid not in existing_ids:
                seen_game_ids.add(gid)
                all_games.append(g)

    inserted = insert_games(all_games)
    log.info(f"Season {season}: {len(all_games)} new games found, {inserted} inserted to DB")
    return all_games


def load_all_seasons(seasons: List[int] = None) -> Dict[int, int]:
    """
    Load all seasons, skipping already-loaded game_ids.
    Returns: {season: num_games_loaded}
    """
    if seasons is None:
        seasons = NCAAB_SEASONS

    init_db()
    results = {}

    for season in seasons:
        log.info(f"\n{'='*50}")
        log.info(f"Loading season {season}...")
        games = load_season_games(season)
        results[season] = len(games)
        log.info(f"Season {season} complete: {len(games)} new games")

    return results


def get_all_games_df():
    """Load all games from DB into a DataFrame."""
    import pandas as pd
    conn = get_db_conn()
    df = pd.read_sql_query(
        "SELECT * FROM ncaab_games ORDER BY date ASC",
        conn
    )
    conn.close()
    return df


def get_season_game_count(season: int) -> int:
    """Return count of games in DB for a season."""
    conn = get_db_conn()
    count = conn.execute(
        "SELECT COUNT(*) FROM ncaab_games WHERE season = ?", (season,)
    ).fetchone()[0]
    conn.close()
    return count


class NCAABLoader:
    """
    Convenience class wrapper around the module-level NCAAB loader functions.

    Usage:
        loader = NCAABLoader()
        loader.load_all_seasons([2022, 2023, 2024, 2025])
        loader.load_season(2026)
    """

    def load_all_seasons(self, seasons=None):
        """Load multiple seasons. Delegates to module-level load_all_seasons()."""
        return load_all_seasons(seasons)

    def load_season(self, season: int):
        """Load a single season. Delegates to module-level load_season_games()."""
        init_db()
        return load_season_games(season)

    def game_count(self, season: int) -> int:
        """Return number of games in DB for a given season."""
        return get_season_game_count(season)

    def get_dataframe(self):
        """Return all NCAAB games from DB as a DataFrame."""
        return get_all_games_df()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NCAAB ESPN Data Loader")
    parser.add_argument("--season", type=int, default=2026, help="Season to load")
    parser.add_argument("--all", action="store_true", help="Load all seasons")
    args = parser.parse_args()

    if args.all:
        results = load_all_seasons()
        print(f"\nLoaded seasons: {results}")
    else:
        games = load_season_games(args.season)
        print(f"\nLoaded {len(games)} new games for season {args.season}")
        total = get_season_game_count(args.season)
        print(f"Total games in DB for season {args.season}: {total}")

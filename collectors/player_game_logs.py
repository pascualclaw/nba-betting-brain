"""
player_game_logs.py — Pull per-player game logs using nba_api.

Uses:
  - nba_api.stats.endpoints.playergamelog: per-player game-by-game stats
  - nba_api.stats.endpoints.commonteamroster: team roster with player IDs
  - nba_api.stats.endpoints.leaguedashplayerstats: season avg for minutes filter

Stores last 15 games per player to SQLite: data/nba_betting.db
Table: player_game_logs
"""

import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import sys
_REPO_ROOT = str(Path(__file__).parent.parent)
_COLLECTORS_DIR = str(Path(__file__).parent)
if _COLLECTORS_DIR in sys.path:
    sys.path.remove(_COLLECTORS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nba_api.stats.endpoints import (
    playergamelog,
    commonteamroster,
    leaguedashplayerstats,
)
from nba_api.stats.static import players as nba_players, teams as nba_teams

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "nba_betting.db"
CURRENT_SEASON = "2025-26"
NBA_API_DELAY = 0.6

# Team abbreviation → full name for nba_api
TEAM_ABBR_TO_NBA_ID = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}

# Some ESPN/DB abbreviations that differ from nba_api
ABBR_MAP = {
    "GS": "GSW", "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "WSH": "WAS", "UTH": "UTA", "PHO": "PHX",
}


def _normalize_abbr(abbr: str) -> str:
    return ABBR_MAP.get(abbr.upper(), abbr.upper())


def ensure_player_logs_table(conn: sqlite3.Connection):
    """Create player_game_logs table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            player_name TEXT,
            team TEXT,
            season TEXT,
            date TEXT,
            matchup TEXT,
            home_away TEXT,
            wl TEXT,
            min REAL,
            pts REAL,
            reb REAL,
            ast REAL,
            stl REAL,
            blk REAL,
            tov REAL,
            fgm REAL,
            fga REAL,
            fg_pct REAL,
            fg3m REAL,
            fg3a REAL,
            fg3_pct REAL,
            ftm REAL,
            fta REAL,
            ft_pct REAL,
            plus_minus REAL,
            fetched_at TEXT,
            UNIQUE(player_id, date)
        )
    """)
    conn.commit()


def get_team_roster(team_abbr: str) -> List[Dict]:
    """
    Pull current roster using nba_api commonteamroster.
    Returns list of {player_id, name, position, jersey}.
    """
    team_abbr = _normalize_abbr(team_abbr)
    nba_team_id = ABBR_TO_NBA_ID = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}
    team_id = nba_team_id.get(team_abbr)
    if not team_id:
        logger.error(f"Unknown team: {team_abbr}")
        return []

    try:
        time.sleep(NBA_API_DELAY)
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=CURRENT_SEASON, timeout=20)
        df = roster.get_data_frames()[0]
        players = []
        for _, row in df.iterrows():
            players.append({
                "player_id": str(row["PLAYER_ID"]),
                "name": row["PLAYER"],
                "position": row.get("POSITION", ""),
                "jersey": str(row.get("NUM", "")),
                "team": team_abbr,
            })
        return players
    except Exception as e:
        logger.error(f"Error getting roster for {team_abbr}: {e}")
        return []


def get_team_players_by_minutes(team_abbr: str, top_n: int = 9) -> List[Dict]:
    """
    Get team players sorted by average minutes played (top_n starters/rotation).
    Uses LeagueDashPlayerStats for season avg minutes filter.
    """
    team_abbr = _normalize_abbr(team_abbr)
    nba_id_map = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}
    team_id = nba_id_map.get(team_abbr)
    if not team_id:
        return []

    try:
        time.sleep(NBA_API_DELAY)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=CURRENT_SEASON,
            team_id_nullable=team_id,
            per_mode_detailed="PerGame",
            timeout=25,
        )
        df = stats.get_data_frames()[0]
        df = df[df["GP"] >= 5]  # at least 5 games played
        df = df.sort_values("MIN", ascending=False).head(top_n)
        
        players = []
        for _, row in df.iterrows():
            players.append({
                "player_id": str(row["PLAYER_ID"]),
                "name": row["PLAYER_NAME"],
                "avg_min": float(row["MIN"]),
                "avg_pts": float(row["PTS"]),
                "avg_reb": float(row["REB"]),
                "avg_ast": float(row["AST"]),
                "gp": int(row["GP"]),
                "team": team_abbr,
            })
        return players
    except Exception as e:
        logger.error(f"Error getting minutes-sorted players for {team_abbr}: {e}")
        return []


def get_player_game_logs(player_id: str, n_games: int = 15) -> List[Dict]:
    """
    Pull last n_games game logs for a player using nba_api.
    Returns list of dicts with per-game stats.
    """
    try:
        time.sleep(NBA_API_DELAY)
        gl = playergamelog.PlayerGameLog(
            player_id=int(player_id),
            season=CURRENT_SEASON,
            timeout=25,
        )
        df = gl.get_data_frames()[0]
        if df.empty:
            return []
        
        df = df.head(n_games)
        logs = []
        for _, row in df.iterrows():
            matchup = row.get("MATCHUP", "")
            home_away = "away" if "@" in matchup else "home"
            logs.append({
                "player_id": str(player_id),
                "date": str(row["GAME_DATE"]),
                "matchup": matchup,
                "home_away": home_away,
                "wl": str(row.get("WL", "")),
                "min": float(row.get("MIN", 0)),
                "pts": float(row.get("PTS", 0)),
                "reb": float(row.get("REB", 0)),
                "ast": float(row.get("AST", 0)),
                "stl": float(row.get("STL", 0)),
                "blk": float(row.get("BLK", 0)),
                "tov": float(row.get("TOV", 0)),
                "fgm": float(row.get("FGM", 0)),
                "fga": float(row.get("FGA", 0)),
                "fg_pct": float(row.get("FG_PCT", 0)),
                "fg3m": float(row.get("FG3M", 0)),
                "fg3a": float(row.get("FG3A", 0)),
                "fg3_pct": float(row.get("FG3_PCT", 0)),
                "ftm": float(row.get("FTM", 0)),
                "fta": float(row.get("FTA", 0)),
                "ft_pct": float(row.get("FT_PCT", 0)),
                "plus_minus": float(row.get("PLUS_MINUS", 0)),
            })
        return logs
    except Exception as e:
        logger.error(f"Error getting game logs for player {player_id}: {e}")
        return []


def save_player_logs(logs: List[Dict], player_name: str, team: str,
                     season: str, conn: sqlite3.Connection):
    """Upsert player game logs to SQLite."""
    fetched_at = datetime.now().isoformat()
    for log in logs:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO player_game_logs
                (player_id, player_name, team, season, date, matchup, home_away, wl,
                 min, pts, reb, ast, stl, blk, tov, fgm, fga, fg_pct,
                 fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log["player_id"], player_name, team.upper(), season,
                log["date"], log.get("matchup", ""), log.get("home_away", ""), log.get("wl", ""),
                log["min"], log["pts"], log["reb"], log["ast"],
                log["stl"], log["blk"], log["tov"],
                log["fgm"], log["fga"], log["fg_pct"],
                log["fg3m"], log["fg3a"], log["fg3_pct"],
                log["ftm"], log["fta"], log["ft_pct"],
                log["plus_minus"], fetched_at
            ))
        except Exception as e:
            logger.warning(f"Error saving log for {player_name} on {log.get('date')}: {e}")
    conn.commit()


def collect_team_player_logs(team_abbr: str, n_games: int = 15,
                              top_n_players: int = 9, db_path: Path = None) -> int:
    """
    Main function: for a given team, pull top-minutes players' game logs → DB.
    Returns count of players successfully fetched.
    """
    if db_path is None:
        db_path = DB_PATH

    logger.info(f"Collecting player logs for {team_abbr}...")
    players = get_team_players_by_minutes(team_abbr, top_n=top_n_players)
    if not players:
        logger.warning(f"No players found for {team_abbr}, falling back to roster")
        players = get_team_roster(team_abbr)[:top_n_players]
    if not players:
        return 0

    conn = sqlite3.connect(db_path)
    ensure_player_logs_table(conn)
    success = 0

    for player in players:
        pid = player["player_id"]
        name = player["name"]
        logs = get_player_game_logs(pid, n_games)

        if logs:
            save_player_logs(logs, name, team_abbr.upper(), CURRENT_SEASON, conn)
            logger.info(f"  {name}: {len(logs)} games saved (avg {player.get('avg_min', '?'):.1f} min/g)")
            success += 1
        else:
            logger.warning(f"  {name}: no logs (player_id={pid})")

    conn.close()
    logger.info(f"Done {team_abbr}: {success}/{len(players)} players")
    return success


def get_player_logs_from_db(team: str, n_games: int = 15,
                             db_path: Path = None) -> Dict[str, List[Dict]]:
    """
    Retrieve player logs from DB for a team.
    Returns dict: {player_name: [list of game log dicts]}
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    ensure_player_logs_table(conn)

    cursor = conn.execute("""
        SELECT player_id, player_name, date, matchup, home_away, wl,
               min, pts, reb, ast, stl, blk, tov, fgm, fga, fg_pct,
               fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus
        FROM player_game_logs
        WHERE team = ?
        ORDER BY player_name, date DESC
    """, (team.upper(),))

    rows = cursor.fetchall()
    conn.close()

    cols = ["player_id", "player_name", "date", "matchup", "home_away", "wl",
            "min", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg_pct",
            "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct", "plus_minus"]

    result = {}
    for row in rows:
        d = dict(zip(cols, row))
        name = d["player_name"]
        if name not in result:
            result[name] = []
        if len(result[name]) < n_games:
            result[name].append(d)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys as _sys
    teams = _sys.argv[1:] if len(_sys.argv) > 1 else ["LAL", "BOS"]
    for t in teams:
        collect_team_player_logs(t)

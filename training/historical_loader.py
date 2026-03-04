"""
Historical data loader — pulls 5 seasons of NBA game data from NBA Stats API.
Builds pre-game rolling stat snapshots for each game (no look-ahead bias).

Seasons: 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26
~6,000+ games with full box scores and rolling team stats.

Usage:
    python training/historical_loader.py --seasons 5
    python training/historical_loader.py --seasons 1  # current season only
"""
import sys
import time
import json
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats
from nba_api.stats.static import teams as nba_teams

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db import get_connection, upsert_game, upsert_team_snapshot
from config import SEASONS, NBA_API_DELAY

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Team ID → tricode mapping
_team_list = nba_teams.get_teams()
TEAM_ID_TO_TRICODE = {t["id"]: t["abbreviation"] for t in _team_list}
TRICODE_TO_ID = {t["abbreviation"]: t["id"] for t in _team_list}


def load_season_games(season: str) -> pd.DataFrame:
    """
    Load all games for a season from NBA Stats API.
    season format: "2024-25"
    Returns DataFrame with one row per game.
    """
    log.info(f"Loading game log for {season}...")
    gl = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        league_id="00",
    )
    df = gl.get_data_frames()[0]
    time.sleep(NBA_API_DELAY)

    # Keep relevant columns
    cols = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
            "MATCHUP", "WL", "PTS", "PLUS_MINUS"]
    df = df[cols].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["season"] = season
    return df


def build_game_records(df: pd.DataFrame) -> list:
    """
    Convert game log (two rows per game — one per team) into one record per game.
    Returns list of game dicts.
    """
    games = {}
    for _, row in df.iterrows():
        gid = row["GAME_ID"]
        if gid not in games:
            games[gid] = {"game_id": gid, "date": row["GAME_DATE"].strftime("%Y-%m-%d"),
                          "season": row["season"], "home": None, "away": None,
                          "home_score": 0, "away_score": 0}
        is_home = "@" not in row["MATCHUP"] or row["MATCHUP"].startswith(row["TEAM_ABBREVIATION"])
        # MATCHUP format: "PHX vs. SAC" (home) or "PHX @ SAC" (away)
        matchup = row["MATCHUP"]
        if " vs. " in matchup:
            is_home = True
        elif " @ " in matchup:
            is_home = False
        else:
            is_home = True  # fallback

        team = row["TEAM_ABBREVIATION"]
        pts = int(row["PTS"]) if pd.notna(row["PTS"]) else 0

        if is_home:
            games[gid]["home"] = team
            games[gid]["home_score"] = pts
        else:
            games[gid]["away"] = team
            games[gid]["away_score"] = pts

    records = []
    for gid, g in games.items():
        if g["home"] and g["away"]:
            g["total"] = g["home_score"] + g["away_score"]
            g["winner"] = g["home"] if g["home_score"] > g["away_score"] else g["away"]
            g["home_margin"] = g["home_score"] - g["away_score"]
            records.append(g)
    return records


def build_rolling_team_stats(games: list, window: int = 20) -> dict:
    """
    For each game, compute rolling team stats using only PRIOR games.
    This is the key anti-look-ahead mechanism.
    
    Returns: {game_id: {home_stats: {...}, away_stats: {...}}}
    """
    # Sort games chronologically
    sorted_games = sorted(games, key=lambda g: g["date"])
    
    # Track per-team rolling data
    team_history = {}  # team → list of (date, pts_for, pts_against, won)
    
    snapshots = {}
    
    for game in sorted_games:
        gid = game["game_id"]
        home, away = game["home"], game["away"]
        
        # Build snapshot BEFORE this game (using history up to but not including this game)
        home_snap = compute_rolling_stats(team_history.get(home, []), window)
        away_snap = compute_rolling_stats(team_history.get(away, []), window)
        
        snapshots[gid] = {
            "home_stats": home_snap,
            "away_stats": away_snap,
            "h2h_games": get_h2h_from_history(home, away, team_history, window=5),
        }
        
        # NOW update history with this game's result
        for team, pts_for, pts_against, won in [
            (home, game["home_score"], game["away_score"], game["home_score"] > game["away_score"]),
            (away, game["away_score"], game["home_score"], game["away_score"] > game["home_score"]),
        ]:
            if team not in team_history:
                team_history[team] = []
            team_history[team].append({
                "date": game["date"],
                "pts_for": pts_for,
                "pts_against": pts_against,
                "total": game["total"],
                "won": won,
                "opponent": away if team == home else home,
                "is_home": team == home,
            })
    
    return snapshots


def compute_rolling_stats(history: list, window: int = 20) -> dict:
    """Compute rolling averages from team's recent game history."""
    if not history:
        return {
            "pts_for_avg": 112.0, "pts_against_avg": 112.0,
            "total_avg": 224.0, "net_rating": 0.0,
            "pace_proxy": 100.0, "win_pct": 0.5,
            "games": 0, "last5_pts_for": 112.0, "last5_pts_against": 112.0,
        }
    
    recent = history[-window:]
    last5 = history[-5:]
    
    pts_for = [g["pts_for"] for g in recent]
    pts_against = [g["pts_against"] for g in recent]
    totals = [g["total"] for g in recent]
    
    return {
        "pts_for_avg": round(np.mean(pts_for), 1),
        "pts_against_avg": round(np.mean(pts_against), 1),
        "total_avg": round(np.mean(totals), 1),
        "net_rating": round(np.mean(pts_for) - np.mean(pts_against), 2),
        "pace_proxy": round(np.mean(totals) / 2.24, 1),  # proxy for pace
        "win_pct": round(sum(g["won"] for g in recent) / len(recent), 3),
        "games": len(history),
        "last5_pts_for": round(np.mean([g["pts_for"] for g in last5]), 1) if last5 else 112.0,
        "last5_pts_against": round(np.mean([g["pts_against"] for g in last5]), 1) if last5 else 112.0,
    }


def get_h2h_from_history(home: str, away: str, team_history: dict, window: int = 5) -> dict:
    """Get H2H stats between two specific teams from history."""
    home_hist = team_history.get(home, [])
    h2h_games = [g for g in home_hist if g["opponent"] == away][-window:]
    
    if not h2h_games:
        return {"h2h_total_avg": 224.0, "h2h_games": 0, "h2h_over220_rate": 0.5}
    
    totals = [g["total"] for g in h2h_games]
    return {
        "h2h_total_avg": round(np.mean(totals), 1),
        "h2h_games": len(h2h_games),
        "h2h_over220_rate": round(sum(1 for t in totals if t > 220) / len(totals), 2),
    }


def build_features(game: dict, snapshot: dict) -> dict:
    """Convert a game + snapshot into ML feature vector."""
    h = snapshot["home_stats"]
    a = snapshot["away_stats"]
    h2h = snapshot["h2h_games"]
    
    # Mathematical total projection
    league_avg = 113.0
    h_proj = (h["pts_for_avg"] * a["pts_against_avg"] / league_avg)
    a_proj = (a["pts_for_avg"] * h["pts_against_avg"] / league_avg)
    math_total = h_proj + a_proj
    
    return {
        # Team quality features
        "home_pts_for": h["pts_for_avg"],
        "home_pts_against": h["pts_against_avg"],
        "home_net_rating": h["net_rating"],
        "home_win_pct": h["win_pct"],
        "home_last5_pts_for": h["last5_pts_for"],
        "home_games_played": h["games"],
        "away_pts_for": a["pts_for_avg"],
        "away_pts_against": a["pts_against_avg"],
        "away_net_rating": a["net_rating"],
        "away_win_pct": a["win_pct"],
        "away_last5_pts_for": a["last5_pts_for"],
        "away_games_played": a["games"],
        # Matchup features
        "math_total_projection": round(math_total, 1),
        "pace_proxy_avg": round((h["pace_proxy"] + a["pace_proxy"]) / 2, 1),
        "combined_pts_avg": round(h["pts_for_avg"] + a["pts_for_avg"], 1),
        "net_rating_diff": round(h["net_rating"] - a["net_rating"], 2),
        # H2H features
        "h2h_total_avg": h2h["h2h_total_avg"],
        "h2h_games": h2h["h2h_games"],
        "h2h_over220_rate": h2h["h2h_over220_rate"],
        # Target
        "actual_total": game["total"],
        "actual_home_margin": game.get("home_margin", 0),
        "home_won": 1 if game["winner"] == game["home"] else 0,
    }


def load_and_store_seasons(seasons: list, conn: sqlite3.Connection):
    """Full pipeline: load seasons → build snapshots → store to DB."""
    all_games = []
    
    for season in seasons:
        log.info(f"\n{'='*50}")
        log.info(f"Processing season: {season}")
        
        # Check if already loaded
        cursor = conn.execute("SELECT COUNT(*) FROM games WHERE season=?", (season,))
        count = cursor.fetchone()[0]
        if count > 500:
            log.info(f"  Season {season} already loaded ({count} games). Skipping.")
            # Load from DB instead
            rows = conn.execute("SELECT * FROM games WHERE season=?", (season,)).fetchall()
            for row in rows:
                all_games.append(dict(zip([d[0] for d in conn.execute("SELECT * FROM games LIMIT 0").description], row)))
            continue
        
        try:
            df = load_season_games(season)
            records = build_game_records(df)
            log.info(f"  Loaded {len(records)} games")
            
            for game in records:
                upsert_game(conn, game)
            conn.commit()
            all_games.extend(records)
            log.info(f"  Saved to DB ✅")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            log.error(f"  Failed to load {season}: {e}")
            continue
    
    return all_games


def build_and_store_snapshots(all_games: list, conn: sqlite3.Connection):
    """Build rolling pre-game snapshots for all games and store to DB."""
    log.info(f"\nBuilding rolling snapshots for {len(all_games)} games...")
    snapshots = build_rolling_team_stats(all_games, window=20)
    
    stored = 0
    for game in all_games:
        gid = game["game_id"]
        if gid not in snapshots:
            continue
        snap = snapshots[gid]
        
        # Store home team snapshot
        upsert_team_snapshot(conn, {
            "game_id": gid,
            "date": game["date"],
            "team": game["home"],
            "is_home": 1,
            **snap["home_stats"],
            **{f"h2h_{k}": v for k, v in snap["h2h_games"].items()},
        })
        # Store away team snapshot
        upsert_team_snapshot(conn, {
            "game_id": gid,
            "date": game["date"],
            "team": game["away"],
            "is_home": 0,
            **snap["away_stats"],
            **{f"h2h_{k}": v for k, v in snap["h2h_games"].items()},
        })
        stored += 1
    
    conn.commit()
    log.info(f"Stored {stored} game snapshots ✅")
    return snapshots


def build_feature_dataset(all_games: list, snapshots: dict) -> pd.DataFrame:
    """Build the full ML feature dataset from all games and snapshots."""
    rows = []
    for game in all_games:
        gid = game["game_id"]
        if gid not in snapshots:
            continue
        # Skip games with insufficient history
        snap = snapshots[gid]
        if snap["home_stats"]["games"] < 5 or snap["away_stats"]["games"] < 5:
            continue
        features = build_features(game, snap)
        features["game_id"] = gid
        features["date"] = game["date"]
        features["season"] = game.get("season", "")
        features["home"] = game["home"]
        features["away"] = game["away"]
        rows.append(features)
    
    df = pd.DataFrame(rows)
    log.info(f"Feature dataset: {len(df)} games, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load historical NBA data")
    parser.add_argument("--seasons", type=int, default=3, help="Number of past seasons to load")
    args = parser.parse_args()

    from config import ALL_SEASONS, DB_PATH
    seasons_to_load = ALL_SEASONS[-(args.seasons):]
    log.info(f"Loading seasons: {seasons_to_load}")

    conn = get_connection()
    all_games = load_and_store_seasons(seasons_to_load, conn)
    snapshots = build_and_store_snapshots(all_games, conn)
    df = build_feature_dataset(all_games, snapshots)
    
    output_path = Path(__file__).parent.parent / "data" / "training_features.parquet"
    df.to_parquet(output_path, index=False)
    log.info(f"\nSaved feature dataset: {output_path}")
    log.info(f"Ready for model training. Run: python training/train.py")
    conn.close()

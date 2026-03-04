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
import warnings
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
    team_last_game_date = {}  # team → last game date (for rest days calculation)
    
    snapshots = {}
    
    for game in sorted_games:
        gid = game["game_id"]
        home, away = game["home"], game["away"]
        game_date = game["date"]
        
        # Build snapshot BEFORE this game (using history up to but not including this game)
        home_snap = compute_rolling_stats(team_history.get(home, []), window)
        away_snap = compute_rolling_stats(team_history.get(away, []), window)
        
        # Compute rest days (days since last game)
        home_rest = compute_rest_days(home, game_date, team_last_game_date)
        away_rest = compute_rest_days(away, game_date, team_last_game_date)
        
        snapshots[gid] = {
            "home_stats": home_snap,
            "away_stats": away_snap,
            "h2h_games": get_h2h_from_history(home, away, team_history, window=5),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "home_b2b": 1 if home_rest == 0 else 0,
            "away_b2b": 1 if away_rest == 0 else 0,
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
            team_last_game_date[team] = game_date
    
    return snapshots


def compute_rest_days(team: str, game_date: str, last_game_dates: dict) -> int:
    """
    Compute number of rest days before this game (0 = back-to-back).
    game_date and last_game_dates values are strings like '2024-01-15'.
    Returns -1 if no prior game (season opener).
    """
    last_date_str = last_game_dates.get(team)
    if last_date_str is None:
        return 7  # No prior game — treat as well-rested
    try:
        last_dt = datetime.strptime(last_date_str, "%Y-%m-%d")
        curr_dt = datetime.strptime(game_date, "%Y-%m-%d")
        delta = (curr_dt - last_dt).days - 1  # days between games (0 = consecutive nights)
        return max(0, delta)
    except Exception:
        return 3  # fallback


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
    
    # Rest days & back-to-back (0 default for historical games where not computed)
    home_rest = snapshot.get("home_rest_days", 3)
    away_rest = snapshot.get("away_rest_days", 3)
    home_b2b = snapshot.get("home_b2b", 0)
    away_b2b = snapshot.get("away_b2b", 0)
    
    # Injury impact (0 for historical — ESPN doesn't have reliable historical injury data)
    home_injury_impact = snapshot.get("home_injury_impact", 0.0)
    away_injury_impact = snapshot.get("away_injury_impact", 0.0)
    
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
        # Rest / fatigue features
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "home_b2b": home_b2b,
        "away_b2b": away_b2b,
        # Injury impact features (0 for most historical, real values for recent/live)
        "home_injury_impact": home_injury_impact,
        "away_injury_impact": away_injury_impact,
        # Target
        "actual_total": game["total"],
        "actual_home_margin": game.get("home_margin", 0),
        "home_won": 1 if game["winner"] == game["home"] else 0,
    }


def add_home_away_split_features(features: dict, game: dict,
                                  home_away_splits: dict) -> dict:
    """
    Enrich a feature dict with home/away split statistics.

    Split features added:
      home_home_ortg   — how good home team scores at home specifically
      home_away_ortg   — how good home team scores on the road
      away_home_ortg   — how good away team scores at home (their best)
      away_away_ortg   — how good away team scores on the road (current game context)
      + corresponding DRTG and win% variants

    These are computed from the full season history in home_away_splits
    (populated by reports/home_away_splits.py).
    """
    home_team = game.get("home", "")
    away_team = game.get("away", "")

    # Sensible defaults based on rolling averages already in features
    defaults = {
        "home_home_ortg": features.get("home_pts_for", 113.0),
        "home_away_ortg": features.get("home_pts_for", 113.0) - 2.0,
        "away_home_ortg": features.get("away_pts_for", 113.0) + 2.0,
        "away_away_ortg": features.get("away_pts_for", 113.0),
        "home_home_drtg": features.get("home_pts_against", 113.0),
        "home_away_drtg": features.get("home_pts_against", 113.0) + 2.0,
        "away_home_drtg": features.get("away_pts_against", 113.0) - 2.0,
        "away_away_drtg": features.get("away_pts_against", 113.0),
        "home_home_win_pct": min(features.get("home_win_pct", 0.5) + 0.05, 1.0),
        "home_away_win_pct": max(features.get("home_win_pct", 0.5) - 0.05, 0.0),
        "away_home_win_pct": min(features.get("away_win_pct", 0.5) + 0.05, 1.0),
        "away_away_win_pct": max(features.get("away_win_pct", 0.5) - 0.05, 0.0),
    }

    if home_away_splits and home_team and away_team:
        h_splits = home_away_splits.get(home_team, {})
        a_splits = home_away_splits.get(away_team, {})
        if h_splits and a_splits:
            split_feats = {
                "home_home_ortg": h_splits.get("home_ortg", defaults["home_home_ortg"]),
                "home_away_ortg": h_splits.get("away_ortg", defaults["home_away_ortg"]),
                "away_home_ortg": a_splits.get("home_ortg", defaults["away_home_ortg"]),
                "away_away_ortg": a_splits.get("away_ortg", defaults["away_away_ortg"]),
                "home_home_drtg": h_splits.get("home_drtg", defaults["home_home_drtg"]),
                "home_away_drtg": h_splits.get("away_drtg", defaults["home_away_drtg"]),
                "away_home_drtg": a_splits.get("home_drtg", defaults["away_home_drtg"]),
                "away_away_drtg": a_splits.get("away_drtg", defaults["away_away_drtg"]),
                "home_home_win_pct": h_splits.get("home_win_pct", defaults["home_home_win_pct"]),
                "home_away_win_pct": h_splits.get("away_win_pct", defaults["home_away_win_pct"]),
                "away_home_win_pct": a_splits.get("home_win_pct", defaults["away_home_win_pct"]),
                "away_away_win_pct": a_splits.get("away_win_pct", defaults["away_away_win_pct"]),
            }
            features.update(split_feats)
            return features

    features.update(defaults)
    return features


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
            **snap["h2h_games"],
        })
        # Store away team snapshot
        upsert_team_snapshot(conn, {
            "game_id": gid,
            "date": game["date"],
            "team": game["away"],
            "is_home": 0,
            **snap["away_stats"],
            **snap["h2h_games"],
        })
        stored += 1
    
    conn.commit()
    log.info(f"Stored {stored} game snapshots ✅")
    return snapshots


def build_feature_dataset(all_games: list, snapshots: dict,
                           home_away_splits: dict = None) -> pd.DataFrame:
    """Build the full ML feature dataset from all games and snapshots.

    Args:
        all_games: List of game records.
        snapshots: Pre-game rolling snapshots (no look-ahead).
        home_away_splits: Optional home/away split stats from
                          reports/home_away_splits.py.  When provided,
                          adds home_home_ortg, away_away_ortg etc. columns.
    """
    # Lazy-load splits from disk if not provided
    if home_away_splits is None:
        try:
            from reports.home_away_splits import load_splits
            splits_path = Path(__file__).parent.parent / "data" / "teams_home_away_splits.json"
            if splits_path.exists():
                home_away_splits = load_splits(splits_path)
                log.info(f"Loaded home/away splits for {len(home_away_splits)} teams")
        except Exception as e:
            log.debug(f"Could not load home/away splits: {e}")

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
        # Enrich with home/away splits
        features = add_home_away_split_features(features, game, home_away_splits or {})
        features["game_id"] = gid
        features["date"] = game["date"]
        features["season"] = game.get("season", "")
        features["home"] = game["home"]
        features["away"] = game["away"]
        rows.append(features)

    df = pd.DataFrame(rows)
    log.info(f"Feature dataset: {len(df)} games, {len(df.columns)} columns")
    return df


def join_odds_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to join saved odds data to the feature dataset.

    For each game row, looks for a matching odds JSON file in
    data/odds/{date}_{home}_{away}_odds.json.

    Adds columns: open_total_line, close_total_line, open_spread
    If no odds file is found, leaves as NaN (HistGradientBoosting handles NaN natively).

    Returns:
        DataFrame with three new odds columns appended.
    """
    odds_dir = Path(__file__).parent.parent / "data" / "odds"

    if not odds_dir.exists():
        log.info("No odds directory found — odds features will be NaN.")
        df["open_total_line"] = np.nan
        df["close_total_line"] = np.nan
        df["open_spread"] = np.nan
        return df

    # Build lookup: {date_HOME_AWAY: path}  and  {date_AWAY_HOME: path}
    odds_lookup: dict[str, Path] = {}
    for p in odds_dir.glob("*_odds.json"):
        # filename: YYYY-MM-DD_HOME_AWAY_odds.json
        parts = p.stem.split("_")
        if len(parts) >= 5:
            date_part = parts[0]   # YYYY-MM-DD  (but stem has no hyphens split)
            # Actually stem = "2026-03-04_PHX_SAC_odds"
            # Split on _ gives: ['2026-03-04', 'PHX', 'SAC', 'odds']
            # But date has hyphens, so split("_") gives 4 elements
            date_part = "_".join(parts[:3]) if "-" not in parts[0] else parts[0]
            # Re-derive: just use filename pattern robustly
            name = p.stem  # e.g. "2026-03-04_PHX_SAC_odds"
            # Strip trailing _odds
            name_no_odds = name[: name.rfind("_odds")] if "_odds" in name else name
            odds_lookup[name_no_odds] = p

    def _lookup_odds(row: pd.Series) -> pd.Series:
        date_s = str(row.get("date", ""))[:10]  # YYYY-MM-DD
        home = str(row.get("home", "")).upper()
        away = str(row.get("away", "")).upper()
        key1 = f"{date_s}_{home}_{away}"
        key2 = f"{date_s}_{away}_{home}"
        path = odds_lookup.get(key1) or odds_lookup.get(key2)
        if path and path.exists():
            try:
                data = json.loads(path.read_text())
                total = data.get("open_total_line")
                spread = data.get("open_spread")
                return pd.Series({
                    "open_total_line": float(total) if total is not None else np.nan,
                    "close_total_line": float(total) if total is not None else np.nan,  # same unless we have snapshots
                    "open_spread": float(spread) if spread is not None else np.nan,
                })
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        return pd.Series({"open_total_line": np.nan, "close_total_line": np.nan, "open_spread": np.nan})

    if len(df) == 0:
        df["open_total_line"] = np.nan
        df["close_total_line"] = np.nan
        df["open_spread"] = np.nan
        return df

    log.info("Joining odds data to features...")
    odds_cols = df.apply(_lookup_odds, axis=1)
    n_matched = odds_cols["open_total_line"].notna().sum()
    log.info(f"Odds joined: {n_matched}/{len(df)} games have a betting line.")

    df["open_total_line"] = odds_cols["open_total_line"]
    df["close_total_line"] = odds_cols["close_total_line"]
    df["open_spread"] = odds_cols["open_spread"]
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
    df = join_odds_to_features(df)

    output_path = Path(__file__).parent.parent / "data" / "training_features.csv"
    df.to_csv(output_path, index=False)
    log.info(f"\nSaved feature dataset: {output_path} ({len(df)} games)")
    log.info(f"Ready for model training. Run: python training/train.py")
    conn.close()

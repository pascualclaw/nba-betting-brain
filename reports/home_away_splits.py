"""
Home/Away Stat Splits

Computes per-team home vs away performance splits from game history.

Metrics per team:
  - Home/Away ORTG (offensive rating proxy = pts scored per game)
  - Home/Away DRTG (defensive rating proxy = pts allowed per game)
  - Home/Away win%
  - Home/Away avg total

Output: data/teams_home_away_splits.json

Usage:
    python reports/home_away_splits.py
    python reports/home_away_splits.py --seasons 2  # use last 2 seasons
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "teams_home_away_splits.json"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_games_from_db(seasons: Optional[int] = None) -> pd.DataFrame:
    """Load game records from the SQLite database."""
    try:
        from database.db import get_connection
        conn = get_connection()
        query = "SELECT * FROM games"
        params: tuple = ()
        if seasons:
            from config import ALL_SEASONS
            recent = ALL_SEASONS[-seasons:]
            placeholders = ",".join("?" * len(recent))
            query += f" WHERE season IN ({placeholders})"
            params = tuple(recent)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        log.info(f"Loaded {len(df)} games from DB")
        return df
    except Exception as e:
        log.warning(f"DB load failed: {e}. Trying CSV fallback.")
        return load_games_from_csv()


def load_games_from_csv() -> pd.DataFrame:
    """Fallback: load from training_features.csv if DB unavailable."""
    csv_path = DATA_DIR / "training_features.csv"
    if not csv_path.exists():
        log.warning("No training_features.csv found. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} rows from training_features.csv")
    return df


def load_games_from_nba_api(seasons: Optional[int] = None) -> pd.DataFrame:
    """
    Directly pull game logs from NBA Stats API if no local data.
    Returns a normalized DataFrame with columns: home, away, home_score, away_score, date, season.
    """
    try:
        import time
        from nba_api.stats.endpoints import leaguegamelog
        from config import ALL_SEASONS, NBA_API_DELAY

        seasons_list = ALL_SEASONS[-(seasons or 2):]
        rows = []
        for season in seasons_list:
            log.info(f"Fetching game log for {season}...")
            gl = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                league_id="00",
            )
            df = gl.get_data_frames()[0]
            time.sleep(NBA_API_DELAY)

            games: Dict[str, dict] = {}
            for _, row in df.iterrows():
                gid = row["GAME_ID"]
                if gid not in games:
                    games[gid] = {"game_id": gid, "date": str(row["GAME_DATE"]),
                                  "season": season, "home": None, "away": None,
                                  "home_score": 0, "away_score": 0}
                is_home = " vs. " in str(row["MATCHUP"])
                team = row["TEAM_ABBREVIATION"]
                pts = int(row["PTS"]) if pd.notna(row["PTS"]) else 0
                if is_home:
                    games[gid]["home"] = team
                    games[gid]["home_score"] = pts
                else:
                    games[gid]["away"] = team
                    games[gid]["away_score"] = pts

            for g in games.values():
                if g["home"] and g["away"]:
                    g["total"] = g["home_score"] + g["away_score"]
                    rows.append(g)

        return pd.DataFrame(rows)
    except Exception as e:
        log.error(f"NBA API pull failed: {e}")
        return pd.DataFrame()


# ── Core split computation ─────────────────────────────────────────────────────

def compute_team_splits(games_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Compute home/away splits for every team in the dataset.

    Expects DataFrame with columns: home, away, home_score, away_score.
    Returns dict keyed by team abbreviation.
    """
    if games_df.empty:
        log.warning("No games data — returning empty splits.")
        return {}

    # Normalize column names (handle training_features.csv layout)
    col_map = {
        "home_pts_for": "home_score",
        "away_pts_for": "away_score",
    }
    df = games_df.rename(columns=col_map).copy()

    required = {"home", "away", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        log.error(f"Missing required columns: {missing}")
        return {}

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0)
    df["total"] = df["home_score"] + df["away_score"]

    all_teams = set(df["home"].dropna()) | set(df["away"].dropna())
    splits: Dict[str, dict] = {}

    for team in sorted(all_teams):
        home_games = df[df["home"] == team]
        away_games = df[df["away"] == team]

        # Home stats
        h_ortg = float(home_games["home_score"].mean()) if len(home_games) > 0 else 113.0
        h_drtg = float(home_games["away_score"].mean()) if len(home_games) > 0 else 113.0
        h_total = float(home_games["total"].mean()) if len(home_games) > 0 else 226.0
        h_wins = int((home_games["home_score"] > home_games["away_score"]).sum())
        h_win_pct = h_wins / len(home_games) if len(home_games) > 0 else 0.5

        # Away stats
        a_ortg = float(away_games["away_score"].mean()) if len(away_games) > 0 else 113.0
        a_drtg = float(away_games["home_score"].mean()) if len(away_games) > 0 else 113.0
        a_total = float(away_games["total"].mean()) if len(away_games) > 0 else 226.0
        a_wins = int((away_games["away_score"] > away_games["home_score"]).sum())
        a_win_pct = a_wins / len(away_games) if len(away_games) > 0 else 0.5

        splits[team] = {
            "team": team,
            "home_games": len(home_games),
            "away_games": len(away_games),
            # Home splits
            "home_ortg": round(h_ortg, 1),
            "home_drtg": round(h_drtg, 1),
            "home_net_rating": round(h_ortg - h_drtg, 1),
            "home_win_pct": round(h_win_pct, 3),
            "home_avg_total": round(h_total, 1),
            # Away splits
            "away_ortg": round(a_ortg, 1),
            "away_drtg": round(a_drtg, 1),
            "away_net_rating": round(a_ortg - a_drtg, 1),
            "away_win_pct": round(a_win_pct, 3),
            "away_avg_total": round(a_total, 1),
            # Home advantage delta
            "home_away_ortg_delta": round(h_ortg - a_ortg, 1),
            "home_away_win_delta": round(h_win_pct - a_win_pct, 3),
        }

    log.info(f"Computed splits for {len(splits)} teams")
    return splits


def get_split_features_for_game(home_team: str, away_team: str,
                                 splits: Optional[Dict[str, dict]] = None) -> dict:
    """
    Return the 4 home/away split features needed for ML feature engineering.

    Features:
      home_home_ortg   — home team's ORTG when playing at home
      home_away_ortg   — home team's ORTG when playing away (how bad they are on road)
      away_home_ortg   — away team's ORTG when playing at home (their best)
      away_away_ortg   — away team's ORTG when playing away (road performance)

    If splits not provided, loads from data/teams_home_away_splits.json.
    """
    if splits is None:
        splits = load_splits()

    default = {
        "home_home_ortg": 113.0, "home_away_ortg": 111.0,
        "away_home_ortg": 113.0, "away_away_ortg": 111.0,
        "home_home_drtg": 113.0, "home_away_drtg": 115.0,
        "away_home_drtg": 113.0, "away_away_drtg": 115.0,
        "home_home_win_pct": 0.55, "home_away_win_pct": 0.45,
        "away_home_win_pct": 0.55, "away_away_win_pct": 0.45,
    }

    home_splits = splits.get(home_team)
    away_splits = splits.get(away_team)

    if not home_splits or not away_splits:
        log.debug(f"No splits for {home_team} or {away_team}. Using defaults.")
        return default

    return {
        # Home team's splits
        "home_home_ortg": home_splits["home_ortg"],
        "home_away_ortg": home_splits["away_ortg"],
        "home_home_drtg": home_splits["home_drtg"],
        "home_away_drtg": home_splits["away_drtg"],
        "home_home_win_pct": home_splits["home_win_pct"],
        "home_away_win_pct": home_splits["away_win_pct"],
        # Away team's splits
        "away_home_ortg": away_splits["home_ortg"],
        "away_away_ortg": away_splits["away_ortg"],
        "away_home_drtg": away_splits["home_drtg"],
        "away_away_drtg": away_splits["away_drtg"],
        "away_home_win_pct": away_splits["home_win_pct"],
        "away_away_win_pct": away_splits["away_win_pct"],
    }


# ── Persistence ────────────────────────────────────────────────────────────────

def save_splits(splits: Dict[str, dict], path: Path = OUTPUT_PATH) -> None:
    """Save splits JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    log.info(f"Saved splits to {path}")


def load_splits(path: Path = OUTPUT_PATH) -> Dict[str, dict]:
    """Load splits from disk. Returns empty dict if not found."""
    if not path.exists():
        log.warning(f"Splits file not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute home/away team stat splits")
    parser.add_argument("--seasons", type=int, default=3,
                        help="Number of recent seasons to use (default: 3)")
    parser.add_argument("--source", choices=["db", "csv", "api"], default="db",
                        help="Data source (default: db)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output JSON path")
    args = parser.parse_args()

    output_path = Path(args.output)

    # Load games
    if args.source == "db":
        df = load_games_from_db(args.seasons)
    elif args.source == "csv":
        df = load_games_from_csv()
    else:
        df = load_games_from_nba_api(args.seasons)

    if df.empty:
        # Try each source in order as fallback
        for loader in [load_games_from_db, load_games_from_csv]:
            df = loader()
            if not df.empty:
                break

    if df.empty:
        log.error("Could not load game data from any source. Exiting.")
        sys.exit(1)

    splits = compute_team_splits(df)
    save_splits(splits, output_path)

    # Pretty print a sample
    print(f"\n✅ Home/Away splits computed for {len(splits)} teams\n")
    sample_teams = ["PHX", "SAC", "GSW", "BOS", "LAL"]
    for team in sample_teams:
        if team in splits:
            s = splits[team]
            print(f"  {team}:")
            print(f"    Home: ORTG={s['home_ortg']} DRTG={s['home_drtg']} W%={s['home_win_pct']:.3f}")
            print(f"    Away: ORTG={s['away_ortg']} DRTG={s['away_drtg']} W%={s['away_win_pct']:.3f}")


if __name__ == "__main__":
    main()

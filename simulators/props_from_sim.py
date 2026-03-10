"""
simulators/props_from_sim.py — Player Props Scanner from Simulation
====================================================================
Uses player game log distributions to compute P(stat > line) for each prop.
Compares against DK lines from The Odds API.
Returns props where EV > 5% and sample size >= 10 games.
"""

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import requests

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "nba_betting.db"
logger = logging.getLogger(__name__)

N_SIMS = 500_000  # fewer sims for props — still statistically solid


def american_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def ev_pct(our_prob: float, odds: int) -> float:
    if odds > 0:
        decimal = odds / 100
    else:
        decimal = 100 / abs(odds)
    return (our_prob * decimal - (1 - our_prob)) * 100


def get_player_distributions(team: str, n_games: int = 15,
                              db_path: Path = None) -> Dict[str, Dict]:
    """
    Load player distributions from player_game_logs.
    Returns {player_name: {mean_pts, std_pts, mean_reb, std_reb, mean_ast, std_ast, n, last_min_pct}}
    """
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    try:
        # Get all game logs for team, up to n_games per player
        rows = conn.execute("""
            SELECT player_name, pts, reb, ast, min, date
            FROM player_game_logs
            WHERE team = ?
            ORDER BY player_name, date DESC
        """, (team.upper(),)).fetchall()

        # Group by player, take last n_games
        from collections import defaultdict
        player_logs = defaultdict(list)
        for name, pts, reb, ast, min_val, date in rows:
            if len(player_logs[name]) < n_games:
                player_logs[name].append({
                    "pts": float(pts or 0),
                    "reb": float(reb or 0),
                    "ast": float(ast or 0),
                    "min": float(min_val or 0),
                    "date": date,
                })

        # Get all game logs (for last-game minutes check)
        last_game_rows = conn.execute("""
            SELECT player_name, min
            FROM player_game_logs
            WHERE team = ? AND date = (
                SELECT MAX(date) FROM player_game_logs WHERE team = ?
            )
        """, (team.upper(), team.upper())).fetchall()
        last_min_map = {r[0]: r[1] for r in last_game_rows}

        distributions = {}
        for name, logs in player_logs.items():
            if len(logs) < 3:
                continue
            pts_arr = [g["pts"] for g in logs]
            reb_arr = [g["reb"] for g in logs]
            ast_arr = [g["ast"] for g in logs]
            min_arr = [g["min"] for g in logs]

            avg_min = np.mean(min_arr)
            last_min = last_min_map.get(name, avg_min)
            last_min_pct = (last_min / avg_min) if avg_min > 0 else 1.0

            distributions[name] = {
                "mean_pts": float(np.mean(pts_arr)),
                "std_pts": max(float(np.std(pts_arr)), 2.0),
                "mean_reb": float(np.mean(reb_arr)),
                "std_reb": max(float(np.std(reb_arr)), 1.0),
                "mean_ast": float(np.mean(ast_arr)),
                "std_ast": max(float(np.std(ast_arr)), 0.8),
                "mean_min": float(avg_min),
                "n_games": len(logs),
                "last_min_pct": float(last_min_pct),
                "high_risk": last_min_pct < 0.65,
            }
        return distributions
    finally:
        conn.close()


def fetch_dk_player_props(event_id: str, api_key: str,
                           markets: str = "player_points,player_rebounds,player_assists") -> List[Dict]:
    """
    Pull DraftKings player prop lines from The Odds API.
    Returns list of {player_name, market, line, over_odds, under_odds}
    """
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "markets": markets,
        "bookmakers": "draftkings",
        "oddsFormat": "american",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        remaining = r.headers.get("x-requests-remaining", "?")
        logger.info(f"Props API remaining: {remaining}")
        data = r.json()

        props = []
        for bk in data.get("bookmakers", []):
            if bk.get("key") != "draftkings":
                continue
            for market in bk.get("markets", []):
                market_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", outcome.get("name", ""))
                    side = outcome.get("name", "")  # "Over" or "Under"
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if point is not None and price is not None:
                        props.append({
                            "player_name": player_name,
                            "market": market_key,
                            "side": side,
                            "line": float(point),
                            "odds": int(price),
                        })
        return props
    except Exception as e:
        logger.warning(f"Props fetch failed for {event_id}: {e}")
        return []


def scan_props(team: str, opponent: str, event_id: str,
               api_key: str, db_path: Path = None,
               min_ev_pct: float = 5.0,
               min_sample: int = 10,
               n_sims: int = N_SIMS) -> List[Dict]:
    """
    Full props scan for a team's players vs DK lines.

    Args:
        team: team abbreviation (e.g. "BOS")
        opponent: opponent abbreviation
        event_id: The Odds API event ID for this game
        api_key: Odds API key
        db_path: path to nba_betting.db
        min_ev_pct: minimum EV% to include in output
        min_sample: minimum games in player log for reliable dist

    Returns: list of picks sorted by EV descending
    """
    dist_map = get_player_distributions(team, db_path=db_path)
    if not dist_map:
        logger.warning(f"No player distributions for {team}")
        return []

    # Fetch DK prop lines
    props_raw = fetch_dk_player_props(event_id, api_key)
    if not props_raw:
        logger.warning(f"No DK props available for event {event_id}")
        return []

    # Group by player + market: find over/under pairs
    from collections import defaultdict
    prop_lines = defaultdict(dict)
    for p in props_raw:
        key = (p["player_name"], p["market"])
        prop_lines[key][p["side"]] = (p["line"], p["odds"])

    # Map market key to stat name
    market_to_stat = {
        "player_points": ("pts", "mean_pts", "std_pts"),
        "player_rebounds": ("reb", "mean_reb", "std_reb"),
        "player_assists": ("ast", "mean_ast", "std_ast"),
    }

    picks = []
    rng = np.random.default_rng()

    for (player_name, market_key), sides in prop_lines.items():
        if "Over" not in sides:
            continue
        line, _ = sides["Over"]
        under_odds = sides.get("Under", (line, -110))[1]
        over_odds = sides["Over"][1]

        # Find player in distribution map (fuzzy match)
        player_dist = None
        for pname, pdist in dist_map.items():
            if player_name.lower() in pname.lower() or pname.lower() in player_name.lower():
                player_dist = pdist
                break

        if not player_dist:
            continue
        if player_dist["n_games"] < min_sample:
            continue

        stat_info = market_to_stat.get(market_key)
        if not stat_info:
            continue

        _, mean_key, std_key = stat_info
        mean = player_dist[mean_key]
        std = player_dist[std_key]

        # Simulate n_sims draws
        sim_vals = rng.normal(mean, std, n_sims)
        our_p_over = float(np.mean(sim_vals > line))
        our_p_under = 1.0 - our_p_over

        mkt_over_impl = american_to_implied_prob(over_odds)
        mkt_under_impl = american_to_implied_prob(under_odds)

        over_ev = ev_pct(our_p_over, over_odds)
        under_ev = ev_pct(our_p_under, under_odds)

        best_ev = max(over_ev, under_ev)
        if best_ev >= min_ev_pct:
            if over_ev >= under_ev:
                side = "OVER"
                our_p = our_p_over
                market_impl = mkt_over_impl
                best_odds = over_odds
                ev = over_ev
            else:
                side = "UNDER"
                our_p = our_p_under
                market_impl = mkt_under_impl
                best_odds = under_odds
                ev = under_ev

            edge_pct = (our_p - market_impl) * 100

            picks.append({
                "player": player_name,
                "team": team,
                "market": market_key.replace("player_", ""),
                "side": side,
                "line": line,
                "our_prob": our_p,
                "market_implied": market_impl,
                "edge_pct": edge_pct,
                "ev_pct": ev,
                "sim_mean": mean,
                "sim_std": std,
                "n_games": player_dist["n_games"],
                "high_risk": player_dist["high_risk"],
                "market_odds": best_odds,
            })

    picks.sort(key=lambda x: x["ev_pct"], reverse=True)
    return picks

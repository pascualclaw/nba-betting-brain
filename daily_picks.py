"""
daily_picks.py — Full pre-game pick generation pipeline.

Usage:
    python daily_picks.py                    # Tonight's slate
    python daily_picks.py --date 2026-03-05  # Specific date
    python daily_picks.py --game LAL DEN     # Single game

Output:
    - Ranked picks with EV%, Kelly bet size, and risk flags
    - Only recommends bets passing 3% EV threshold
    - Includes motivation flags, referee context, possessions projection

Architecture:
    ESPN API → tonight's games
    → ML model projection (15yr gradient boosting)
    → Possessions × efficiency projection (Four Factors)
    → EV gate (3% threshold)
    → Kelly sizing (quarter Kelly, max 5%)
    → Motivation flags (tank mode, revenge, B2B, playoff push)
    → Referee context (high/low foul crew → total adjustment)
    → Ranked output (highest EV first)
"""

import argparse
import json
import logging
import pickle
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

# ── Tricode normalization (ESPN → DB) ──────────────────────────────────────
# ESPN uses different tricodes than NBA CDN in some cases
ESPN_TO_DB = {
    "WSH": "WAS",   # Washington
    "GS": "GSW",    # Golden State
    "SA": "SAS",    # San Antonio
    "NO": "NOP",    # New Orleans
    "NY": "NYK",    # New York (rare)
    "UTH": "UTA",   # Utah alt
    "UTAH": "UTA",  # ESPN full abbreviation
    "NOH": "NOP",   # Old New Orleans
    "NJN": "BKN",   # Old Brooklyn
}

def normalize_tri(tri: str) -> str:
    """Normalize ESPN tricode to DB tricode."""
    return ESPN_TO_DB.get(tri, tri)

# ── Setup ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "latest.pkl"
SPREAD_MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "spread_latest.pkl"
DB_PATH = PROJECT_ROOT / "data" / "nba_betting.db"
DATA_DIR = PROJECT_ROOT / "data"


# ── ESPN Slate Fetcher ──────────────────────────────────────────────────────

def fetch_tonight_slate(game_date: Optional[str] = None) -> List[Dict]:
    """Fetch tonight's NBA games from ESPN API."""
    d = game_date or date.today().strftime("%Y%m%d")
    url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
    try:
        with urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
    except Exception as e:
        log.error(f"ESPN fetch failed: {e}")
        return []

    games = []
    for event in data.get("events", []):
        comps = event.get("competitions", [{}])[0]
        teams = comps.get("competitors", [])
        home = next((t for t in teams if t["homeAway"] == "home"), {})
        away = next((t for t in teams if t["homeAway"] == "away"), {})
        status = event.get("status", {}).get("type", {}).get("shortDetail", "")
        start_time = event.get("date", "")

        games.append({
            "home": home.get("team", {}).get("abbreviation", ""),
            "away": away.get("team", {}).get("abbreviation", ""),
            "home_name": home.get("team", {}).get("displayName", ""),
            "away_name": away.get("team", {}).get("displayName", ""),
            "status": status,
            "espn_id": event.get("id", ""),
            "start_time": start_time,
        })
    return games


# ── Odds Fetcher ───────────────────────────────────────────────────────────

def fetch_dk_lines() -> Dict[str, Dict]:
    """Fetch DraftKings lines from The Odds API. Returns {home_team: {spread, total, odds}}."""
    try:
        creds_path = Path.home() / ".openclaw" / "credentials" / "jarvis-github.env"
        api_key = None
        if creds_path.exists():
            for line in creds_path.read_text().splitlines():
                if line.startswith("ODDS_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()

        if not api_key:
            return {}

        url = (
            f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
            f"?apiKey={api_key}&regions=us&markets=spreads,totals&bookmakers=draftkings&oddsFormat=american"
        )
        with urlopen(url, timeout=10) as r:
            games = json.loads(r.read())

        lines = {}
        for game in games:
            home = game.get("home_team", "")
            bms = game.get("bookmakers", [])
            if not bms:
                continue
            markets = {m["key"]: m for m in bms[0].get("markets", [])}

            spread = None
            total = None
            if "spreads" in markets:
                for outcome in markets["spreads"]["outcomes"]:
                    if outcome["name"] == home:
                        spread = outcome.get("point")
                        break
            if "totals" in markets:
                for outcome in markets["totals"]["outcomes"]:
                    if outcome["name"] == "Over":
                        total = outcome.get("point")
                        break

            # Map full team name to tricode
            home_tri = _name_to_tri(home)
            if home_tri:
                lines[home_tri] = {
                    "spread": spread,
                    "total": total,
                    "home_name": home,
                }

        # Log API usage
        usage_path = DATA_DIR / "odds_api_usage.json"
        usage = json.loads(usage_path.read_text()) if usage_path.exists() else {"queries_used": 0}
        usage["queries_used"] = usage.get("queries_used", 0) + 1
        usage_path.write_text(json.dumps(usage))

        return lines

    except Exception as e:
        log.warning(f"Odds API failed: {e}")
        return {}


def _name_to_tri(name: str) -> Optional[str]:
    """Convert full team name to tricode."""
    mapping = {
        "Orlando Magic": "ORL", "Washington Wizards": "WSH", "Miami Heat": "MIA",
        "Houston Rockets": "HOU", "Minnesota Timberwolves": "MIN", "San Antonio Spurs": "SA",
        "Phoenix Suns": "PHX", "Denver Nuggets": "DEN", "Sacramento Kings": "SAC",
        "Dallas Mavericks": "DAL", "Utah Jazz": "UTAH", "Brooklyn Nets": "BKN",
        "Golden State Warriors": "GS", "Toronto Raptors": "TOR", "Detroit Pistons": "DET",
        "Chicago Bulls": "CHI", "Los Angeles Lakers": "LAL", "New Orleans Pelicans": "NO",
        "Boston Celtics": "BOS", "Cleveland Cavaliers": "CLE", "Indiana Pacers": "IND",
        "Milwaukee Bucks": "MIL", "Oklahoma City Thunder": "OKC", "Memphis Grizzlies": "MEM",
        "Los Angeles Clippers": "LAC", "Philadelphia 76ers": "PHI", "Atlanta Hawks": "ATL",
        "Charlotte Hornets": "CHA", "Portland Trail Blazers": "POR", "New York Knicks": "NYK",
    }
    return mapping.get(name)


# ── ML Model Projection ────────────────────────────────────────────────────

def load_model():
    """Load the trained totals gradient boosting model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle.get("feature_cols", [])


def load_spread_model():
    """Load the trained spread model."""
    if not SPREAD_MODEL_PATH.exists():
        return None, []
    with open(SPREAD_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle.get("feature_cols", [])


def get_team_features(team: str) -> Dict[str, float]:
    """Get rolling team stats from DB for prediction."""
    team = normalize_tri(team)
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT home, away, home_score, away_score, date
            FROM games
            WHERE (home=? OR away=?)
            ORDER BY date DESC LIMIT 20
        """, [team, team]).fetchall()
        conn.close()

        if not rows:
            return {}

        pts_for = []
        pts_against = []
        totals = []
        wins = []
        for home, away, hs, aws, d in rows:
            is_home = (home == team)
            pf = hs if is_home else aws
            pa = aws if is_home else hs
            pts_for.append(pf)
            pts_against.append(pa)
            totals.append(pf + pa)
            wins.append(1 if pf > pa else 0)

        avg_pts = np.mean(pts_for)
        avg_opp = np.mean(pts_against)
        avg_total = np.mean(totals)
        pace_proxy = avg_total / 2.24

        # Four Factors estimates
        pts_rel = (avg_pts - 108.0) / 15.0
        efg = min(0.60, max(0.48, 0.530 + pts_rel * 0.015))
        tov_rate = min(0.17, max(0.09, 0.130 - pts_rel * 0.005))
        ft_rate = min(0.32, max(0.15, 0.240 + pts_rel * 0.01))

        pts_std = float(np.std(pts_for)) if len(pts_for) > 1 else 12.0
        last5 = pts_for[:5]
        momentum = float(np.mean(last5) - avg_pts) if last5 else 0.0

        return {
            "pts_for_avg": round(avg_pts, 1),
            "pts_against_avg": round(avg_opp, 1),
            "net_rating": round(avg_pts - avg_opp, 2),
            "win_pct": round(sum(wins) / len(wins), 3),
            "last5_pts_for": round(np.mean(last5), 1) if last5 else avg_pts,
            "pace_proxy": round(pace_proxy, 1),
            "efg_pct": round(efg, 3),
            "tov_rate": round(tov_rate, 3),
            "ft_rate": round(ft_rate, 3),
            "pts_std_dev": round(pts_std, 2),
            "momentum": round(momentum, 2),
            "ortg_est": round((avg_pts / max(pace_proxy, 1)) * 100, 1),
            "drtg_est": round((avg_opp / max(pace_proxy, 1)) * 100, 1),
        }
    except Exception as e:
        log.error(f"Team features failed for {team}: {e}")
        return {}


def ml_project(home: str, away: str, model, feature_cols: list) -> Optional[float]:
    """Run ML model projection for total points."""
    try:
        h = get_team_features(home)
        a = get_team_features(away)
        if not h or not a:
            return None

        league_avg = 113.0
        h_proj = (h["pts_for_avg"] * a["pts_against_avg"] / league_avg)
        a_proj = (a["pts_for_avg"] * h["pts_against_avg"] / league_avg)
        math_total = h_proj + a_proj

        row = {
            "home_pts_for": h["pts_for_avg"],
            "home_pts_against": h["pts_against_avg"],
            "home_net_rating": h["net_rating"],
            "home_win_pct": h["win_pct"],
            "home_last5_pts_for": h["last5_pts_for"],
            "home_games_played": 20,
            "away_pts_for": a["pts_for_avg"],
            "away_pts_against": a["pts_against_avg"],
            "away_net_rating": a["net_rating"],
            "away_win_pct": a["win_pct"],
            "away_last5_pts_for": a["last5_pts_for"],
            "away_games_played": 20,
            "math_total_projection": round(math_total, 1),
            "pace_proxy_avg": round((h["pace_proxy"] + a["pace_proxy"]) / 2, 1),
            "combined_pts_avg": round(h["pts_for_avg"] + a["pts_for_avg"], 1),
            "net_rating_diff": round(h["net_rating"] - a["net_rating"], 2),
            "h2h_total_avg": 224.0,
            "h2h_games": 0,
            "h2h_over220_rate": 0.5,
            "home_rest_days": 2,
            "away_rest_days": 2,
            "home_b2b": 0,
            "away_b2b": 0,
            "home_injury_impact": 0.0,
            "away_injury_impact": 0.0,
            # Lines (NaN = not available; model handles with imputation)
            "open_total_line": float("nan"),
            "close_total_line": float("nan"),
            # Four Factors
            "home_efg": h["efg_pct"],
            "away_efg": a["efg_pct"],
            "efg_diff": round(h["efg_pct"] - a["efg_pct"], 3),
            "home_tov_rate": h["tov_rate"],
            "away_tov_rate": a["tov_rate"],
            "tov_diff": round(h["tov_rate"] - a["tov_rate"], 3),
            "home_orb_rate": 0.280,
            "away_orb_rate": 0.280,
            "orb_diff": 0.0,
            "home_ft_rate": h["ft_rate"],
            "away_ft_rate": a["ft_rate"],
            "ft_rate_diff": round(h["ft_rate"] - a["ft_rate"], 3),
            "home_net_rtg": h["ortg_est"] - h["drtg_est"],
            "away_net_rtg": a["ortg_est"] - a["drtg_est"],
            "net_rtg_diff_ff": round((h["ortg_est"] - h["drtg_est"]) - (a["ortg_est"] - a["drtg_est"]), 2),
            "home_pts_std": h["pts_std_dev"],
            "away_pts_std": a["pts_std_dev"],
            "home_pts_floor": h["pts_for_avg"] - 1.5 * h["pts_std_dev"],
            "home_pts_ceiling": h["pts_for_avg"] + 1.5 * h["pts_std_dev"],
            "away_pts_floor": a["pts_for_avg"] - 1.5 * a["pts_std_dev"],
            "away_pts_ceiling": a["pts_for_avg"] + 1.5 * a["pts_std_dev"],
            "home_momentum": h["momentum"],
            "away_momentum": a["momentum"],
            "upset_risk_score": round(
                (a["pts_for_avg"] - a["pts_against_avg"]) - (h["pts_for_avg"] - h["pts_against_avg"])
                + (a["momentum"] - h["momentum"]) * 0.5, 2
            ),
            "combined_std_dev": round(h["pts_std_dev"] + a["pts_std_dev"], 2),
            "home_3pa_rate": 0.42,
            "away_3pa_rate": 0.42,
            "home_3p_pct": 0.36,
            "away_3p_pct": 0.36,
        }

        # Use only features the model was trained on
        available = [c for c in feature_cols if c in row]
        X = pd.DataFrame([{c: row.get(c, 0) for c in available}])
        # Impute NaN line columns with league average
        X = X.fillna({"open_total_line": 224.0, "close_total_line": 224.0}).fillna(0)
        pred = float(model.predict(X)[0])
        return round(pred, 1)

    except Exception as e:
        log.error(f"ML projection failed: {e}")
        return None


# ── EV + Kelly ─────────────────────────────────────────────────────────────

from scipy.stats import norm as scipy_norm

# ── Minimum gap thresholds (based on empirical MAE from 18,708-game backtest) ──
# Only present total picks when model disagrees with market by > MAE.
# Below this threshold, the signal is inside the noise and has no provable edge.
# NBA backtest: gap>15pts → Under 80.6% win rate, Over 90.3% win rate (108/7710 signals)
MIN_TOTAL_GAP_NBA = 15.0     # pts — minimum model vs market gap for NBA totals
MIN_TOTAL_GAP_NCAAB = 14.0   # pts — NCAAB model MAE
MIN_SPREAD_GAP = 4.0          # pts — minimum model vs market gap for spreads

def implied_prob(american_odds: float) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def remove_vig(prob_over: float, prob_under: float) -> tuple:
    total = prob_over + prob_under
    return prob_over / total, prob_under / total

def spread_project(home: str, away: str, spread_model, spread_feature_cols: list) -> Optional[float]:
    """Run spread model projection. Returns predicted home margin."""
    try:
        h = get_team_features(home)
        a = get_team_features(away)
        if not h or not a:
            return None

        row = {
            "home_pts_for": h["pts_for_avg"],
            "home_pts_against": h["pts_against_avg"],
            "home_net_rating": h["net_rating"],
            "home_win_pct": h["win_pct"],
            "home_last5_pts_for": h["last5_pts_for"],
            "away_pts_for": a["pts_for_avg"],
            "away_pts_against": a["pts_against_avg"],
            "away_net_rating": a["net_rating"],
            "away_win_pct": a["win_pct"],
            "away_last5_pts_for": a["last5_pts_for"],
            "net_rating_diff": round(h["net_rating"] - a["net_rating"], 2),
            "win_pct_diff": round(h["win_pct"] - a["win_pct"], 3),
            "pts_for_diff": round(h["pts_for_avg"] - a["pts_for_avg"], 1),
            "pts_against_diff": round(h["pts_against_avg"] - a["pts_against_avg"], 1),
            "rest_diff": 0,  # would need live schedule data
            "home_b2b": 0,
            "away_b2b": 0,
            # Home/away splits (use net rating as proxy when splits unavailable)
            "home_home_ortg": h["pts_for_avg"] + 2.0,   # home teams score more at home
            "home_away_ortg": h["pts_for_avg"] - 2.0,
            "away_home_ortg": a["pts_for_avg"] + 2.0,
            "away_away_ortg": a["pts_for_avg"] - 2.0,
            "home_home_drtg": h["pts_against_avg"] - 1.5,
            "home_away_drtg": h["pts_against_avg"] + 1.5,
            "away_home_drtg": a["pts_against_avg"] - 1.5,
            "away_away_drtg": a["pts_against_avg"] + 1.5,
            "home_home_win_pct": min(1.0, h["win_pct"] + 0.08),
            "away_away_win_pct": max(0.0, a["win_pct"] - 0.08),
            "h2h_total_avg": 224.0,
            "h2h_games": 0,
            "home_injury_impact": 0.0,
            "away_injury_impact": 0.0,
            "open_spread": 0.0,  # unknown pre-game; model learned to ignore when 0
        }

        available = [c for c in spread_feature_cols if c in row]
        X = pd.DataFrame([{c: row.get(c, 0) for c in available}]).fillna(0)
        pred_margin = float(spread_model.predict(X)[0])
        return round(pred_margin, 1)

    except Exception as e:
        log.error(f"Spread projection failed: {e}")
        return None


def model_prob_spread(market_spread: float, model_margin: float, sigma: float = 14.2) -> tuple:
    """
    Probability of home covering the spread.
    sigma=14.2 = historical std dev of NBA margins.
    """
    from scipy.stats import norm
    p_home_covers = 1 - norm.cdf(market_spread, loc=model_margin, scale=sigma)
    p_away_covers = norm.cdf(market_spread, loc=model_margin, scale=sigma)
    return p_home_covers, p_away_covers


def model_prob_total(market_line: float, model_total: float, sigma: float = 18.0) -> tuple:
    p_over = 1 - scipy_norm.cdf(market_line, loc=model_total, scale=sigma)
    p_under = scipy_norm.cdf(market_line, loc=model_total, scale=sigma)
    return p_over, p_under

def ev_and_kelly(p_win: float, odds: float = -110, bankroll: float = 500, kelly_fraction: float = 0.25) -> Dict:
    decimal = (100 / abs(odds) + 1) if odds < 0 else (odds / 100 + 1)
    b = decimal - 1
    q = 1 - p_win
    ev_pct = (p_win * b - q) * 100
    full_kelly = max(0, (b * p_win - q) / b)
    bet_size = min(bankroll * full_kelly * kelly_fraction, bankroll * 0.05)
    return {
        "ev_pct": round(ev_pct, 1),
        "p_win": round(p_win, 3),
        "kelly_bet": round(bet_size, 0),
        "recommended": ev_pct >= 3.0 and bet_size >= 5.0,
    }


# ── Main Pick Generator ────────────────────────────────────────────────────

def generate_picks(game_date: Optional[str] = None, single_game: Optional[tuple] = None) -> List[Dict]:
    """Generate ranked picks for tonight's slate."""
    log.info("Loading models...")
    try:
        model, feature_cols = load_model()
    except Exception as e:
        log.error(f"Totals model load failed: {e}")
        model, feature_cols = None, []

    spread_model, spread_feature_cols = load_spread_model()
    if spread_model:
        log.info("Spread model loaded ✅")
    else:
        log.warning("Spread model not found — spread picks disabled")

    log.info("Fetching slate...")
    if single_game:
        games = [{"home": single_game[0], "away": single_game[1], "status": "scheduled"}]
    else:
        games = fetch_tonight_slate(game_date)

    if not games:
        log.warning("No games found.")
        return []

    log.info(f"Fetching DraftKings lines...")
    lines = fetch_dk_lines()

    log.info("Running picks engine...")
    picks = []

    for game in games:
        home, away = game["home"], game["away"]
        if not home or not away:
            continue

        result: Dict[str, Any] = {
            "home": home, "away": away,
            "status": game.get("status", ""),
            "tip_time": game.get("status", ""),
        }

        # ── ML projection (totals) ──
        ml_total = None
        if model:
            ml_total = ml_project(home, away, model, feature_cols)
        result["ml_total"] = ml_total

        # ── Spread projection ──
        ml_margin = None
        if spread_model:
            ml_margin = spread_project(normalize_tri(home), normalize_tri(away), spread_model, spread_feature_cols)
        result["ml_margin"] = ml_margin  # positive = home favored

        # ── Possessions projection ──
        try:
            from analyzers.four_factors import FourFactors
            ff = FourFactors()
            poss_proj = ff.project_game_total(normalize_tri(home), normalize_tri(away))
            result["poss_total"] = poss_proj["projected_total"]
            result["poss_spread"] = poss_proj["projected_spread"]
            result["expected_poss"] = poss_proj["expected_possessions"]
        except Exception:
            poss_proj = None
            result["poss_total"] = None
            result["poss_spread"] = None

        # ── Consensus projection (blend ML + possessions) ──
        if ml_total and result["poss_total"]:
            result["consensus_total"] = round(0.6 * ml_total + 0.4 * result["poss_total"], 1)
        elif ml_total:
            result["consensus_total"] = ml_total
        elif result["poss_total"]:
            result["consensus_total"] = result["poss_total"]
        else:
            result["consensus_total"] = None

        # ── Market lines ──
        game_lines = lines.get(home, {})
        market_total = game_lines.get("total")
        market_spread = game_lines.get("spread")
        result["market_total"] = market_total
        result["market_spread"] = market_spread

        # ── Motivation flags ──
        try:
            from analyzers.motivation_flags import compute_motivation_flags
            mot = compute_motivation_flags(home, away, game_date)
            result["motivation"] = mot
            # Apply adjustments to consensus
            if result["consensus_total"] and mot["total_adjustment"] != 0:
                result["consensus_total"] = round(
                    result["consensus_total"] + mot["total_adjustment"], 1
                )
        except Exception:
            result["motivation"] = {"flags": [], "total_adjustment": 0, "summary": "UNAVAILABLE"}

        # ── Referee context ──
        try:
            from analyzers.referee_lookup import get_game_ref_context
            ref_ctx = get_game_ref_context(home, game_date)
            result["referee"] = ref_ctx
            if result["consensus_total"] and ref_ctx["crew_total_adj"] != 0:
                result["consensus_total"] = round(
                    result["consensus_total"] + ref_ctx["crew_total_adj"], 1
                )
        except Exception:
            result["referee"] = {"crew_total_adj": 0, "tier": "UNKNOWN", "note": "unavailable"}

        # ── EV analysis ──
        bets = []

        # Spread EV
        # market_spread convention: negative = home favored (e.g., -5.5 = home -5.5)
        # ml_margin: positive = home wins by X, negative = away wins by X
        # coverage_edge: how much model margin exceeds what home needs to cover
        #   coverage_edge = ml_margin - (-market_spread) = ml_margin + market_spread
        #   positive = model thinks home covers; negative = model thinks away covers
        if ml_margin is not None and market_spread is not None:
            coverage_edge = ml_margin + market_spread  # positive = home covers
            if abs(coverage_edge) > 2.5:  # require meaningful edge
                # Use market_spread as the cover line (home needs to win by -market_spread)
                cover_line = -market_spread  # positive = home must win by this much
                p_home, p_away = model_prob_spread(cover_line, ml_margin)
                if coverage_edge > 0:  # model says home covers
                    ev = ev_and_kelly(p_home, odds=-110, bankroll=500)
                    direction = f"{home} {market_spread:+.1f}"
                    if ev["recommended"]:
                        bets.append({
                            "type": "SPREAD",
                            "direction": direction,
                            "line": market_spread,
                            "model_proj": ml_margin,
                            "edge_pts": round(coverage_edge, 1),
                            "ev_pct": ev["ev_pct"],
                            "kelly_bet": ev["kelly_bet"],
                            "confidence": "HIGH" if abs(coverage_edge) > 5 else "MEDIUM",
                        })
                else:  # model says away covers
                    ev = ev_and_kelly(p_away, odds=-110, bankroll=500)
                    away_spread = -market_spread
                    direction = f"{away} {away_spread:+.1f}"
                    if ev["recommended"]:
                        bets.append({
                            "type": "SPREAD",
                            "direction": direction,
                            "line": away_spread,
                            "model_proj": ml_margin,
                            "edge_pts": round(coverage_edge, 1),
                            "ev_pct": ev["ev_pct"],
                            "kelly_bet": ev["kelly_bet"],
                            "confidence": "HIGH" if abs(coverage_edge) > 5 else "MEDIUM",
                        })

        if result["consensus_total"] and market_total:
            edge = result["consensus_total"] - market_total
            p_over, p_under = model_prob_total(market_total, result["consensus_total"])
            # ── Minimum gap filter ──
            # Only recommend total bets when model gap exceeds 1× MAE (15 pts NBA).
            # Gaps below this are inside the model's noise floor — no provable edge.
            # Backtest: gap>15pts → 80%+ win rate; gap<15pts → ~62% win rate (barely +EV at -110 juice).
            if abs(edge) >= MIN_TOTAL_GAP_NBA:
                direction = "OVER" if edge > 0 else "UNDER"
                p_win = p_over if direction == "OVER" else p_under
                ev = ev_and_kelly(p_win, odds=-110, bankroll=500)
                if ev["recommended"]:
                    bets.append({
                        "type": "TOTAL",
                        "direction": direction,
                        "line": market_total,
                        "model_proj": result["consensus_total"],
                        "edge_pts": round(edge, 1),
                        "ev_pct": ev["ev_pct"],
                        "kelly_bet": ev["kelly_bet"],
                        "confidence": "HIGH" if abs(edge) > 20 else "MEDIUM",
                        "gap_filter": f"PASSED (gap {abs(edge):.1f}pts ≥ {MIN_TOTAL_GAP_NBA}pt min)",
                    })
            elif abs(edge) > 2.0:
                # Log that it was filtered out — not silently dropped
                result.setdefault("filtered_bets", []).append({
                    "type": "TOTAL",
                    "direction": "OVER" if edge > 0 else "UNDER",
                    "line": market_total,
                    "model_proj": result["consensus_total"],
                    "edge_pts": round(edge, 1),
                    "reason": f"Gap {abs(edge):.1f}pts < {MIN_TOTAL_GAP_NBA}pt minimum (inside model noise floor)",
                })

        result["bets"] = bets
        result["top_ev"] = max((b["ev_pct"] for b in bets), default=0.0)
        picks.append(result)

    # Sort by top EV descending
    picks.sort(key=lambda x: x["top_ev"], reverse=True)
    return picks


def print_picks(picks: List[Dict]):
    """Format and print picks to terminal."""
    today = date.today().strftime("%B %d, %Y")
    print(f"\n{'='*60}")
    print(f"🏀 NBA PICKS — {today}")
    print(f"{'='*60}\n")

    has_bets = False
    for p in picks:
        home, away = p["home"], p["away"]
        tip = p.get("tip_time", "")
        print(f"{'─'*50}")
        print(f"{away} @ {home}  |  {tip}")

        # Projections
        if p.get("ml_total"):
            print(f"  ML model:    {p['ml_total']} total pts")
        if p.get("poss_total"):
            print(f"  Poss model:  {p['poss_total']} total pts")
        if p.get("consensus_total"):
            print(f"  ⚡ Consensus: {p['consensus_total']} total pts", end="")
            if p.get("market_total"):
                diff = p["consensus_total"] - p["market_total"]
                print(f"  (market: {p['market_total']}, edge: {diff:+.1f})", end="")
            print()
        else:
            print(f"  No projection available")

        # Motivation flags
        mot = p.get("motivation", {})
        if mot.get("flags"):
            for f in mot["flags"]:
                icon = "🚨" if f["risk"] == "HIGH" else "⚠️"
                print(f"  {icon} {f['flag']}: {f['description'][:60]}")

        # Referee context
        ref = p.get("referee", {})
        if ref.get("tier") not in ("UNKNOWN", None) and ref.get("crew_total_adj", 0) != 0:
            print(f"  👨‍⚖️ Refs: [{ref['tier']}] {ref.get('note', '')}")

        # Bets
        if p.get("ml_margin") is not None:
            margin = p["ml_margin"]
            print(f"  📐 Spread model: {home} by {margin:+.1f} pts", end="")
            if p.get("market_spread") is not None:
                print(f" (market: {home} {p['market_spread']:+.1f})", end="")
            print()

        if p["bets"]:
            has_bets = True
            for bet in p["bets"]:
                conf_icon = "🔥" if bet["confidence"] == "HIGH" else "✅"
                bet_type = f"[{bet['type']}]" if "type" in bet else ""
                print(f"\n  {conf_icon} BET {bet_type}: {bet['direction']}")
                print(f"     EV: {bet['ev_pct']:+.1f}% | Kelly: ${bet['kelly_bet']:.0f} (on $500 bankroll)")
                print(f"     Model {bet['model_proj']:+.1f} vs Market {bet['line']:+.1f} → {bet['edge_pts']:+.1f} pts edge")
        else:
            print(f"  ⛔ No bet (EV < 3% threshold)")

        print()

    if not has_bets:
        print("\n⚠️  No games passed the 3% EV threshold today — pass the slate.\n")
    else:
        print("\n📊 Only bet what's above. Everything else is noise.\n")


def picks_to_discord(picks: List[Dict]) -> str:
    """Format picks as a Discord message."""
    today = date.today().strftime("%b %d")
    lines = [f"🏀 **NBA Picks — {today}**\n"]

    bet_games = [p for p in picks if p["bets"]]
    no_bet_games = [p for p in picks if not p["bets"]]

    if bet_games:
        lines.append("**✅ BETS (passed 3% EV gate):**")
        for p in bet_games:
            home, away = p["home"], p["away"]
            for bet in p["bets"]:
                conf = "🔥" if bet["confidence"] == "HIGH" else "✅"
                bet_label = f"[{bet.get('type','TOTAL')}] " if bet.get("type") else ""
                lines.append(
                    f"{conf} **{away} @ {home}** — {bet_label}{bet['direction']}"
                    f" | EV {bet['ev_pct']:+.1f}% | Kelly ${bet['kelly_bet']:.0f}"
                )
                lines.append(f"   Model: {bet['model_proj']:+.1f} vs Market: {bet['line']:+.1f} ({bet['edge_pts']:+.1f} pts edge)")
            # Flags
            mot = p.get("motivation", {})
            for f in mot.get("flags", []):
                icon = "🚨" if f["risk"] == "HIGH" else "⚠️"
                lines.append(f"   {icon} {f['flag']}")
            lines.append("")
    else:
        lines.append("⛔ **No bets today** — no game cleared 3% EV threshold.\n")

    if no_bet_games:
        skipped = ", ".join(f"{p['away']}@{p['home']}" for p in no_bet_games)
        lines.append(f"_Passed (no edge): {skipped}_")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Daily Picks")
    parser.add_argument("--date", help="Date YYYYMMDD (default: today)")
    parser.add_argument("--game", nargs=2, metavar=("HOME", "AWAY"), help="Single game")
    parser.add_argument("--discord", action="store_true", help="Output Discord-formatted message")
    parser.add_argument("--save", help="Save picks to file")
    args = parser.parse_args()

    picks = generate_picks(
        game_date=args.date,
        single_game=tuple(args.game) if args.game else None,
    )

    if args.discord:
        print(picks_to_discord(picks))
    else:
        print_picks(picks)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(picks, f, indent=2, default=str)
        log.info(f"Saved picks to {args.save}")

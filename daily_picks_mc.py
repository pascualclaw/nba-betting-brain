"""
daily_picks_mc.py — Pre-game Monte Carlo Pick Generator (NBA + NCAAB)
======================================================================
Runs 1,000,000 vectorized simulations per game.

Key design:
- Pre-game precision model (Renaissance/Kambi style)
- ALL inputs locked: confirmed injuries, B2B status, referee crew, H2H pace
- Arbitrage scan: only outputs where our P > market implied by >10%
- Quality > quantity: 2-3 high-confidence picks >> 10 marginal ones
- No fake EV — never fabricates edge

Usage:
    source venv/bin/activate
    ODDS_API_KEY=xxx python daily_picks_mc.py [--date 2026-03-10] [--sport nba|ncaab|both]

Output: Discord message + console log
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from simulators.monte_carlo import GameSimulator, american_to_implied_prob
from simulators.ncaab_monte_carlo import NCAABGameSimulator
from simulators.props_from_sim import scan_props, get_player_distributions
from collectors.player_game_logs import collect_team_player_logs, get_player_logs_from_db
from analyzers.referee_lookup import get_all_games_ref_context, analyze_crew

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
today_str = datetime.now().strftime("%Y%m%d")
log_file = LOG_DIR / f"mc_build_{today_str}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
DB_PATH = ROOT / "data" / "nba_betting.db"
NCAAB_DB_PATH = ROOT / "data" / "ncaab_betting.db"
ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_NCAAB = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
N_SIMS = 1_000_000
MIN_EDGE_PCT = 10.0
ESPN_RATE_LIMIT = 0.5

# ESPN team abbr normalization
ESPN_ABBR_MAP = {
    "GS": "GSW", "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "WSH": "WAS", "UTH": "UTA", "PHO": "PHX", "UTAH": "UTA",
}


def norm_abbr(a: str) -> str:
    return ESPN_ABBR_MAP.get(a.upper(), a.upper())


def fetch_json(url: str) -> Optional[dict]:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        logger.warning(f"Fetch failed: {url} — {e}")
        return None


def american_odds_str(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


# ── Step 1: Fetch Tonight's Games ──────────────────────────────────────────

def fetch_nba_games(game_date: str) -> List[Dict]:
    """
    Fetch NBA games from ESPN scoreboard for given date.
    Returns list of game dicts with team names, IDs, odds, and injury info.
    """
    url = f"{ESPN_NBA}/scoreboard?dates={game_date.replace('-', '')}"
    data = fetch_json(url)
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home or not away:
            continue

        home_abbr = norm_abbr(home["team"]["abbreviation"])
        away_abbr = norm_abbr(away["team"]["abbreviation"])
        event_id = event["id"]

        # Get detailed game info (odds + injuries)
        time.sleep(ESPN_RATE_LIMIT)
        detail = fetch_json(f"{ESPN_NBA}/summary?event={event_id}")

        # Parse DK odds from pickcenter
        dk_odds = {}
        if detail:
            for provider in detail.get("pickcenter", []):
                name = provider.get("provider", {}).get("name", "")
                if "Draft" in name:
                    dk_odds = {
                        "spread": provider.get("spread"),
                        "total": provider.get("overUnder"),
                        "home_spread": provider.get("spread"),  # negative = home fav
                    }
                    break

        # Parse injuries
        injuries = {"home_out": [], "home_dtd": [], "away_out": [], "away_dtd": []}
        if detail:
            for inj_team in detail.get("injuries", []):
                t_abbr = norm_abbr(inj_team.get("team", {}).get("abbreviation", ""))
                is_home = (t_abbr == home_abbr)
                for inj in inj_team.get("injuries", []):
                    pname = inj.get("athlete", {}).get("displayName", "")
                    status = inj.get("status", "").lower()
                    if "out" in status or "suspension" in status:
                        if is_home:
                            injuries["home_out"].append(pname)
                        else:
                            injuries["away_out"].append(pname)
                    elif "day-to-day" in status or "dtd" in status:
                        if is_home:
                            injuries["home_dtd"].append(pname)
                        else:
                            injuries["away_dtd"].append(pname)

        games.append({
            "event_id": event_id,
            "home": home_abbr,
            "away": away_abbr,
            "home_name": home["team"].get("displayName", home_abbr),
            "away_name": away["team"].get("displayName", away_abbr),
            "date": event.get("date", "")[:16],
            "dk_spread": dk_odds.get("spread"),
            "dk_total": dk_odds.get("total"),
            **injuries,
        })
        logger.info(
            f"  {away_abbr} @ {home_abbr} | "
            f"DK: spread={dk_odds.get('spread','?')} O/U={dk_odds.get('total','?')} | "
            f"Injuries — home OUT: {injuries['home_out']} | away OUT: {injuries['away_out']}"
        )

    return games


def fetch_ncaab_games(game_date: str) -> List[Dict]:
    """
    Fetch NCAAB tournament games from ESPN scoreboard.
    Uses groups=50 for small conferences + limit=200 for majors.
    Returns list of game dicts.
    """
    date_str = game_date.replace("-", "")
    all_events = {}

    for url_suffix in [
        f"scoreboard?groups=50&limit=100&dates={date_str}",
        f"scoreboard?limit=200&dates={date_str}",
    ]:
        data = fetch_json(f"{ESPN_NCAAB}/{url_suffix}")
        if not data:
            continue
        for event in data.get("events", []):
            if event["id"] not in all_events:
                all_events[event["id"]] = event
        time.sleep(ESPN_RATE_LIMIT)

    games = []
    for event_id, event in all_events.items():
        comp = event["competitions"][0]
        note = comp.get("notes", [{}])
        note_text = note[0].get("headline", "") if note else ""

        # Only tournament/championship games
        is_tournament = any(kw in note_text for kw in [
            "Tournament", "Championship", "Final", "Semifinal", "Quarterfinal",
            "Round", "Playoff", "Conference"
        ])
        if not is_tournament:
            continue

        # Parse conference from note
        conference = note_text.split(" Tournament")[0].split(" Championship")[0] if note_text else ""
        round_info = ""
        for rnd in ["Final", "Semifinal", "Quarterfinal", "1st Round", "2nd Round", "3rd Round"]:
            if rnd in note_text:
                round_info = rnd
                break

        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c["homeAway"] == "home"), competitors[0] if competitors else {})
        away = next((c for c in competitors if c["homeAway"] == "away"), competitors[1] if len(competitors) > 1 else {})

        home_name = home.get("team", {}).get("displayName", "?")
        away_name = away.get("team", {}).get("displayName", "?")
        home_id = home.get("team", {}).get("id", "")
        away_id = away.get("team", {}).get("id", "")

        # Check neutral site
        neutral_site = comp.get("neutralSite", True)  # most tournament games are neutral
        if "Final" in note_text or "Semifinal" in note_text:
            neutral_site = True

        games.append({
            "event_id": event_id,
            "home_name": home_name,
            "away_name": away_name,
            "home_id": str(home_id),
            "away_id": str(away_id),
            "conference": conference,
            "round": round_info,
            "neutral_site": neutral_site,
            "date": event.get("date", "")[:16],
            "note": note_text,
        })

    logger.info(f"Found {len(games)} NCAAB tournament games")
    return games


# ── Step 2: Fetch Odds API Lines ───────────────────────────────────────────

def fetch_nba_odds(api_key: str) -> Dict[str, Dict]:
    """
    Pull NBA DK lines. Returns dict keyed by "away_name @ home_name" pattern.
    Also returns by team abbreviation pair.
    """
    url = f"{ODDS_API_BASE}/sports/basketball_nba/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "bookmakers": "draftkings",
        "oddsFormat": "american",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        logger.info(f"NBA odds API remaining: {r.headers.get('x-requests-remaining','?')}")
        data = r.json()
    except Exception as e:
        logger.error(f"NBA odds fetch failed: {e}")
        return {}

    result = {}
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        eid = game.get("id", "")
        dk = next((b for b in game.get("bookmakers", []) if b.get("key") == "draftkings"), {})
        markets = {m["key"]: m for m in dk.get("markets", [])}

        h2h = markets.get("h2h", {}).get("outcomes", [])
        spreads = markets.get("spreads", {}).get("outcomes", [])
        totals = markets.get("totals", {}).get("outcomes", [])

        ml_home = next((o["price"] for o in h2h if o["name"] == home), None)
        ml_away = next((o["price"] for o in h2h if o["name"] == away), None)
        home_spread = next((o for o in spreads if o["name"] == home), {})
        sp_line = home_spread.get("point")
        total_out = next((o for o in totals if o["name"] == "Over"), {})
        ou_line = total_out.get("point")

        key = f"{away} @ {home}"
        result[key] = {
            "odds_api_id": eid,
            "home_team": home,
            "away_team": away,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "spread": sp_line,   # from home team perspective
            "total": ou_line,
        }

    return result


def fetch_ncaab_odds(api_key: str) -> Dict[str, Dict]:
    """Pull NCAAB DK lines. Keyed by 'away_team @ home_team'."""
    url = f"{ODDS_API_BASE}/sports/basketball_ncaab/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "bookmakers": "draftkings",
        "oddsFormat": "american",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        logger.info(f"NCAAB odds API remaining: {r.headers.get('x-requests-remaining','?')}")
        data = r.json()
    except Exception as e:
        logger.error(f"NCAAB odds fetch failed: {e}")
        return {}

    result = {}
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        eid = game.get("id", "")
        dk = next((b for b in game.get("bookmakers", []) if b.get("key") == "draftkings"), {})
        markets = {m["key"]: m for m in dk.get("markets", [])}

        h2h = markets.get("h2h", {}).get("outcomes", [])
        spreads = markets.get("spreads", {}).get("outcomes", [])
        totals = markets.get("totals", {}).get("outcomes", [])

        ml_home = next((o["price"] for o in h2h if o["name"] == home), None)
        ml_away = next((o["price"] for o in h2h if o["name"] == away), None)
        home_spread = next((o for o in spreads if o["name"] == home), {})
        sp_line = home_spread.get("point")
        total_out = next((o for o in totals if o["name"] == "Over"), {})
        ou_line = total_out.get("point")

        result[f"{away} @ {home}"] = {
            "odds_api_id": eid,
            "home_team": home,
            "away_team": away,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "spread": sp_line,
            "total": ou_line,
        }

    return result


# ── Step 3: B2B Detection ──────────────────────────────────────────────────

def get_b2b_teams(game_date: str) -> set:
    """
    Returns set of team abbreviations that played yesterday (on B2B tonight).
    """
    try:
        yesterday = datetime.strptime(game_date, "%Y-%m-%d")
        from datetime import timedelta
        yesterday = yesterday - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y%m%d")

        data = fetch_json(f"{ESPN_NBA}/scoreboard?dates={yesterday_str}")
        b2b_teams = set()
        if data:
            for event in data.get("events", []):
                for c in event["competitions"][0].get("competitors", []):
                    abbr = norm_abbr(c["team"]["abbreviation"])
                    b2b_teams.add(abbr)
        logger.info(f"B2B teams (played {yesterday.strftime('%Y-%m-%d')}): {sorted(b2b_teams)}")
        return b2b_teams
    except Exception as e:
        logger.warning(f"B2B detection failed: {e}")
        return set()


# ── Step 4: Collect Player Logs ────────────────────────────────────────────

def collect_player_logs_for_teams(teams: List[str], force_refresh: bool = False):
    """
    Collect player game logs for all teams playing tonight.
    Skip if already have fresh data (fetched today).
    """
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime("%Y-%m-%d")

    teams_to_fetch = []
    if not force_refresh:
        for team in teams:
            # Check if we have recent data
            row = conn.execute("""
                SELECT MAX(fetched_at) FROM player_game_logs WHERE team = ?
            """, (team,)).fetchone()
            last_fetch = row[0] if row else None
            if last_fetch and last_fetch[:10] == today:
                logger.info(f"  {team}: player logs already fresh (fetched today)")
            else:
                teams_to_fetch.append(team)
    else:
        teams_to_fetch = teams

    conn.close()

    if not teams_to_fetch:
        logger.info("All team player logs are fresh — skipping collection")
        return

    logger.info(f"Collecting player logs for: {teams_to_fetch}")
    for team in teams_to_fetch:
        try:
            n = collect_team_player_logs(team, n_games=15, top_n_players=9)
            logger.info(f"  {team}: collected {n} players")
        except Exception as e:
            logger.warning(f"  {team}: collection failed — {e}")


# ── Step 5: Run NBA Simulations ────────────────────────────────────────────

def run_nba_sims(games: List[Dict], nba_odds: Dict[str, Dict],
                 b2b_teams: set, ref_context: Dict) -> Dict[str, Any]:
    """
    Run 1M MC sims for all NBA games. Returns all results + qualifying picks.
    """
    all_results = []
    all_picks = []
    conn = sqlite3.connect(DB_PATH)

    for game in games:
        home = game["home"]
        away = game["away"]
        home_name = game["home_name"]
        away_name = game["away_name"]

        # Match to Odds API
        odds_key = None
        for k in nba_odds:
            if home in k and away in k:
                odds_key = k
                break
        if not odds_key:
            # Try partial match on team names
            for k, v in nba_odds.items():
                if (home.lower() in v["home_team"].lower() or v["home_team"].lower() in home.lower()):
                    odds_key = k
                    break

        if not odds_key:
            logger.warning(f"No DK odds found for {away} @ {home} — skipping")
            continue

        odds = nba_odds[odds_key]
        market_total = odds.get("total")
        market_spread = odds.get("spread")
        ml_home = odds.get("ml_home")
        ml_away = odds.get("ml_away")

        if market_total is None:
            logger.warning(f"No DK total for {away} @ {home} — skipping")
            continue

        logger.info(f"\n[SIM] {away} @ {home}")
        logger.info(f"  DK: spread={market_spread} O/U={market_total} ML: {ml_away}/{ml_home}")
        logger.info(f"  Injuries — home OUT: {game['home_out']} DTD: {game['home_dtd']}")
        logger.info(f"  Injuries — away OUT: {game['away_out']} DTD: {game['away_dtd']}")

        # Referee context
        ref = ref_context.get(home, {})
        ref_adj = ref.get("crew_total_adj", 0.0)
        ref_home_bias = (ref.get("home_foul_rate", 1.0) - 1.0) * 3.0  # rough translation
        ref_note = ref.get("note", "Refs TBD")
        if ref.get("refs"):
            logger.info(f"  Refs: {ref['refs']} → {ref_note}")

        # Build and run simulator
        sim = GameSimulator(
            home_team=home,
            away_team=away,
            market_total=float(market_total) if market_total else None,
            market_spread=float(market_spread) if market_spread else None,
            market_ml_home=int(ml_home) if ml_home else None,
            market_ml_away=int(ml_away) if ml_away else None,
            n_sims=N_SIMS,
        )

        sim.load_player_distributions(conn=conn)
        sim.set_injuries(
            home_out=game["home_out"], home_dtd=game["home_dtd"],
            away_out=game["away_out"], away_dtd=game["away_dtd"],
        )
        sim.set_b2b(
            home_b2b=(home in b2b_teams),
            away_b2b=(away in b2b_teams),
        )
        sim.set_referee_adj(
            crew_total_adj=ref_adj,
            home_bias=ref_home_bias,
        )

        run_result = sim.run(N_SIMS)
        summary = sim.summary()

        logger.info(
            f"  Result: Home proj={run_result['home_mean']:.1f} Away proj={run_result['away_mean']:.1f} "
            f"Total={run_result['proj_total']:.1f} Margin={run_result['proj_margin']:+.1f}"
        )

        picks = sim.arbitrage_scan(
            market_total=float(market_total) if market_total else None,
            market_spread=float(market_spread) if market_spread else None,
            market_ml_home=int(ml_home) if ml_home else None,
            market_ml_away=int(ml_away) if ml_away else None,
            min_edge_pct=MIN_EDGE_PCT,
        )

        for pick in picks:
            pick["game"] = f"{away_name} @ {home_name}"
            pick["home_b2b"] = home in b2b_teams
            pick["away_b2b"] = away in b2b_teams
            pick["home_out"] = game["home_out"]
            pick["away_out"] = game["away_out"]
            pick["ref_note"] = ref_note
            pick["proj_total"] = run_result["proj_total"]
            pick["proj_margin"] = run_result["proj_margin"]
            all_picks.append(pick)
            logger.info(
                f"  ✅ PICK: {pick['market']} {pick['line']} | "
                f"Our: {pick['our_prob']*100:.1f}% vs Market: {pick['market_implied']*100:.1f}% | "
                f"Edge: {pick['edge_pct']:.1f}% EV: {pick['ev_pct']:+.1f}% [{pick['tier']}]"
            )

        if not picks:
            logger.info(f"  ❌ No picks (no edge >{MIN_EDGE_PCT}%)")

        all_results.append({
            "game": f"{away} @ {home}",
            "summary": summary,
            "picks": picks,
        })

    conn.close()
    all_picks.sort(key=lambda x: x["ev_pct"], reverse=True)
    return {"results": all_results, "picks": all_picks}


# ── Step 6: Run NCAAB Simulations ─────────────────────────────────────────

def run_ncaab_sims(games: List[Dict], ncaab_odds: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Run 1M MC sims for all NCAAB tournament games.
    Returns qualifying picks sorted by EV.
    """
    all_results = []
    all_picks = []
    conn = sqlite3.connect(NCAAB_DB_PATH)

    for game in games:
        home_name = game["home_name"]
        away_name = game["away_name"]
        home_id = game["home_id"]
        away_id = game["away_id"]
        conference = game["conference"]
        round_info = game["round"]
        neutral = game["neutral_site"]

        # Match to Odds API
        odds_key = None
        odds = {}
        for k, v in ncaab_odds.items():
            home_match = any(w in v["home_team"] for w in home_name.split()[:2] if len(w) > 3)
            away_match = any(w in v["away_team"] for w in away_name.split()[:2] if len(w) > 3)
            if home_match and away_match:
                odds_key = k
                odds = v
                break

        if not odds or odds.get("total") is None:
            logger.info(f"NCAAB: No DK odds for {away_name} @ {home_name} — skipping")
            continue

        market_total = float(odds["total"])
        market_spread = float(odds["spread"]) if odds.get("spread") is not None else None
        ml_home = int(odds["ml_home"]) if odds.get("ml_home") else None
        ml_away = int(odds["ml_away"]) if odds.get("ml_away") else None

        logger.info(f"\n[NCAAB SIM] {away_name} @ {home_name} | {conference} {round_info}")
        logger.info(f"  DK: spread={market_spread} O/U={market_total} ML: {ml_away}/{ml_home}")
        logger.info(f"  Neutral site: {neutral}")

        sim = NCAABGameSimulator(
            home_team=home_name,
            away_team=away_name,
            home_id=home_id,
            away_id=away_id,
            market_total=market_total,
            market_spread=market_spread,
            market_ml_home=ml_home,
            market_ml_away=ml_away,
            neutral_site=neutral,
            tournament_round=round_info,
            conference=conference,
            n_sims=N_SIMS,
        )
        sim.load_distributions(conn=conn)
        run_result = sim.run(N_SIMS)

        logger.info(
            f"  Proj: Home={run_result['home_mean']:.1f} Away={run_result['away_mean']:.1f} "
            f"Total={run_result['proj_total']:.1f} Margin={run_result['proj_margin']:+.1f}"
        )

        picks = sim.arbitrage_scan(
            market_total=market_total,
            market_spread=market_spread,
            market_ml_home=ml_home,
            market_ml_away=ml_away,
            min_edge_pct=MIN_EDGE_PCT,
        )

        for pick in picks:
            all_picks.append(pick)
            logger.info(
                f"  ✅ NCAAB PICK: {pick['market']} {pick['line']} | "
                f"Our: {pick['our_prob']*100:.1f}% vs Market: {pick['market_implied']*100:.1f}% | "
                f"Edge: {pick['edge_pct']:.1f}% EV: {pick['ev_pct']:+.1f}% [{pick['tier']}]"
            )

        if not picks:
            logger.info("  ❌ No NCAAB picks (no edge >10%)")

        all_results.append({
            "game": f"{away_name} @ {home_name}",
            "picks": picks,
            "proj_total": run_result["proj_total"],
            "proj_margin": run_result["proj_margin"],
        })

    conn.close()
    all_picks.sort(key=lambda x: x["ev_pct"], reverse=True)
    return {"results": all_results, "picks": all_picks}


# ── Step 7: Props Scan ─────────────────────────────────────────────────────

def run_props_scan(games: List[Dict], nba_odds: Dict, api_key: str,
                   max_games: int = 4) -> List[Dict]:
    """
    Run props scan for top NBA games (by market interest).
    Limits API calls to stay within quota.
    """
    all_props = []

    # Focus on highest-interest games (big market teams or large injury context)
    priority_games = sorted(games, key=lambda g: (
        len(g.get("home_out", [])) + len(g.get("away_out", []))
    ), reverse=True)[:max_games]

    for game in priority_games:
        home = game["home"]
        away = game["away"]

        # Find Odds API event ID
        event_id = None
        for k, v in nba_odds.items():
            if home in k or away in k:
                event_id = v.get("odds_api_id")
                break

        if not event_id:
            continue

        for team in [home, away]:
            try:
                props = scan_props(
                    team=team,
                    opponent=away if team == home else home,
                    event_id=event_id,
                    api_key=api_key,
                    db_path=DB_PATH,
                    min_ev_pct=5.0,
                    min_sample=8,
                )
                if props:
                    logger.info(f"  Props for {team}: {len(props)} edges found")
                    all_props.extend(props[:3])  # top 3 per team
            except Exception as e:
                logger.warning(f"Props scan failed for {team}: {e}")
            time.sleep(0.3)

    all_props.sort(key=lambda x: x["ev_pct"], reverse=True)
    return all_props[:10]  # top 10 props total


# ── Step 8: Format Discord Output ─────────────────────────────────────────

def format_nba_picks_discord(picks: List[Dict], props: List[Dict],
                              game_date: str) -> str:
    """Format NBA picks for Discord output."""
    if not picks and not props:
        return (
            f"🏀 **NBA — {game_date}**\n"
            "No picks passed the 10% edge threshold tonight.\n"
            "Market is efficiently priced on all games — sitting out."
        )

    lines = [f"🏀 **NBA PICKS — {game_date}**"]
    lines.append("*Picks shown only where sim P% exceeds DK implied by >10%*\n")

    if picks:
        for i, pick in enumerate(picks[:6], 1):
            game = pick.get("game", "?")
            mkt = pick["market"]
            line = pick["line"]
            our_p = pick["our_prob"] * 100
            mkt_p = pick["market_implied"] * 100
            ev = pick["ev_pct"]
            edge = pick["edge_pct"]
            tier = pick["tier"]
            kelly = pick.get("kelly_pct", 0)
            proj = pick.get("proj_total", 0)
            proj_margin = pick.get("proj_margin", 0)

            out_flags = []
            if pick.get("home_out"):
                out_flags.append(f"OUT: {', '.join(pick['home_out'][:3])}")
            if pick.get("away_out"):
                out_flags.append(f"OUT: {', '.join(pick['away_out'][:3])}")
            if pick.get("home_b2b"):
                out_flags.append("HOME B2B")
            if pick.get("away_b2b"):
                out_flags.append("AWAY B2B")

            tier_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "📊"}.get(tier, "📊")

            lines.append(
                f"**{i}. {game}**\n"
                f"  {tier_emoji} **{mkt} {line}** [{tier}]\n"
                f"  Our: **{our_p:.1f}%** | Market: {mkt_p:.1f}% | Edge: +{edge:.1f}%\n"
                f"  EV: **+{ev:.1f}%** | Kelly: {kelly:.1f}% bankroll\n"
                f"  Sim: proj total={proj:.0f} | margin={proj_margin:+.1f}\n"
                + (f"  ⚠️ {' | '.join(out_flags)}\n" if out_flags else "")
                + (f"  📋 Refs: {pick.get('ref_note','TBD')}\n" if pick.get('ref_note') and 'TBD' not in pick.get('ref_note','') else "")
            )
    else:
        lines.append("*No NBA game picks passed the edge threshold.*")

    if props:
        lines.append("\n**🎯 PLAYER PROPS:**")
        for prop in props[:5]:
            player = prop["player"]
            mkt = prop["market"]
            side = prop["side"]
            line = prop["line"]
            our_p = prop["our_prob"] * 100
            ev = prop["ev_pct"]
            risk = " ⚠️ HIGH RISK" if prop.get("high_risk") else ""
            lines.append(
                f"  • **{player}** {mkt} {side} {line} | "
                f"Our {our_p:.0f}% | EV +{ev:.1f}%{risk}"
            )

    return "\n".join(lines)


def format_ncaab_picks_discord(picks: List[Dict], game_date: str) -> str:
    """Format NCAAB tournament picks for Discord."""
    if not picks:
        return (
            f"🏫 **NCAAB TOURNAMENTS — {game_date}**\n"
            "No tournament picks passed the 10% edge threshold.\n"
            "Sitting out — market is sharp on tonight's slate."
        )

    lines = [f"🏫 **NCAAB TOURNAMENT PICKS — {game_date}**"]
    lines.append("*Conference tournament week — neutral site, higher variance*")
    lines.append("*Only picks where our sim P% > DK implied by >10% shown*\n")

    for i, pick in enumerate(picks[:5], 1):
        game = pick.get("game", "?")
        conf = pick.get("conference", "")
        rnd = pick.get("round", "")
        mkt = pick["market"]
        line = pick["line"]
        our_p = pick["our_prob"] * 100
        mkt_p = pick["market_implied"] * 100
        ev = pick["ev_pct"]
        edge = pick["edge_pct"]
        tier = pick["tier"]
        kelly = pick.get("kelly_pct", 0)
        flags = pick.get("flags", [])

        tier_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "📊"}.get(tier, "📊")
        conf_str = f"{conf} {rnd}" if conf else rnd

        lines.append(
            f"**{i}. {game}** — {conf_str}\n"
            f"  {tier_emoji} **{mkt} {line}** [{tier}]\n"
            f"  Our: **{our_p:.1f}%** | Market: {mkt_p:.1f}% | Edge: +{edge:.1f}%\n"
            f"  EV: **+{ev:.1f}%** | Kelly: {kelly:.1f}% bankroll"
        )
        if flags:
            lines.append(f"  🚩 {' | '.join(flags)}")
        lines.append("")

    return "\n".join(lines)


def format_methodology_note() -> str:
    return (
        "\n---\n"
        "**📐 Methodology:**\n"
        "1M Monte Carlo sims/game | Player distributions from last 15 games\n"
        "Inputs: confirmed injuries, B2B status, matchup pace, referee crew\n"
        "Edge threshold: our P% must exceed DK implied by >10% to appear\n"
        "Quarter-Kelly sizing | Confidence: 🔥 HIGH (≥75%) ⚡ MED (65-75%) 📊 LOW (55-65%)\n"
        "⚠️ Never bet more than you can afford to lose. These are model outputs, not guarantees."
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sport", default="both", choices=["nba", "ncaab", "both"])
    parser.add_argument("--no-props", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    game_date = args.date
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        logger.error("ODDS_API_KEY not set — check ~/.openclaw/credentials/jarvis-github.env")
        sys.exit(1)

    logger.info(f"=== MC PICKS GENERATOR — {game_date} ===")
    logger.info(f"N_SIMS={N_SIMS:,} | MIN_EDGE={MIN_EDGE_PCT}%")

    nba_picks_msg = ""
    ncaab_picks_msg = ""

    # ── NBA ──────────────────────────────────────────────────────────
    if args.sport in ("nba", "both"):
        logger.info("\n[PHASE 1] Fetching NBA games & odds...")
        nba_games = fetch_nba_games(game_date)
        logger.info(f"Found {len(nba_games)} NBA games")

        nba_odds = fetch_nba_odds(api_key)
        logger.info(f"Got DK lines for {len(nba_odds)} NBA games")

        logger.info("\n[PHASE 2] Detecting B2B teams...")
        b2b_teams = get_b2b_teams(game_date)

        logger.info("\n[PHASE 3] Collecting player logs...")
        all_nba_teams = list(set(
            [g["home"] for g in nba_games] + [g["away"] for g in nba_games]
        ))
        collect_player_logs_for_teams(all_nba_teams, force_refresh=args.force_refresh)

        logger.info("\n[PHASE 4] Fetching referee context...")
        ref_context = get_all_games_ref_context(game_date.replace("-", ""))

        logger.info(f"\n[PHASE 5] Running NBA sims ({N_SIMS:,}/game)...")
        nba_results = run_nba_sims(nba_games, nba_odds, b2b_teams, ref_context)

        props = []
        if not args.no_props and nba_results["picks"]:
            logger.info("\n[PHASE 5b] Scanning player props...")
            try:
                props = run_props_scan(nba_games, nba_odds, api_key, max_games=3)
            except Exception as e:
                logger.warning(f"Props scan failed: {e}")

        nba_picks_msg = format_nba_picks_discord(
            nba_results["picks"], props, game_date
        )

        n_nba_picks = len(nba_results["picks"])
        logger.info(f"\nNBA: {n_nba_picks} picks passed {MIN_EDGE_PCT}% edge threshold")

    # ── NCAAB ────────────────────────────────────────────────────────
    if args.sport in ("ncaab", "both"):
        logger.info("\n[NCAAB PHASE 1] Fetching tournament games...")
        ncaab_games = fetch_ncaab_games(game_date)

        logger.info("\n[NCAAB PHASE 2] Fetching NCAAB odds...")
        ncaab_odds = fetch_ncaab_odds(api_key)
        logger.info(f"Got DK lines for {len(ncaab_odds)} NCAAB games")

        logger.info(f"\n[NCAAB PHASE 3] Running sims ({N_SIMS:,}/game)...")
        ncaab_results = run_ncaab_sims(ncaab_games, ncaab_odds)

        ncaab_picks_msg = format_ncaab_picks_discord(
            ncaab_results["picks"], game_date
        )

        n_ncaab_picks = len(ncaab_results["picks"])
        logger.info(f"\nNCAAB: {n_ncaab_picks} picks passed {MIN_EDGE_PCT}% edge threshold")

    # ── Post to Discord ───────────────────────────────────────────────
    logger.info("\n[PHASE FINAL] Building Discord output...")

    header = (
        f"🎯 **MONTE CARLO PICKS — {game_date}**\n"
        f"1,000,000 simulations per game | Pre-game precision model\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    full_msg = header

    if nba_picks_msg:
        full_msg += "\n" + nba_picks_msg

    if ncaab_picks_msg:
        full_msg += "\n\n" + ncaab_picks_msg

    full_msg += "\n" + format_methodology_note()

    # Split message if too long (Discord 2000 char limit)
    MAX_DISCORD = 1900
    messages = []
    if len(full_msg) <= MAX_DISCORD:
        messages = [full_msg]
    else:
        # Split at NBA/NCAAB boundary
        if nba_picks_msg and ncaab_picks_msg:
            msg1 = header + "\n" + nba_picks_msg
            msg2 = "\n" + ncaab_picks_msg + "\n" + format_methodology_note()
            messages = [msg1, msg2]
        else:
            # Chunk by lines
            chunk = ""
            for line in full_msg.split("\n"):
                if len(chunk) + len(line) + 1 > MAX_DISCORD:
                    messages.append(chunk)
                    chunk = line + "\n"
                else:
                    chunk += line + "\n"
            if chunk:
                messages.append(chunk)

    logger.info(f"\nFull output ({len(full_msg)} chars):\n{full_msg}\n")

    # Print to stdout (Discord posting done by caller/agent)
    print("\n" + "="*60)
    print("DISCORD OUTPUT:")
    print("="*60)
    for msg in messages:
        print(msg)
        print("---MESSAGE BREAK---")

    # Save picks to file for Discord agent to post
    output_path = ROOT / "data" / f"picks_{game_date}.json"
    picks_data = {
        "date": game_date,
        "nba_picks": nba_results["picks"] if args.sport in ("nba", "both") else [],
        "ncaab_picks": ncaab_results["picks"] if args.sport in ("ncaab", "both") else [],
        "discord_messages": messages,
    }
    output_path.write_text(json.dumps(picks_data, indent=2, default=str))
    logger.info(f"Picks saved to {output_path}")

    return picks_data


if __name__ == "__main__":
    main()

"""
Live Odds Collector — DraftKings live spread/total via The Odds API.

Fetches live in-game lines for DraftKings every few minutes.
Compares to pre-game lines to detect significant movement.

Usage:
    from collectors.live_odds import get_live_dk_lines, get_live_line_for_game

    lines = get_live_dk_lines()
    game_line = get_live_line_for_game("LAC", "IND")
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

# Load API key
def _load_odds_key() -> str:
    env_file = Path.home() / ".openclaw/credentials/jarvis-github.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("ODDS_API_KEY", "")

ODDS_API_KEY = _load_odds_key()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

DATA_DIR = Path(__file__).parent.parent / "data" / "live_odds"
DATA_DIR.mkdir(parents=True, exist_ok=True)
USAGE_FILE = Path(__file__).parent.parent / "data" / "odds_api_usage.json"

# Team name normalization (Odds API uses full names)
TEAM_NAME_TO_ABBREV = {
    "Los Angeles Clippers": "LAC", "Indiana Pacers": "IND",
    "Milwaukee Bucks": "MIL", "Atlanta Hawks": "ATL",
    "Oklahoma City Thunder": "OKC", "New York Knicks": "NYK",
    "Boston Celtics": "BOS", "Charlotte Hornets": "CHA",
    "Utah Jazz": "UTA", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Sacramento Kings": "SAC",
    "Miami Heat": "MIA", "Cleveland Cavaliers": "CLE",
    "Denver Nuggets": "DEN", "Minnesota Timberwolves": "MIN",
    "Los Angeles Lakers": "LAL", "Golden State Warriors": "GSW",
    "Dallas Mavericks": "DAL", "Memphis Grizzlies": "MEM",
    "New Orleans Pelicans": "NOP", "San Antonio Spurs": "SAS",
    "Portland Trail Blazers": "POR", "Houston Rockets": "HOU",
    "Chicago Bulls": "CHI", "Detroit Pistons": "DET",
    "Toronto Raptors": "TOR", "Brooklyn Nets": "BKN",
    "Washington Wizards": "WAS", "Orlando Magic": "ORL",
}

def _team_abbrev(name: str) -> str:
    return TEAM_NAME_TO_ABBREV.get(name, name[:3].upper())


def _track_usage(requests_remaining: int):
    try:
        usage = {"requests_remaining": requests_remaining, "last_updated": datetime.now(timezone.utc).isoformat()}
        if USAGE_FILE.exists():
            existing = json.loads(USAGE_FILE.read_text())
            existing.update(usage)
            USAGE_FILE.write_text(json.dumps(existing, indent=2))
        else:
            USAGE_FILE.write_text(json.dumps(usage, indent=2))
    except Exception:
        pass


def get_live_dk_lines() -> List[Dict[str, Any]]:
    """
    Fetch live DraftKings spread + total for all in-progress NBA games.
    Uses The Odds API live endpoint.
    """
    if not ODDS_API_KEY:
        log.error("No ODDS_API_KEY found")
        return []

    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "spreads,totals",
                "oddsFormat": "american",
                "bookmakers": "draftkings",
            },
            timeout=10,
        )
        remaining = int(r.headers.get("x-requests-remaining", -1))
        if remaining >= 0:
            _track_usage(remaining)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        log.error(f"Odds API fetch failed: {e}")
        return []

    lines = []
    for event in events:
        home = _team_abbrev(event.get("home_team", ""))
        away = _team_abbrev(event.get("away_team", ""))
        commence = event.get("commence_time", "")

        result = {
            "home": home, "away": away,
            "commence_time": commence,
            "spread": None, "total": None,
            "home_spread_odds": None, "away_spread_odds": None,
        }

        for bookmaker in event.get("bookmakers", []):
            if bookmaker.get("key") != "draftkings":
                continue
            for market in bookmaker.get("markets", []):
                mkt_key = market.get("key", "")
                if mkt_key == "spreads":
                    for outcome in market.get("outcomes", []):
                        team_abbrev = _team_abbrev(outcome.get("name", ""))
                        if team_abbrev == home:
                            result["spread"] = outcome.get("point")
                            result["home_spread_odds"] = outcome.get("price")
                        elif team_abbrev == away:
                            result["away_spread_odds"] = outcome.get("price")
                elif mkt_key == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Over":
                            result["total"] = outcome.get("point")

        lines.append(result)

    return lines


def get_live_line_for_game(home: str, away: str) -> Optional[Dict[str, Any]]:
    """
    Get DK live spread + total for a specific game.
    """
    all_lines = get_live_dk_lines()
    h = home.upper()
    a = away.upper()
    for line in all_lines:
        if {line.get("home"), line.get("away")} == {h, a}:
            return line
    return None


def save_live_snapshot(home: str, away: str, snapshot_type: str = "live") -> Optional[str]:
    """
    Save a live odds snapshot to disk.
    snapshot_type: "live" (called during game)
    Returns filepath if saved, None if game not found.
    """
    line = get_live_line_for_game(home, away)
    if not line:
        log.warning(f"No live line found for {away}@{home}")
        return None

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    fname = f"{date_str}_{away}_{home}_{snapshot_type}_{now.strftime('%H%M')}.json"
    fpath = DATA_DIR / fname

    line["snapshot_type"] = snapshot_type
    line["snapshot_time"] = now.isoformat()
    fpath.write_text(json.dumps(line, indent=2))
    log.info(f"Saved live odds snapshot: {fpath}")
    return str(fpath)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        home, away = sys.argv[1], sys.argv[2]
        line = get_live_line_for_game(home, away)
        if line:
            print(f"{away} @ {home}")
            print(f"  DK Spread: {home} {line.get('spread'):+.1f}")
            print(f"  DK Total: {line.get('total')}")
        else:
            print(f"No live line found for {away}@{home}")
    else:
        lines = get_live_dk_lines()
        print(f"\n=== DraftKings Live Lines ({datetime.now().strftime('%H:%M')}) ===")
        for l in lines:
            spread_str = f"{l['home']} {l['spread']:+.1f}" if l['spread'] is not None else "N/A"
            total_str = f"{l['total']}" if l['total'] else "N/A"
            print(f"  {l['away']} @ {l['home']} | Spread: {spread_str} | Total: {total_str}")

"""
Line movement tracker — logs opening and closing lines for each game.

This is the #1 missing signal from the BOS vs CHA disaster:
  If sharp money moved BOS from -8.5 → -6.5, we should have seen it.

Usage:
- Run at 9am ET each day to log OPENING lines
- Run 30min before tip to log CLOSING lines
- Compute movement = close - open (negative = line moved toward underdog = sharp action on dog)
- Store in data/line_movement/{date}_{away}_{home}.json

Line movement signals:
  - Line moves AWAY from favorite (-8 → -5.5) = sharp money on underdog
  - Line moves TOWARD favorite (-5 → -7.5) = sharp money on favorite
  - Large movement (>2 pts) = significant sharp action
  - Reverse line movement: public on favorite but line goes other way = biggest tell
"""
import json
import time
import logging
import requests
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

log = logging.getLogger(__name__)

LINE_MOVEMENT_DIR = DATA_DIR / "line_movement"
LINE_MOVEMENT_DIR.mkdir(exist_ok=True)

ODDS_API_KEY = None  # loaded on first use


def _get_api_key() -> str:
    global ODDS_API_KEY
    if ODDS_API_KEY:
        return ODDS_API_KEY
    env_path = Path.home() / ".openclaw" / "credentials" / "jarvis-github.env"
    for line in env_path.read_text().splitlines():
        if line.startswith("ODDS_API_KEY="):
            ODDS_API_KEY = line.split("=", 1)[1].strip()
            return ODDS_API_KEY
    raise ValueError("ODDS_API_KEY not found in credentials")


TEAM_NAME_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def fetch_current_lines() -> list:
    """Fetch current NBA odds from The Odds API."""
    key = _get_api_key()
    url = (f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
           f"?apiKey={key}&regions=us&markets=spreads,totals&oddsFormat=american")
    resp = requests.get(url, timeout=10)
    remaining = resp.headers.get("x-requests-remaining", "?")
    log.info(f"Odds API requests remaining: {remaining}")

    if resp.status_code != 200:
        log.error(f"Odds API error: {resp.status_code}")
        return []

    games = []
    for g in resp.json():
        home = TEAM_NAME_MAP.get(g["home_team"], g["home_team"])
        away = TEAM_NAME_MAP.get(g["away_team"], g["away_team"])
        game_date = g["commence_time"][:10]

        spread, total = None, None
        for bm in g.get("bookmakers", []):
            if bm["key"] == "draftkings":
                for mkt in bm.get("markets", []):
                    mkt_key = mkt.get("key", mkt.get("type", ""))
                    if mkt_key == "spreads":
                        for outcome in mkt.get("outcomes", []):
                            if TEAM_NAME_MAP.get(outcome["name"], outcome["name"]) == home:
                                spread = outcome["point"]
                    elif mkt_key == "totals":
                        for outcome in mkt.get("outcomes", []):
                            if outcome["name"] == "Over":
                                total = outcome["point"]
                break

        if spread is not None or total is not None:
            games.append({
                "game_date": game_date,
                "home": home, "away": away,
                "spread": spread, "total": total,
                "timestamp": datetime.utcnow().isoformat(),
            })
    return games


def log_lines(snapshot_type: str = "open"):
    """
    Log current lines as opening or closing snapshot.
    snapshot_type: 'open' (morning) or 'close' (pre-tip)
    """
    games = fetch_current_lines()
    today = date.today().strftime("%Y-%m-%d")

    for g in games:
        if g["game_date"] != today:
            continue
        key = f"{g['away']}_{g['home']}"
        path = LINE_MOVEMENT_DIR / f"{today}_{key}.json"

        # Load existing or create new
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = {"date": today, "home": g["home"], "away": g["away"],
                    "open": None, "close": None, "movement": None}

        data[snapshot_type] = {
            "spread": g["spread"], "total": g["total"],
            "timestamp": g["timestamp"],
        }

        # Compute movement if we have both
        if data["open"] and data["close"]:
            open_spread = data["open"]["spread"]
            close_spread = data["close"]["spread"]
            if open_spread is not None and close_spread is not None:
                movement = close_spread - open_spread
                data["movement"] = {
                    "spread_change": round(movement, 1),
                    "direction": "toward_favorite" if movement < 0 else "toward_underdog",
                    "significant": abs(movement) >= 2.0,
                    "sharp_signal": abs(movement) >= 1.5,
                }

        path.write_text(json.dumps(data, indent=2))
        log.info(f"Logged {snapshot_type} lines: {g['away']}@{g['home']} spread={g['spread']} total={g['total']}")

    log.info(f"Logged {len(games)} games ({snapshot_type} lines)")
    return games


def get_line_movement(home: str, away: str, game_date: str) -> Optional[dict]:
    """Get line movement data for a specific game."""
    path = LINE_MOVEMENT_DIR / f"{game_date}_{away}_{home}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("movement")


def get_sharp_flags(game_date: str) -> list:
    """
    Return all games today with significant line movement (sharp action).
    These are the games to fade the public on.
    """
    flags = []
    for path in LINE_MOVEMENT_DIR.glob(f"{game_date}_*.json"):
        data = json.loads(path.read_text())
        if data.get("movement", {}).get("significant"):
            mv = data["movement"]
            flags.append({
                "game": f"{data['away']}@{data['home']}",
                "spread_change": mv["spread_change"],
                "direction": mv["direction"],
                "interpretation": (
                    f"Line moved {mv['spread_change']:+.1f} pts — "
                    f"{'SHARP ON UNDERDOG 🔥' if mv['direction']=='toward_underdog' else 'SHARP ON FAVORITE'}"
                ),
            })
    return flags


def print_movement_report(game_date: str):
    """Print today's line movement summary."""
    print(f"\n📊 LINE MOVEMENT REPORT — {game_date}")
    print("=" * 60)
    flags = get_sharp_flags(game_date)
    if not flags:
        # Show all games
        for path in LINE_MOVEMENT_DIR.glob(f"{game_date}_*.json"):
            data = json.loads(path.read_text())
            o = data.get("open", {})
            c = data.get("close", {})
            mv = data.get("movement")
            print(f"\n  {data['away']}@{data['home']}")
            if o:
                print(f"    Open:  spread={o.get('spread')} | total={o.get('total')}")
            if c:
                print(f"    Close: spread={c.get('spread')} | total={c.get('total')}")
            if mv:
                print(f"    Movement: {mv['spread_change']:+.1f} pts → {mv['direction']}")
    else:
        print("🚨 SHARP ACTION DETECTED:")
        for f in flags:
            print(f"\n  {f['game']}: {f['interpretation']}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    action = sys.argv[1] if len(sys.argv) > 1 else "open"
    print(f"Logging {action} lines...")
    games = log_lines(action)
    print_movement_report(date.today().strftime("%Y-%m-%d"))

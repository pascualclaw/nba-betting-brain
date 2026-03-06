"""
Live Odds Collector — pulls true in-play odds via Odds API event-level endpoint.
Unlike the scoreboard endpoint, this reflects real-time live lines mid-game.
"""
import os, json, time
from urllib.request import urlopen
from urllib.parse import urlencode
from datetime import datetime, timezone, timedelta

API_KEY = os.getenv("ODDS_API_KEY", "")
BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
BOOKS = ["draftkings"]
MARKETS = "h2h,spreads,totals"

# ESPN abbreviation → full team name (for matching Odds API team names)
ESPN_TO_FULL = {
    "CHI": "Chicago Bulls", "PHX": "Phoenix Suns",
    "LAL": "Los Angeles Lakers", "DEN": "Denver Nuggets",
    "NOP": "New Orleans Pelicans", "NO": "New Orleans Pelicans",
    "SAC": "Sacramento Kings", "GS": "Golden State Warriors",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets",
    "BOS": "Boston Celtics", "MIL": "Milwaukee Bucks",
    "MIA": "Miami Heat", "NYK": "New York Knicks",
    "BKN": "Brooklyn Nets", "TOR": "Toronto Raptors",
    "CLE": "Cleveland Cavaliers", "IND": "Indiana Pacers",
    "ORL": "Orlando Magic", "ATL": "Atlanta Hawks",
    "WAS": "Washington Wizards", "WSH": "Washington Wizards",
    "DAL": "Dallas Mavericks", "OKC": "Oklahoma City Thunder",
    "MIN": "Minnesota Timberwolves", "UTA": "Utah Jazz",
    "POR": "Portland Trail Blazers", "MEM": "Memphis Grizzlies",
    "SAS": "San Antonio Spurs", "SA": "San Antonio Spurs",
}


def _fetch(url: str) -> dict | list:
    with urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def get_live_events() -> list[dict]:
    """List all in-progress NBA events with their Odds API IDs."""
    url = f"{BASE}/sports/{SPORT}/events?apiKey={API_KEY}"
    events = _fetch(url)
    now = datetime.now(timezone.utc)
    # Events that started in the last 4 hours = likely live
    live = []
    for e in events:
        ct = e.get("commence_time", "")
        if ct:
            try:
                start = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                hours_ago = (now - start).total_seconds() / 3600
                if 0 <= hours_ago <= 4:
                    live.append(e)
            except Exception:
                pass
    return live


def get_live_odds(event_id: str, books: list[str] = BOOKS, markets: str = MARKETS) -> dict:
    """Pull live DK odds for a specific event. Returns structured dict."""
    params = urlencode({
        "apiKey": API_KEY,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": ",".join(books),
    })
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds?{params}"
    data = _fetch(url)

    result = {
        "event_id": event_id,
        "home": data.get("home_team", ""),
        "away": data.get("away_team", ""),
        "commence_time": data.get("commence_time", ""),
        "bookmakers": {},
    }

    for bk in data.get("bookmakers", []):
        key = bk.get("key", "")
        book_data = {"last_update": bk.get("last_update"), "markets": {}}
        for mkt in bk.get("markets", []):
            mname = mkt.get("key", "")
            outcomes = {}
            for o in mkt.get("outcomes", []):
                name = o.get("name", "")
                outcomes[name] = {
                    "price": o.get("price"),
                    "point": o.get("point"),
                }
            book_data["markets"][mname] = outcomes
        result["bookmakers"][key] = book_data

    return result


def get_all_live_odds(books: list[str] = BOOKS) -> list[dict]:
    """Pull live odds for all currently live NBA games."""
    events = get_live_events()
    results = []
    for e in events:
        try:
            odds = get_live_odds(e["id"], books=books)
            # Attach ESPN-style lookup
            odds["event_meta"] = e
            results.append(odds)
            time.sleep(0.2)  # gentle on API
        except Exception as ex:
            print(f"Error fetching {e.get('id')}: {ex}")
    return results


def print_live_snapshot(books: list[str] = BOOKS):
    """Print a formatted snapshot of all live NBA odds."""
    all_odds = get_all_live_odds(books)
    if not all_odds:
        print("No live NBA games found.")
        return

    for game in all_odds:
        print(f"\n{'='*55}")
        print(f"  {game['away']} @ {game['home']}")
        for bkname, bkdata in game.get("bookmakers", {}).items():
            print(f"  [{bkname.upper()} | updated: {bkdata.get('last_update')}]")
            mkts = bkdata.get("markets", {})
            # ML
            if "h2h" in mkts:
                for team, data in mkts["h2h"].items():
                    short = team.split()[-1]
                    print(f"    ML  {short}: {data['price']:+d}")
            # Spread
            if "spreads" in mkts:
                for team, data in mkts["spreads"].items():
                    short = team.split()[-1]
                    print(f"    SPR {short}: {data.get('point','')} @ {data['price']:+d}")
            # Total
            if "totals" in mkts:
                for side, data in mkts["totals"].items():
                    print(f"    TOT {side}: {data.get('point','')} @ {data['price']:+d}")


def ev_check(price: int, true_prob: float) -> float:
    """Compute EV given American odds and estimated true probability."""
    if price > 0:
        payout = price / 100
    else:
        payout = 100 / abs(price)
    return true_prob * payout - (1 - true_prob)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "event":
        eid = sys.argv[2]
        odds = get_live_odds(eid)
        print(json.dumps(odds, indent=2))
    else:
        print_live_snapshot()

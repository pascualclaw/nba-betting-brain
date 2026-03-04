"""
Odds API collector — pulls live betting lines from The Odds API.
Free tier: 500 requests/month. Sign up at https://the-odds-api.com

Set your key in env: ODDS_API_KEY=your_key_here
Or add to ~/.openclaw/credentials/odds_api.env

This gives us:
- Live spreads, moneylines, totals from all major books
- Player props (DraftKings, FanDuel, BetMGM, etc.)
- Line movement by comparing snapshots over time
"""
import os
import json
import requests
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
ODDS_DIR = DATA_DIR / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Load API key
def get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        creds_path = Path.home() / ".openclaw/credentials/odds_api.env"
        if creds_path.exists():
            for line in creds_path.read_text().splitlines():
                if line.startswith("ODDS_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    return key


def get_live_odds(markets: str = "h2h,spreads,totals", bookmakers: str = None) -> list:
    """
    Pull live odds for all NBA games today.
    markets: comma-separated from h2h, spreads, totals
    bookmakers: comma-sep e.g. "draftkings,fanduel,betmgm" (None = all)
    """
    api_key = get_api_key()
    if not api_key:
        return [{"error": "No ODDS_API_KEY found. Get one free at https://the-odds-api.com"}]

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    r = requests.get(f"{BASE_URL}/sports/{SPORT}/odds", params=params, timeout=15)
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"Odds API requests remaining: {remaining}")
    r.raise_for_status()
    return r.json()


def get_player_props(event_id: str, markets: str = "player_points,player_rebounds,player_assists,player_threes") -> dict:
    """
    Pull player props for a specific game event.
    Get event_id from get_live_odds() → game['id']
    """
    api_key = get_api_key()
    if not api_key:
        return {"error": "No ODDS_API_KEY"}

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm",
    }
    r = requests.get(f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds", params=params, timeout=15)
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"Odds API requests remaining: {remaining}")
    r.raise_for_status()
    return r.json()


def parse_game_lines(games: list) -> list:
    """Parse odds API response into clean game line summaries."""
    results = []
    for g in games:
        game_info = {
            "id": g["id"],
            "home": g["home_team"],
            "away": g["away_team"],
            "commence_time": g["commence_time"],
            "lines": {},
        }
        for book in g.get("bookmakers", []):
            book_name = book["key"]
            game_info["lines"][book_name] = {}
            for market in book.get("markets", []):
                mkey = market["key"]
                outcomes = {o["name"]: o for o in market["outcomes"]}
                if mkey == "totals":
                    over = outcomes.get("Over", {})
                    game_info["lines"][book_name]["total"] = {
                        "line": over.get("point"),
                        "over_odds": over.get("price"),
                        "under_odds": outcomes.get("Under", {}).get("price"),
                    }
                elif mkey == "spreads":
                    home = outcomes.get(g["home_team"], {})
                    game_info["lines"][book_name]["spread"] = {
                        "home_spread": home.get("point"),
                        "home_odds": home.get("price"),
                        "away_spread": outcomes.get(g["away_team"], {}).get("point"),
                        "away_odds": outcomes.get(g["away_team"], {}).get("price"),
                    }
                elif mkey == "h2h":
                    game_info["lines"][book_name]["moneyline"] = {
                        "home": outcomes.get(g["home_team"], {}).get("price"),
                        "away": outcomes.get(g["away_team"], {}).get("price"),
                    }
        results.append(game_info)
    return results


def get_best_line(game_lines: dict, market: str) -> dict:
    """Find the best available line across all books."""
    best = {}
    for book, lines in game_lines.get("lines", {}).items():
        if market in lines:
            best[book] = lines[market]
    return best


def parse_player_props(event_data: dict) -> dict:
    """Parse player props into a clean player → prop → line structure."""
    props = {}
    for book in event_data.get("bookmakers", []):
        for market in book.get("markets", []):
            prop_type = market["key"]  # e.g. "player_points"
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", outcome.get("name", ""))
                direction = "over" if outcome["name"].lower() == "over" else "under"
                point = outcome.get("point", 0)
                price = outcome.get("price", 0)

                if player not in props:
                    props[player] = {}
                if prop_type not in props[player]:
                    props[player][prop_type] = {}
                props[player][prop_type][direction] = {
                    "line": point,
                    "odds": price,
                    "book": book["key"],
                }
    return props


def find_line_discrepancies(props: dict, threshold: float = 0.5) -> list:
    """
    Find player props where line differs meaningfully across books.
    Discrepancies = potential mispricing = edge.
    """
    discrepancies = []
    # Would compare same prop across multiple books here
    # Placeholder for multi-book comparison logic
    return discrepancies


def save_odds_snapshot(games: list, label: str = ""):
    """Save an odds snapshot for line movement tracking."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"odds_{ts}_{label}.json" if label else f"odds_{ts}.json"
    path = ODDS_DIR / fname
    with open(path, "w") as f:
        json.dump(games, f, indent=2)
    print(f"Saved odds snapshot: {path.name}")


def get_line_movement(team1: str, team2: str) -> dict:
    """
    Compare oldest vs newest odds snapshot for a game.
    Shows how the line has moved (sharp money signal).
    """
    snapshots = sorted(ODDS_DIR.glob("odds_*.json"))
    if len(snapshots) < 2:
        return {"error": "Need at least 2 snapshots for line movement"}

    def find_game(snapshot_path):
        with open(snapshot_path) as f:
            games = json.load(f)
        for g in games:
            teams = {g.get("home", ""), g.get("away", "")}
            if team1.upper() in str(teams).upper() and team2.upper() in str(teams).upper():
                return g
        return None

    early = find_game(snapshots[0])
    latest = find_game(snapshots[-1])
    if not early or not latest:
        return {"error": f"No game found for {team1} vs {team2}"}

    movement = {
        "game": f"{team1} vs {team2}",
        "from_time": snapshots[0].stem,
        "to_time": snapshots[-1].stem,
        "changes": {},
    }

    for book in set(list(early.get("lines", {}).keys()) + list(latest.get("lines", {}).keys())):
        e_total = early.get("lines", {}).get(book, {}).get("total", {})
        l_total = latest.get("lines", {}).get(book, {}).get("total", {})
        if e_total and l_total and e_total.get("line") != l_total.get("line"):
            movement["changes"][book] = {
                "total": {"from": e_total.get("line"), "to": l_total.get("line")},
            }
        e_spread = early.get("lines", {}).get(book, {}).get("spread", {})
        l_spread = latest.get("lines", {}).get(book, {}).get("spread", {})
        if e_spread and l_spread and e_spread.get("home_spread") != l_spread.get("home_spread"):
            if book not in movement["changes"]:
                movement["changes"][book] = {}
            movement["changes"][book]["spread"] = {
                "from": e_spread.get("home_spread"),
                "to": l_spread.get("home_spread"),
            }

    return movement


if __name__ == "__main__":
    print("Testing Odds API...")
    games_raw = get_live_odds()
    if games_raw and "error" not in games_raw[0] if games_raw else True:
        games = parse_game_lines(games_raw)
        save_odds_snapshot(games, "test")
        for g in games[:3]:
            print(f"\n{g['away']} @ {g['home']}")
            for book, lines in g["lines"].items():
                if "total" in lines:
                    t = lines["total"]
                    print(f"  {book}: Total {t['line']} (O{t['over_odds']}/U{t['under_odds']})")
    else:
        print("Get a free API key at https://the-odds-api.com to enable live lines.")
        print("Set ODDS_API_KEY in ~/.openclaw/credentials/odds_api.env")

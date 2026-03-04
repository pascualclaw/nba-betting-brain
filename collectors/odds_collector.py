"""
Odds Collector — fetches NBA betting lines from The Odds API.

Endpoints used:
  - GET /v4/sports/basketball_nba/odds/  (today's lines)
  - GET /v4/sports/basketball_nba/odds-history/  (historical, may require paid tier)

Free tier: 500 requests/month. Key stored in ~/.openclaw/credentials/odds_api.env

Output: data/odds/{YYYY-MM-DD}_{HOME}_{AWAY}_odds.json  per game
Usage tracking: data/odds_api_usage.json
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
ODDS_DIR = _ROOT / "data" / "odds"
USAGE_FILE = _ROOT / "data" / "odds_api_usage.json"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

# ── API constants ──────────────────────────────────────────────────────────────
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# ── Full team name → NBA tricode mapping ──────────────────────────────────────
# The Odds API uses full city+name strings (e.g. "Phoenix Suns").
TEAM_NAME_TO_TRICODE: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def name_to_tricode(full_name: str) -> str:
    """Map a full team name to its NBA tricode. Falls back to first 3 chars uppercased."""
    return TEAM_NAME_TO_TRICODE.get(full_name, full_name[:3].upper())


# ── API key loading ────────────────────────────────────────────────────────────

def _load_api_key() -> Optional[str]:
    """Load ODDS_API_KEY from env or credentials file."""
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key.strip()
    creds = Path.home() / ".openclaw" / "credentials" / "odds_api.env"
    if creds.exists():
        for line in creds.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


# ── API usage tracking ─────────────────────────────────────────────────────────

def _update_usage(remaining: str, used: str = "") -> None:
    """Persist remaining API quota to disk."""
    usage: dict = {}
    if USAGE_FILE.exists():
        try:
            usage = json.loads(USAGE_FILE.read_text())
        except json.JSONDecodeError:
            usage = {}
    usage["last_checked"] = datetime.now().isoformat()
    if remaining not in ("?", ""):
        usage["requests_remaining"] = int(remaining)
    if used not in ("?", ""):
        usage["requests_used"] = int(used)
    USAGE_FILE.write_text(json.dumps(usage, indent=2))


# ── Today's lines ─────────────────────────────────────────────────────────────

def fetch_today_odds(
    markets: str = "totals,spreads,h2h",
    regions: str = "us",
    odds_format: str = "american",
) -> list[dict]:
    """
    Fetch today's NBA game lines from The Odds API.

    Args:
        markets:      Comma-separated markets: totals, spreads, h2h
        regions:      Betting region: us, uk, eu, au
        odds_format:  american | decimal | fractional

    Returns:
        Raw JSON list of game objects from the API, or [] on failure.
    """
    api_key = _load_api_key()
    if not api_key:
        log.error("No ODDS_API_KEY found. Set in ~/.openclaw/credentials/odds_api.env")
        return []

    url = f"{BASE_URL}/sports/{SPORT}/odds/"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        _update_usage(remaining, used)
        log.info(f"Odds API — requests remaining: {remaining} | used: {used}")

        if resp.status_code == 401:
            log.error("Odds API: 401 Unauthorized — invalid or missing API key.")
            return []
        if resp.status_code == 429:
            log.error("Odds API: 429 Rate limit exceeded (500/month free tier).")
            return []
        resp.raise_for_status()
        return resp.json()

    except requests.Timeout:
        log.error("Odds API: request timed out.")
    except requests.RequestException as exc:
        log.error(f"Odds API request error: {exc}")
    return []


# ── Historical odds (free-tier check) ─────────────────────────────────────────

def fetch_historical_odds(date_str: str) -> list[dict]:
    """
    Attempt to pull historical NBA odds for a past date.
    NOTE: Historical endpoint typically requires a paid Odds API subscription.
    This will gracefully log the tier restriction if unavailable.

    Args:
        date_str: ISO 8601 date string, e.g. "2026-03-01T00:00:00Z"

    Returns:
        List of game objects, or [] if unavailable.
    """
    api_key = _load_api_key()
    if not api_key:
        log.error("No ODDS_API_KEY for historical fetch.")
        return []

    url = f"{BASE_URL}/sports/{SPORT}/odds-history/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals,spreads",
        "oddsFormat": "american",
        "date": date_str,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        _update_usage(remaining)

        if resp.status_code == 401:
            log.warning("Historical odds: 401 — likely requires paid Odds API plan.")
            return []
        if resp.status_code == 422:
            log.warning(f"Historical odds: 422 Unprocessable — date format may be wrong ({date_str}).")
            return []
        if resp.status_code == 402:
            log.warning("Historical odds: 402 — requires paid tier. Free tier gives today's lines only.")
            return []
        resp.raise_for_status()
        data = resp.json()
        # API wraps in {"data": [...]} for historical
        if isinstance(data, dict):
            return data.get("data", [])
        return data

    except requests.RequestException as exc:
        log.error(f"Historical odds request error: {exc}")
    return []


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_game_lines(raw_games: list[dict]) -> list[dict]:
    """
    Parse raw Odds API game objects into clean, tricode-keyed records.

    Each game record contains:
        home_team, away_team, home_tricode, away_tricode,
        commence_time, open_total_line, open_spread,
        moneyline_home, moneyline_away, bookmakers (full detail)

    Strategy: average totals/spreads across all bookmakers available.
    """
    parsed = []

    for g in raw_games:
        home_full = g.get("home_team", "")
        away_full = g.get("away_team", "")
        home_code = name_to_tricode(home_full)
        away_code = name_to_tricode(away_full)

        totals: list[float] = []
        spreads: list[float] = []
        home_mls: list[int] = []
        away_mls: list[int] = []
        bookmakers_detail: dict[str, dict] = {}

        for book in g.get("bookmakers", []):
            bkey = book["key"]
            book_data: dict = {}
            for market in book.get("markets", []):
                mkey = market["key"]
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if mkey == "totals":
                    over = outcomes.get("Over", {})
                    point = over.get("point")
                    if point is not None:
                        totals.append(float(point))
                        book_data["total_line"] = float(point)
                        book_data["over_odds"] = over.get("price")
                        under = outcomes.get("Under", {})
                        book_data["under_odds"] = under.get("price")

                elif mkey == "spreads":
                    home_spread_outcome = outcomes.get(home_full, {})
                    point = home_spread_outcome.get("point")
                    if point is not None:
                        spreads.append(float(point))
                        book_data["home_spread"] = float(point)
                        book_data["home_spread_odds"] = home_spread_outcome.get("price")
                        away_spread_outcome = outcomes.get(away_full, {})
                        book_data["away_spread"] = away_spread_outcome.get("point")
                        book_data["away_spread_odds"] = away_spread_outcome.get("price")

                elif mkey == "h2h":
                    home_ml = outcomes.get(home_full, {}).get("price")
                    away_ml = outcomes.get(away_full, {}).get("price")
                    if home_ml is not None:
                        home_mls.append(int(home_ml))
                        book_data["moneyline_home"] = int(home_ml)
                    if away_ml is not None:
                        away_mls.append(int(away_ml))
                        book_data["moneyline_away"] = int(away_ml)

            if book_data:
                bookmakers_detail[bkey] = book_data

        def _avg(vals: list) -> Optional[float]:
            return round(sum(vals) / len(vals), 1) if vals else None

        parsed.append({
            "game_id": g.get("id"),
            "home_team": home_full,
            "away_team": away_full,
            "home_tricode": home_code,
            "away_tricode": away_code,
            "commence_time": g.get("commence_time"),
            "open_total_line": _avg(totals),      # avg across books = consensus open
            "open_spread": _avg(spreads),           # home spread avg
            "moneyline_home": _avg(home_mls),
            "moneyline_away": _avg(away_mls),
            "bookmakers": bookmakers_detail,
            "n_books": len(bookmakers_detail),
        })

    return parsed


# ── Saving ─────────────────────────────────────────────────────────────────────

def save_game_odds(game: dict, date_str: Optional[str] = None) -> Path:
    """
    Save a single game's odds to data/odds/{YYYY-MM-DD}_{HOME}_{AWAY}_odds.json.

    Args:
        game:     Parsed game dict from parse_game_lines().
        date_str: Override date (YYYY-MM-DD). Defaults to today.

    Returns:
        Path to saved file.
    """
    date = date_str or datetime.now().strftime("%Y-%m-%d")
    home = game["home_tricode"]
    away = game["away_tricode"]
    fname = f"{date}_{home}_{away}_odds.json"
    path = ODDS_DIR / fname
    with open(path, "w") as f:
        json.dump(game, f, indent=2)
    log.info(f"Saved: {fname}")
    return path


def fetch_and_save_today() -> list[dict]:
    """
    One-shot: fetch today's NBA odds and save per-game JSON files.

    Returns:
        List of parsed game dicts.
    """
    log.info("Fetching today's NBA odds...")
    raw = fetch_today_odds()
    if not raw:
        log.warning("No odds returned for today.")
        return []

    games = parse_game_lines(raw)
    today = datetime.now().strftime("%Y-%m-%d")
    for game in games:
        save_game_odds(game, date_str=today)

    log.info(f"Saved {len(games)} game odds files for {today}.")
    return games


# ── Lookup helpers ─────────────────────────────────────────────────────────────

def load_odds_for_game(home: str, away: str, date_str: Optional[str] = None) -> Optional[dict]:
    """
    Load saved odds JSON for a specific game.

    Args:
        home:     Home team tricode (e.g. "PHX")
        away:     Away team tricode (e.g. "SAC")
        date_str: YYYY-MM-DD; defaults to today.

    Returns:
        Odds dict or None if not found.
    """
    date = date_str or datetime.now().strftime("%Y-%m-%d")
    path = ODDS_DIR / f"{date}_{home}_{away}_odds.json"
    if path.exists():
        return json.loads(path.read_text())
    # Try reverse (sometimes home/away designation differs)
    path2 = ODDS_DIR / f"{date}_{away}_{home}_odds.json"
    if path2.exists():
        return json.loads(path2.read_text())
    return None


def load_all_odds_for_date(date_str: str) -> dict[str, dict]:
    """
    Load all saved odds for a given date.

    Returns:
        Dict keyed by "{HOME}_{AWAY}" → odds dict.
    """
    result: dict[str, dict] = {}
    for path in ODDS_DIR.glob(f"{date_str}_*_odds.json"):
        parts = path.stem.split("_")
        # Format: YYYY-MM-DD_HOME_AWAY_odds → parts after date are [HOME, AWAY, 'odds']
        if len(parts) >= 4:
            home = parts[3]
            away = parts[4] if len(parts) > 4 else ""
            key = f"{home}_{away}"
            try:
                result[key] = json.loads(path.read_text())
            except json.JSONDecodeError:
                log.warning(f"Failed to parse {path}")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("\n🏀 NBA Odds Collector")
    print("=" * 60)

    # Fetch today's lines
    games = fetch_and_save_today()

    if not games:
        print("No NBA games found for today (or API error).")
        sys.exit(0)

    # Print clean summary table
    print(f"\n{'Matchup':<28} {'Total':>7} {'Spread':>8} {'ML Home':>9} {'ML Away':>9} {'Books':>6}")
    print("-" * 72)
    for g in games:
        matchup = f"{g['away_tricode']} @ {g['home_tricode']}"
        total = f"{g['open_total_line']}" if g["open_total_line"] else "N/A"
        spread = f"{g['open_spread']:+.1f}" if g["open_spread"] is not None else "N/A"
        ml_home = f"{g['moneyline_home']:+.0f}" if g["moneyline_home"] is not None else "N/A"
        ml_away = f"{g['moneyline_away']:+.0f}" if g["moneyline_away"] is not None else "N/A"
        books = g["n_books"]
        print(f"{matchup:<28} {total:>7} {spread:>8} {ml_home:>9} {ml_away:>9} {books:>6}")

    # Check usage
    if USAGE_FILE.exists():
        usage = json.loads(USAGE_FILE.read_text())
        remaining = usage.get("requests_remaining", "?")
        print(f"\n📊 Odds API requests remaining this month: {remaining}")

    # Try historical endpoint (sanity check — likely 402 on free tier)
    print("\n🔍 Checking historical odds endpoint availability...")
    hist = fetch_historical_odds("2026-03-01T00:00:00Z")
    if hist:
        print(f"  ✅ Historical endpoint works! Got {len(hist)} games.")
    else:
        print("  ℹ️  Historical endpoint unavailable (requires paid plan — expected on free tier).")

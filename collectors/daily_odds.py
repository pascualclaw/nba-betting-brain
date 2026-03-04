"""
Daily odds printer — fetches today's NBA lines and prints a clean table.
Called from daily_run.py each morning (or standalone).

Usage:
    python collectors/daily_odds.py
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.odds_collector import (
    fetch_and_save_today,
    load_all_odds_for_date,
    USAGE_FILE,
)

logging.basicConfig(level=logging.WARNING)  # quiet for daily use


def print_odds_table(games: list[dict]) -> None:
    """Pretty-print today's NBA lines as a console table."""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n🏀 NBA Lines for {today}")
    print("=" * 76)
    print(f"  {'Matchup':<26} {'Total':>7} {'Spread':>8} {'ML Home':>9} {'ML Away':>9} {'Books':>5}")
    print("  " + "-" * 68)

    for g in games:
        matchup = f"{g['away_tricode']} @ {g['home_tricode']}"
        total = f"{g['open_total_line']}" if g.get("open_total_line") is not None else "N/A"
        spread_val = g.get("open_spread")
        spread = f"{spread_val:+.1f}" if spread_val is not None else "N/A"
        ml_home_val = g.get("moneyline_home")
        ml_home = f"{ml_home_val:+.0f}" if ml_home_val is not None else "N/A"
        ml_away_val = g.get("moneyline_away")
        ml_away = f"{ml_away_val:+.0f}" if ml_away_val is not None else "N/A"
        books = g.get("n_books", 0)

        print(f"  {matchup:<26} {total:>7} {spread:>8} {ml_home:>9} {ml_away:>9} {books:>5}")

    print("=" * 76)

    # Show API quota
    if USAGE_FILE.exists():
        try:
            usage = json.loads(USAGE_FILE.read_text())
            remaining = usage.get("requests_remaining", "?")
            used = usage.get("requests_used", "?")
            print(f"  📊 Odds API quota: {remaining} requests remaining | {used} used this month")
        except (json.JSONDecodeError, KeyError):
            pass

    print()


def run_daily_odds() -> list[dict]:
    """
    Fetch (or reload from cache) today's NBA odds and print them.

    Strategy: try loading cached files first (already fetched today),
    then fall back to live API call.

    Returns:
        List of parsed game dicts.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    cached = load_all_odds_for_date(today)

    if cached:
        games = list(cached.values())
        print(f"  (Loaded {len(games)} games from cache)")
    else:
        games = fetch_and_save_today()

    if games:
        print_odds_table(games)
    else:
        print(f"  ⚠️  No NBA games found for {today} (API may be unavailable or no games scheduled).")

    return games


if __name__ == "__main__":
    run_daily_odds()

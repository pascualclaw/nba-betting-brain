"""
Daily run script — pull yesterday's results, score props, update H2H, log lessons.
Run this every morning via cron.
"""
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collectors.nba_api import (
    get_live_scoreboard, get_game_summary, get_live_boxscore,
    get_player_stats, get_pace_analysis, save_game_result
)
from collectors.h2h_collector import add_game_to_h2h, compute_h2h_stats
from trackers.prop_tracker import get_hit_rates, get_lessons

DATA_DIR = Path(__file__).parent / "data"


def run_daily_update():
    print(f"\n{'='*60}")
    print(f"🏀 NBA BETTING BRAIN — Daily Update")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}\n")

    # 1. Pull today's scoreboard
    print("📡 Fetching scoreboard...")
    try:
        games = get_live_scoreboard()
        summaries = get_game_summary(games)
    except Exception as e:
        print(f"❌ Failed to fetch scoreboard: {e}")
        return

    final_games = [g for g in summaries if g["is_final"]]
    live_games = [g for g in summaries if g["is_live"]]
    upcoming = [g for g in summaries if not g["is_final"] and not g["is_live"]]

    print(f"   Final: {len(final_games)} | Live: {len(live_games)} | Upcoming: {len(upcoming)}\n")

    # 2. Process completed games
    for game in final_games:
        print(f"✅ Processing: {game['away_team']} @ {game['home_team']} — {game['away_score']}-{game['home_score']}")
        
        # Compute pace
        pace = get_pace_analysis(game)
        
        # Build game record
        winner = game["away_team"] if game["away_score"] > game["home_score"] else game["home_team"]
        record = {
            "game_id": game.get("game_id"),
            "game_code": game["game_code"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "home": game["home_team"],
            "away": game["away_team"],
            "home_score": game["home_score"],
            "away_score": game["away_score"],
            "total": game["total"],
            "winner": winner,
            "quarters": pace["quarters"],
            "avg_per_quarter": pace["avg_per_quarter"],
            "hottest_quarter": pace["hottest_quarter"],
        }
        
        # Save result
        save_game_result(record)
        
        # Update H2H
        add_game_to_h2h(game["home_team"], game["away_team"], record)

    # 3. Show upcoming games
    if upcoming:
        print(f"\n📅 Today's upcoming games:")
        for g in upcoming:
            print(f"   {g['away_team']} @ {g['home_team']} — {g['status']}")

    # 4. Show current hit rates
    print(f"\n📊 Prop hit rates:")
    rates = get_hit_rates()
    if rates["total"] > 0:
        print(f"   Overall: {rates['hits']}/{rates['total']} ({rates.get('overall_hit_rate', 0)*100:.0f}%)")
        for ptype, data in rates["by_type"].items():
            print(f"   {ptype}: {data['hits']}/{data['total']} ({data['hit_rate']*100:.0f}%)")
    else:
        print("   No scored props yet.")

    # 5. Surface recent lessons
    print(f"\n📚 Most recent lessons:")
    lessons = get_lessons(limit=3)
    for l in lessons:
        print(f"   [{l['category']}] {l['what_happened'][:70]}...")
        print(f"   → {l['fix_for_next_time'][:70]}...")
        print()

    print(f"{'='*60}")
    print(f"✅ Daily update complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_daily_update()

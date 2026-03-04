"""
NBA data collector — pulls from NBA CDN (free, no API key needed).
"""
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

BASE = "https://cdn.nba.com/static/json/liveData"
STATS_BASE = "https://stats.nba.com/stats"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
}

DATA_DIR = Path(__file__).parent.parent / "data"


def get_live_scoreboard():
    """Get today's live scores and game status."""
    url = f"{BASE}/scoreboard/todaysScoreboard_00.json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()["scoreboard"]["games"]


def get_live_boxscore(game_id: str) -> dict:
    """Get live box score for a game by game_id."""
    url = f"{BASE}/boxscore/boxscore_{game_id}.json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()["game"]


def get_player_stats(game: dict) -> list:
    """Extract player stat lines from a live boxscore game object."""
    players = []
    for team_key in ["homeTeam", "awayTeam"]:
        team = game[team_key]
        for p in team["players"]:
            s = p["statistics"]
            players.append({
                "name": p["name"],
                "team": team["teamTricode"],
                "minutes": s.get("minutesCalculated", "PT0M"),
                "points": s.get("points", 0),
                "rebounds": s.get("reboundsTotal", 0),
                "assists": s.get("assists", 0),
                "steals": s.get("steals", 0),
                "blocks": s.get("blocks", 0),
                "turnovers": s.get("turnovers", 0),
                "fg_made": s.get("fieldGoalsMade", 0),
                "fg_attempted": s.get("fieldGoalsAttempted", 0),
                "three_made": s.get("threePointersMade", 0),
                "pra": s.get("points", 0) + s.get("reboundsTotal", 0) + s.get("assists", 0),
            })
    return players


def get_game_summary(games: list) -> list:
    """Parse scoreboard into clean game summaries."""
    summaries = []
    for g in games:
        h = g["homeTeam"]
        a = g["awayTeam"]
        summaries.append({
            "game_id": g["gameId"],
            "game_code": g["gameCode"],
            "status": g["gameStatusText"],
            "period": g["period"],
            "home_team": h["teamTricode"],
            "home_score": h["score"],
            "home_q": {p["period"]: p["score"] for p in h["periods"]},
            "away_team": a["teamTricode"],
            "away_score": a["score"],
            "away_q": {p["period"]: p["score"] for p in a["periods"]},
            "total": h["score"] + a["score"],
            "is_final": g["gameStatus"] == 3,
            "is_live": g["gameStatus"] == 2,
        })
    return summaries


def get_h2h_matchup_code(team1: str, team2: str) -> str:
    """Build a consistent H2H key regardless of home/away order."""
    return "_".join(sorted([team1.upper(), team2.upper()]))


def save_game_result(summary: dict):
    """Save a final game result to data/games/."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = DATA_DIR / "games" / f"{date_str}_{summary['game_code'].replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved game result: {path.name}")


def get_pace_analysis(summary: dict) -> dict:
    """Analyze quarter-by-quarter pace for a game."""
    h_q = summary["home_q"]
    a_q = summary["away_q"]
    quarters = {}
    for q in range(1, 5):
        combined = h_q.get(q, 0) + a_q.get(q, 0)
        quarters[f"Q{q}"] = combined
    total = summary["total"]
    return {
        "quarters": quarters,
        "total": total,
        "avg_per_quarter": total / max(len([v for v in quarters.values() if v > 0]), 1),
        "hottest_quarter": max(quarters, key=quarters.get),
        "coolest_quarter": min((k for k, v in quarters.items() if v > 0), key=quarters.get),
    }


if __name__ == "__main__":
    print("Fetching live scoreboard...")
    games = get_live_scoreboard()
    summaries = get_game_summary(games)
    for g in summaries:
        pace = get_pace_analysis(g)
        print(f"\n{g['away_team']} @ {g['home_team']} | {g['status']}")
        print(f"  Score: {g['away_score']}-{g['home_score']} | Total: {g['total']}")
        print(f"  Quarters: {pace['quarters']}")
        print(f"  Avg/Q: {pace['avg_per_quarter']:.1f} | Hottest: {pace['hottest_quarter']}")

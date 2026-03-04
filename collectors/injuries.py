"""
Injury report collector — pulls from ESPN game feed.
"""
import requests
from datetime import datetime

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

def get_injury_report(game_id: str = None) -> dict:
    """
    Pull injury report from ESPN.
    If game_id provided, gets injuries for that specific game.
    Otherwise returns all active injuries.
    """
    if game_id:
        url = f"{ESPN_BASE}/summary?event={game_id}"
    else:
        url = f"{ESPN_BASE}/injuries"

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def parse_game_injuries(espn_game_id: str) -> dict:
    """
    Parse injury report for a specific game into a clean format.
    Returns dict of {team: [player injury entries]}
    """
    try:
        data = get_injury_report(espn_game_id)
        injuries = {"home": [], "away": []}

        # Extract from injuries section if available
        if "injuries" in data:
            for side, key in [("home", "homeTeam"), ("away", "awayTeam")]:
                team_injuries = data.get("injuries", {}).get(key, [])
                for inj in team_injuries:
                    injuries[side].append({
                        "player": inj.get("athlete", {}).get("displayName", "Unknown"),
                        "status": inj.get("status", "Unknown"),
                        "detail": inj.get("longComment", ""),
                        "return_date": inj.get("fantasyStatus", {}).get("description", ""),
                    })
        return injuries
    except Exception as e:
        return {"error": str(e), "home": [], "away": []}


def format_injury_report(injuries: dict, home_team: str, away_team: str) -> str:
    """Format injury report for display."""
    lines = []
    for side, team in [("away", away_team), ("home", home_team)]:
        team_injuries = injuries.get(side, [])
        if team_injuries:
            lines.append(f"\n**{team} Injuries:**")
            for inj in team_injuries:
                lines.append(f"  - {inj['player']} ({inj['status']}) {inj['detail']}")
        else:
            lines.append(f"\n**{team}:** No reported injuries")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Testing injury collector...")
    # Test with today's games via scoreboard
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from nba_api import get_live_scoreboard, get_game_summary
    games = get_live_scoreboard()
    for g in get_game_summary(games):
        print(f"{g['away_team']} @ {g['home_team']}: {g['status']}")

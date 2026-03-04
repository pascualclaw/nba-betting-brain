"""
Rotowire scraper — injury news, player status updates, minute projections.
Rotowire is the gold standard for fantasy/betting injury intel.
"""
import requests
import json
import re
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"

ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news"


def get_all_injuries() -> dict:
    """
    Pull full NBA injury report from ESPN API.
    Returns dict of {team_tricode: [injury_entries]}
    """
    r = requests.get(ESPN_INJURY_URL, timeout=10)
    r.raise_for_status()
    data = r.json()

    injuries = {}
    for item in data.get("injuries", []):
        team = item.get("team", {})
        tricode = team.get("abbreviation", "UNK").upper()
        team_injuries = []
        for inj in item.get("injuries", []):
            athlete = inj.get("athlete", {})
            team_injuries.append({
                "player": athlete.get("displayName", ""),
                "position": athlete.get("position", {}).get("abbreviation", ""),
                "status": inj.get("status", ""),
                "type": inj.get("type", {}).get("description", ""),
                "detail": inj.get("longComment", ""),
                "return_date": inj.get("fantasyStatus", {}).get("description", ""),
                "dnp": inj.get("status", "").lower() in ["out", "doubtful"],
            })
        if team_injuries:
            injuries[tricode] = team_injuries
    return injuries


def get_team_injuries(team_tricode: str) -> list:
    """Get injuries for a specific team."""
    all_inj = get_all_injuries()
    return all_inj.get(team_tricode.upper(), [])


def get_notable_absences(team_tricode: str, min_role: str = "starter") -> list:
    """
    Get OUT/Doubtful players for a team.
    Returns list of player names who are confirmed out.
    """
    injuries = get_team_injuries(team_tricode)
    return [i for i in injuries if i["dnp"]]


def get_recent_news(team_tricode: str = None, limit: int = 20) -> list:
    """
    Pull recent NBA news from ESPN.
    Useful for catching late scratches and surprise announcements.
    """
    params = {"limit": limit}
    if team_tricode:
        # ESPN team IDs for news filtering
        params["teams"] = team_tricode

    r = requests.get(ESPN_NEWS_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    articles = []
    for item in data.get("articles", []):
        articles.append({
            "headline": item.get("headline", ""),
            "description": item.get("description", ""),
            "published": item.get("published", ""),
            "url": item.get("links", {}).get("web", {}).get("href", ""),
        })
    return articles


def get_minute_projections_from_news(team_tricode: str) -> dict:
    """
    Parse news headlines to infer minute projections.
    Looks for: "expected to start", "on a minutes restriction", "DNP", etc.
    """
    news = get_recent_news(limit=50)
    relevant = []
    team_upper = team_tricode.upper()

    for article in news:
        headline = article["headline"].lower()
        desc = article.get("description", "").lower()
        full_text = headline + " " + desc

        # Keywords that affect betting
        keywords = [
            "out tonight", "ruled out", "doubtful", "questionable",
            "minutes restriction", "load management", "day-to-day",
            "expected to return", "cleared", "available", "start"
        ]
        if any(k in full_text for k in keywords):
            relevant.append(article)

    return {"team": team_tricode, "relevant_news": relevant[:10]}


def format_injury_report(team1: str, team2: str) -> str:
    """Generate a formatted injury report for two teams."""
    lines = [f"## 🏥 Injury Report: {team1} vs {team2}"]

    for team in [team1, team2]:
        absences = get_notable_absences(team)
        all_inj = get_team_injuries(team)
        lines.append(f"\n**{team}:**")
        if not all_inj:
            lines.append("  ✅ No reported injuries")
        else:
            for inj in all_inj:
                status_emoji = "❌" if inj["dnp"] else "⚠️"
                lines.append(f"  {status_emoji} {inj['player']} ({inj['position']}) — {inj['status']}: {inj['type']}")
                if inj.get("return_date"):
                    lines.append(f"     Return: {inj['return_date']}")

        if absences:
            lines.append(f"\n  ⚠️ **IMPACT:** {len(absences)} key player(s) OUT — adjust props accordingly")

    return "\n".join(lines)


def assess_injury_impact(team_tricode: str, player_name: str) -> str:
    """
    When a star player is out, assess the impact on remaining players.
    Returns a string noting who benefits (opportunity players).
    """
    absences = get_notable_absences(team_tricode)
    absent_names = [a["player"] for a in absences]

    if player_name in absent_names:
        return f"⚠️ {player_name} is OUT. Look for usage/minutes boost in remaining roster."
    return f"✅ {player_name} is active (not on injury report)"


def save_injury_snapshot(injuries: dict):
    """Save injury snapshot for tracking changes."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / "injuries" / f"injuries_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(injuries, f, indent=2)
    print(f"Saved injury snapshot: {path.name}")


if __name__ == "__main__":
    print("Testing injury collector...\n")
    try:
        injuries = get_all_injuries()
        print(f"Got injury data for {len(injuries)} teams\n")
        
        # Show PHX and SAC as demo
        for team in ["PHX", "SAC"]:
            print(f"\n{team} injuries:")
            team_inj = injuries.get(team, [])
            if team_inj:
                for inj in team_inj:
                    print(f"  {inj['player']} — {inj['status']}: {inj['type']}")
            else:
                print("  None reported")
        
        save_injury_snapshot(injuries)
        print(f"\nFull report:\n{format_injury_report('PHX', 'SAC')}")
    except Exception as e:
        print(f"Error: {e}")

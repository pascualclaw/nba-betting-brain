"""
Advanced stats collector — pulls from stats.nba.com
Key metrics: PACE, ORTG, DRTG, lineup net ratings, shot quality

These are the metrics that should drive total projections, not vibes.
"""
import requests
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}

SEASON = "2025-26"

# Team tricode → NBA team ID
TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}


def get_team_advanced_stats() -> dict:
    """
    Pull team-level advanced stats: PACE, ORTG, DRTG, NET RTG.
    This is the foundation for projecting game totals mathematically.
    """
    url = "https://stats.nba.com/stats/leaguedashteamstats"
    params = {
        "Conference": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "GameScope": "",
        "GameSegment": "",
        "LastNGames": "0",
        "LeagueID": "00",
        "Location": "",
        "MeasureType": "Advanced",
        "Month": "0",
        "OpponentTeamID": "0",
        "Outcome": "",
        "PORound": "0",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": "0",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": SEASON,
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": "0",
        "TwoWay": "0",
        "VsConference": "",
        "VsDivision": "",
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    headers = data["resultSets"][0]["headers"]
    rows = data["resultSets"][0]["rowSet"]

    stats = {}
    for row in rows:
        d = dict(zip(headers, row))
        tricode = _id_to_tricode(d["TEAM_ID"])
        if tricode:
            stats[tricode] = {
                "team": d["TEAM_NAME"],
                "pace": round(d.get("PACE", 0), 1),
                "ortg": round(d.get("OFF_RATING", 0), 1),
                "drtg": round(d.get("DEF_RATING", 0), 1),
                "net_rtg": round(d.get("NET_RATING", 0), 1),
                "efg_pct": round(d.get("EFG_PCT", 0), 3),
                "ts_pct": round(d.get("TS_PCT", 0), 3),
                "ast_ratio": round(d.get("AST_RATIO", 0), 1),
                "oreb_pct": round(d.get("OREB_PCT", 0), 3),
                "dreb_pct": round(d.get("DREB_PCT", 0), 3),
                "tov_ratio": round(d.get("TM_TOV_PCT", 0), 3),
            }
    return stats


def get_player_advanced_stats(player_name: str = None) -> list:
    """
    Pull player advanced stats. Filter by name if provided.
    """
    url = "https://stats.nba.com/stats/leaguedashplayerstats"
    params = {
        "College": "", "Conference": "", "Country": "", "DateFrom": "",
        "DateTo": "", "Division": "", "DraftPick": "", "DraftYear": "",
        "GameScope": "", "GameSegment": "", "Height": "", "LastNGames": "0",
        "LeagueID": "00", "Location": "", "MeasureType": "Advanced",
        "Month": "0", "OpponentTeamID": "0", "Outcome": "",
        "PORound": "0", "PaceAdjust": "N", "PerMode": "PerGame",
        "Period": "0", "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "Season": SEASON,
        "SeasonSegment": "", "SeasonType": "Regular Season",
        "ShotClockRange": "", "StarterBench": "", "TeamID": "0",
        "TwoWay": "0", "VsConference": "", "VsDivision": "", "Weight": "",
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    headers = data["resultSets"][0]["headers"]
    rows = data["resultSets"][0]["rowSet"]
    players = [dict(zip(headers, row)) for row in rows]

    if player_name:
        players = [p for p in players if player_name.lower() in p.get("PLAYER_NAME", "").lower()]
    return players


def project_game_total(team1: str, team2: str, stats: dict = None) -> dict:
    """
    Project game total using ORTG/DRTG/PACE formula.
    
    Formula:
      Team1 pts = (Team1_ORTG × Team2_DRTG / League_avg_DRTG) × avg_PACE / 100
      Team2 pts = (Team2_ORTG × Team1_DRTG / League_avg_DRTG) × avg_PACE / 100
    
    This is how sharp bettors project totals — not vibes.
    """
    if stats is None:
        stats = get_team_advanced_stats()

    t1 = stats.get(team1.upper(), {})
    t2 = stats.get(team2.upper(), {})

    if not t1 or not t2:
        return {"error": f"Missing stats for {team1} or {team2}"}

    # League averages (approx 2025-26)
    league_avg_drtg = 113.5
    league_avg_ortg = 113.5
    avg_pace = (t1["pace"] + t2["pace"]) / 2

    # Projected points per team
    t1_proj = (t1["ortg"] * t2["drtg"] / league_avg_drtg) * avg_pace / 100
    t2_proj = (t2["ortg"] * t1["drtg"] / league_avg_drtg) * avg_pace / 100
    total_proj = t1_proj + t2_proj

    return {
        "team1": team1,
        "team2": team2,
        "team1_projected_pts": round(t1_proj, 1),
        "team2_projected_pts": round(t2_proj, 1),
        "projected_total": round(total_proj, 1),
        "avg_pace": round(avg_pace, 1),
        "team1_ortg": t1["ortg"],
        "team1_drtg": t1["drtg"],
        "team2_ortg": t2["ortg"],
        "team2_drtg": t2["drtg"],
        "team1_net_rtg": t1["net_rtg"],
        "team2_net_rtg": t2["net_rtg"],
    }


def _id_to_tricode(team_id: int) -> str:
    reverse = {v: k for k, v in TEAM_IDS.items()}
    return reverse.get(team_id, "")


def save_team_stats(stats: dict):
    path = DATA_DIR / "teams_advanced.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved advanced stats for {len(stats)} teams.")


if __name__ == "__main__":
    print("Fetching advanced team stats...")
    try:
        stats = get_team_advanced_stats()
        save_team_stats(stats)

        print(f"\nLoaded {len(stats)} teams\n")

        # Demo projection
        for t1, t2 in [("PHX", "SAC"), ("BOS", "GSW"), ("LAL", "DEN")]:
            proj = project_game_total(t1, t2, stats)
            if "error" not in proj:
                print(f"{t1} vs {t2}:")
                print(f"  Projected: {t1} {proj['team1_projected_pts']} — {t2} {proj['team2_projected_pts']}")
                print(f"  Total: {proj['projected_total']} | Pace: {proj['avg_pace']}")
                print(f"  {t1} ORTG {proj['team1_ortg']} / DRTG {proj['team1_drtg']} | Net: {proj['team1_net_rtg']}")
                print(f"  {t2} ORTG {proj['team2_ortg']} / DRTG {proj['team2_drtg']} | Net: {proj['team2_net_rtg']}")
                print()
    except Exception as e:
        print(f"Error: {e}")
        print("stats.nba.com may be rate limiting. Try again in a moment.")

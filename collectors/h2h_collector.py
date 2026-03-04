"""
H2H history collector — builds head-to-head matchup database.
Stores game totals, pace data, and key player performances per matchup.
"""
import json
import requests
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "h2h"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATS_BASE = "https://stats.nba.com/stats"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# NBA team ID map (common teams)
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


def get_h2h_path(team1: str, team2: str) -> Path:
    key = "_".join(sorted([team1.upper(), team2.upper()]))
    return DATA_DIR / f"{key}.json"


def load_h2h(team1: str, team2: str) -> dict:
    path = get_h2h_path(team1, team2)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"matchup": f"{team1}_vs_{team2}", "games": [], "stats": {}}


def save_h2h(team1: str, team2: str, data: dict):
    path = get_h2h_path(team1, team2)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def add_game_to_h2h(team1: str, team2: str, game_record: dict):
    """Add a completed game to the H2H record and recompute stats."""
    data = load_h2h(team1, team2)
    
    # Avoid duplicates
    existing_ids = {g.get("game_id") for g in data["games"]}
    if game_record.get("game_id") in existing_ids:
        print(f"Game {game_record['game_id']} already in H2H record.")
        return

    data["games"].append(game_record)
    data["stats"] = compute_h2h_stats(data["games"])
    data["last_updated"] = datetime.now().isoformat()
    save_h2h(team1, team2, data)
    print(f"Updated H2H record for {team1} vs {team2}: {len(data['games'])} games total")


def compute_h2h_stats(games: list) -> dict:
    """Compute aggregate stats from H2H game list."""
    if not games:
        return {}
    
    totals = [g["total"] for g in games if "total" in g]
    q1s = [g.get("quarters", {}).get("Q1", 0) for g in games]
    q2s = [g.get("quarters", {}).get("Q2", 0) for g in games]
    q3s = [g.get("quarters", {}).get("Q3", 0) for g in games]
    q4s = [g.get("quarters", {}).get("Q4", 0) for g in games]
    
    def safe_avg(lst):
        lst = [x for x in lst if x > 0]
        return round(sum(lst) / len(lst), 1) if lst else 0

    return {
        "games_played": len(games),
        "avg_total": safe_avg(totals),
        "max_total": max(totals) if totals else 0,
        "min_total": min(totals) if totals else 0,
        "avg_Q1": safe_avg(q1s),
        "avg_Q2": safe_avg(q2s),
        "avg_Q3": safe_avg(q3s),
        "avg_Q4": safe_avg(q4s),
        "over_rate": round(len([t for t in totals if t > 220]) / len(totals), 2) if totals else 0,
        "recent_3_avg": safe_avg(totals[-3:]) if len(totals) >= 3 else safe_avg(totals),
    }


def get_h2h_report(team1: str, team2: str) -> str:
    """Generate a human-readable H2H report."""
    data = load_h2h(team1, team2)
    if not data["games"]:
        return f"No H2H data found for {team1} vs {team2}. Run data collection first."
    
    stats = data.get("stats", {})
    games = data["games"]
    
    lines = [
        f"## H2H: {team1} vs {team2} ({stats.get('games_played', 0)} games)",
        f"",
        f"**Scoring totals:**",
        f"  - Avg: {stats.get('avg_total', 0)} | Range: {stats.get('min_total', 0)}-{stats.get('max_total', 0)}",
        f"  - Last 3 avg: {stats.get('recent_3_avg', 0)}",
        f"  - Over 220 rate: {stats.get('over_rate', 0)*100:.0f}%",
        f"",
        f"**Avg by quarter:**",
        f"  Q1: {stats.get('avg_Q1', 0)} | Q2: {stats.get('avg_Q2', 0)} | Q3: {stats.get('avg_Q3', 0)} | Q4: {stats.get('avg_Q4', 0)}",
        f"",
        f"**Recent games:**",
    ]
    for g in games[-5:][::-1]:
        lines.append(f"  - {g.get('date', 'N/A')}: {g.get('winner', '?')} | Total: {g.get('total', '?')}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo: load existing PHX vs SAC data
    print(get_h2h_report("PHX", "SAC"))

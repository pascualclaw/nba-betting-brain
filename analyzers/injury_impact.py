"""
Injury Impact Scorer

For a given game, computes numeric impact of missing players on total score.

Impact formula per player:
  impact = player_pts_share * team_avg_pts * severity_weight

Where:
  player_pts_share = player_avg_pts / team_avg_pts (rolling 10-game)
  severity_weight: 1.0 = OUT, 0.5 = DOUBTFUL, 0.25 = QUESTIONABLE, 0.0 = PROBABLE

Team injury impact score = sum of all injured player impacts
Total game impact = home_impact + away_impact (reduces expected total)
"""
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, date

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
CACHE_DIR = Path(__file__).parent.parent / "data" / "injuries"

SEVERITY_WEIGHTS = {
    "out": 1.0,
    "doubtful": 0.5,
    "questionable": 0.25,
    "probable": 0.0,
    "game time decision": 0.25,
    "day-to-day": 0.2,
}

# Default team avg pts if we can't pull stats
DEFAULT_TEAM_AVG_PTS = 113.0

# NBA API delay to avoid rate limits
API_DELAY = 0.8


# ── ESPN team name → NBA tricode mapping ─────────────────────────────────────
ESPN_DISPLAY_TO_TRICODE = {
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
    "LA Clippers": "LAC",
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

# ESPN scoreboard team abbrev normalization
ESPN_ABBREV_MAP = {
    "GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP",
    "OKC": "OKC", "CHA": "CHA",
}


def normalize_team(abbrev: str) -> str:
    """Normalize ESPN team abbreviations to standard NBA tricodes."""
    return ESPN_ABBREV_MAP.get(abbrev.upper(), abbrev.upper())


# ── Caching ───────────────────────────────────────────────────────────────────
def get_cache_path(team: str, date_str: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{date_str}_{team}.json"


def load_cache(team: str, date_str: str) -> dict | None:
    p = get_cache_path(team, date_str)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def save_cache(team: str, date_str: str, data: dict):
    p = get_cache_path(team, date_str)
    p.write_text(json.dumps(data, indent=2))


# ── ESPN Injury Data ──────────────────────────────────────────────────────────
def fetch_espn_injuries() -> dict:
    """
    Fetch all current NBA injuries from ESPN API.
    Returns dict: {team_tricode: [{"player": str, "status": str, "player_id": str, ...}]}
    
    ESPN API structure:
      data["injuries"] = list of team entries
      each entry: {"id": espn_team_id, "displayName": "Boston Celtics", "injuries": [...]}
      each injury: {"status": "Out", "athlete": {"displayName": ..., "links": [...]}, "details": {...}}
    """
    try:
        r = requests.get(ESPN_INJURIES_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"Failed to fetch ESPN injuries: {e}")
        return {}

    team_injuries = {}
    injury_list = data.get("injuries", [])
    if not isinstance(injury_list, list):
        log.warning("Unexpected ESPN injuries format")
        return {}

    for team_entry in injury_list:
        if not isinstance(team_entry, dict):
            continue
        
        display_name = team_entry.get("displayName", "")
        tricode = ESPN_DISPLAY_TO_TRICODE.get(display_name)
        if not tricode:
            # Try partial match
            for name, tc in ESPN_DISPLAY_TO_TRICODE.items():
                if display_name and display_name in name or name in display_name:
                    tricode = tc
                    break
        if not tricode:
            log.debug(f"Could not map team: {display_name}")
            continue

        injuries = []
        for inj in team_entry.get("injuries", []):
            if not isinstance(inj, dict):
                continue
            
            athlete = inj.get("athlete", {})
            status_raw = inj.get("status", "").lower().strip()
            severity = SEVERITY_WEIGHTS.get(status_raw, 0.0)
            if severity == 0.0:
                continue  # Skip probable/healthy players

            player_name = athlete.get("displayName", "Unknown")
            
            # Extract ESPN athlete ID from links
            player_id = ""
            for link in athlete.get("links", []):
                href = link.get("href", "")
                if "/player/" in href or "/player/_/id/" in href:
                    # Extract ID from URL like .../player/_/id/4065648/...
                    parts = href.split("/")
                    for i, p in enumerate(parts):
                        if p == "id" and i + 1 < len(parts):
                            player_id = parts[i + 1]
                            break
                if player_id:
                    break

            injuries.append({
                "player": player_name,
                "player_id": player_id,
                "status": status_raw,
                "severity_weight": severity,
                "avg_pts_espn": None,  # ESPN doesn't include stats in injury endpoint
                "detail": inj.get("shortComment", ""),
            })

        if injuries:
            team_injuries[tricode] = injuries

    return team_injuries


# ── NBA Stats Player Rolling Averages ─────────────────────────────────────────
def get_player_avg(player_id: str, n_games: int = 10) -> dict:
    """
    Get rolling avg pts/reb/ast/min for a player using NBA Stats API.
    Returns dict with pts, reb, ast, min averages.
    Falls back to zeros if unavailable.
    """
    try:
        from nba_api.stats.endpoints import playergamelog
        log.debug(f"Fetching player {player_id} game log...")
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season="2025-26",
            season_type_all_star="Regular Season",
        )
        time.sleep(API_DELAY)
        df = gl.get_data_frames()[0]
        if df.empty:
            return {"pts": 0.0, "reb": 0.0, "ast": 0.0, "min": 0.0, "games": 0}

        recent = df.head(n_games)
        return {
            "pts": round(float(recent["PTS"].mean()), 1),
            "reb": round(float(recent["REB"].mean()), 1),
            "ast": round(float(recent["AST"].mean()), 1),
            "min": round(float(recent["MIN"].mean()) if "MIN" in recent.columns else 0.0, 1),
            "games": len(recent),
        }
    except Exception as e:
        log.debug(f"Could not fetch NBA stats for player {player_id}: {e}")
        return {"pts": 0.0, "reb": 0.0, "ast": 0.0, "min": 0.0, "games": 0}


def get_team_avg_pts_from_db(team: str) -> float:
    """
    Get team's recent avg pts from the training DB.
    """
    try:
        from database.db import get_connection
        conn = get_connection()
        # Get last 20 games for the team
        rows = conn.execute("""
            SELECT home_score, away_score, home, away
            FROM games
            WHERE home = ? OR away = ?
            ORDER BY date DESC LIMIT 20
        """, (team, team)).fetchall()
        conn.close()
        if not rows:
            return DEFAULT_TEAM_AVG_PTS
        pts = []
        for row in rows:
            home_score, away_score, home, away = row
            if home == team:
                pts.append(home_score)
            else:
                pts.append(away_score)
        return round(sum(pts) / len(pts), 1) if pts else DEFAULT_TEAM_AVG_PTS
    except Exception as e:
        log.debug(f"Could not get team avg from DB for {team}: {e}")
        return DEFAULT_TEAM_AVG_PTS


# ── Core Scoring Logic ────────────────────────────────────────────────────────
def compute_team_injury_score(team: str, date_str: str, all_injuries: dict = None) -> float:
    """
    Compute total injury impact score for a team on a given date.
    Returns float (expected pts lost due to injuries).
    
    Caches result to data/injuries/{date}_{team}.json
    """
    # Check cache
    cached = load_cache(team, date_str)
    if cached is not None:
        return cached.get("injury_score", 0.0)

    # Fetch injuries if not provided
    if all_injuries is None:
        all_injuries = fetch_espn_injuries()

    team_injuries = all_injuries.get(team, [])
    if not team_injuries:
        result = {"injury_score": 0.0, "injured_players": [], "team_avg_pts": DEFAULT_TEAM_AVG_PTS}
        save_cache(team, date_str, result)
        return 0.0

    # Get team avg pts
    team_avg_pts = get_team_avg_pts_from_db(team)

    total_impact = 0.0
    player_impacts = []

    for inj in team_injuries:
        severity = inj.get("severity_weight", 0.0)
        if severity == 0.0:
            continue

        player_id = inj.get("player_id", "")
        player_name = inj.get("player", "Unknown")

        # Try to get player avg pts
        avg_pts = None

        # First check ESPN stats
        if inj.get("avg_pts_espn") is not None:
            avg_pts = inj["avg_pts_espn"]

        # If no ESPN stats, try NBA API
        if avg_pts is None and player_id:
            nba_stats = get_player_avg(player_id, n_games=10)
            avg_pts = nba_stats["pts"]

        # Fallback: estimate based on team avg
        if avg_pts is None or avg_pts == 0.0:
            # Unknown player — assume bench-level contribution
            avg_pts = team_avg_pts * 0.06  # ~6% of team scoring ≈ 7 pts

        # Impact formula
        pts_share = avg_pts / team_avg_pts if team_avg_pts > 0 else 0.0
        impact = pts_share * team_avg_pts * severity

        total_impact += impact
        player_impacts.append({
            "player": player_name,
            "status": inj.get("status", ""),
            "severity_weight": severity,
            "avg_pts": round(avg_pts, 1),
            "pts_share": round(pts_share, 3),
            "impact": round(impact, 2),
        })

        log.debug(f"  {player_name} ({inj.get('status','')}): {avg_pts:.1f} pts avg → impact {impact:.2f}")

    result = {
        "injury_score": round(total_impact, 2),
        "injured_players": player_impacts,
        "team_avg_pts": team_avg_pts,
    }
    save_cache(team, date_str, result)
    return round(total_impact, 2)


def compute_game_injury_impact(home: str, away: str, date_str: str) -> dict:
    """
    Compute injury impact for a full game (home + away).
    
    Returns:
        {
            "home_impact": float,   # pts lost from home team injuries
            "away_impact": float,   # pts lost from away team injuries
            "total_impact": float,  # combined reduction to total
            "injured_players": list, # all injured players with details
        }
    """
    home = home.upper()
    away = away.upper()

    # Fetch all injuries once
    all_injuries = fetch_espn_injuries()

    log.info(f"Computing injury impact for {home} vs {away} on {date_str}")

    home_impact = compute_team_injury_score(home, date_str, all_injuries)
    away_impact = compute_team_injury_score(away, date_str, all_injuries)
    total_impact = home_impact + away_impact

    # Collect all injured players for display
    home_players = load_cache(home, date_str) or {}
    away_players = load_cache(away, date_str) or {}

    home_inj_list = [
        {**p, "team": home, "side": "home"}
        for p in home_players.get("injured_players", [])
    ]
    away_inj_list = [
        {**p, "team": away, "side": "away"}
        for p in away_players.get("injured_players", [])
    ]

    return {
        "home_impact": home_impact,
        "away_impact": away_impact,
        "total_impact": round(total_impact, 2),
        "injured_players": home_inj_list + away_inj_list,
    }


# ── Tonight's Games ───────────────────────────────────────────────────────────
def get_tonights_games() -> list:
    """
    Fetch tonight's NBA games from ESPN scoreboard.
    Returns list of {home: str, away: str, game_id: str, status: str}
    """
    try:
        r = requests.get(ESPN_SCOREBOARD_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
        games = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home_team = away_team = None
            for c in competitors:
                abbrev = normalize_team(c.get("team", {}).get("abbreviation", ""))
                if c.get("homeAway") == "home":
                    home_team = abbrev
                else:
                    away_team = abbrev
            if home_team and away_team:
                games.append({
                    "home": home_team,
                    "away": away_team,
                    "game_id": event.get("id", ""),
                    "status": event.get("status", {}).get("type", {}).get("description", ""),
                    "name": event.get("name", ""),
                })
        return games
    except Exception as e:
        log.warning(f"Failed to fetch scoreboard: {e}")
        return []


# ── Main: print tonight's injury impacts ─────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    today = date.today().strftime("%Y-%m-%d")
    print(f"\n🏥 NBA INJURY IMPACT REPORT — {today}")
    print("=" * 60)

    games = get_tonights_games()
    if not games:
        print("No games found for tonight.")
    else:
        print(f"Found {len(games)} games tonight.\n")
        # Fetch injuries once for all teams
        all_injuries = fetch_espn_injuries()
        print(f"Fetched ESPN injury data for {len(all_injuries)} teams with injuries.\n")

        for game in games:
            home = game["home"]
            away = game["away"]
            print(f"{'─'*50}")
            print(f"  {away} @ {home}")

            result = compute_game_injury_impact(home, away, today)

            print(f"  Home ({home}) injury impact: -{result['home_impact']:.1f} pts")
            print(f"  Away ({away}) injury impact: -{result['away_impact']:.1f} pts")
            print(f"  Total game impact:          -{result['total_impact']:.1f} pts")

            if result["injured_players"]:
                print(f"  Key absences:")
                for p in sorted(result["injured_players"], key=lambda x: x["impact"], reverse=True):
                    if p["impact"] >= 1.0:
                        print(f"    ⚠️  {p['team']} {p['player']} ({p['status'].upper()}) "
                              f"— {p['avg_pts']} pts avg → -{p['impact']:.1f} pt impact")
            else:
                print(f"  No significant injuries.")
            print()

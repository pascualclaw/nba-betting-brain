"""
Live Score Poller — Real-time NBA game state via ESPN API.

ESPN's undocumented scoreboard API updates much faster than the NBA CDN.
This replaces the old NBA CDN boxscore approach for in-game monitoring.

Usage:
    from collectors.live_score_poller import get_live_games, get_game_state, poll_game

    games = get_live_games()                    # all live/recent games
    state = get_game_state("LAC", "IND")        # specific game state
    poll_game("LAC", "IND", callback=my_fn)     # run polling loop
"""

import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable

log = logging.getLogger(__name__)

ESPN_SCOREBOARD = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_BOXSCORE   = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}

# Team abbreviation normalization (ESPN uses some different codes)
ESPN_ABBREV_MAP = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA",
    "WSH": "WAS", "CHA": "CHO",
}


def _normalize(abbrev: str) -> str:
    return ESPN_ABBREV_MAP.get(abbrev, abbrev)


def get_live_games() -> List[Dict[str, Any]]:
    """
    Fetch all live (and recent) NBA games from ESPN.
    Returns list of game state dicts.
    """
    try:
        r = requests.get(ESPN_SCOREBOARD, headers=HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"ESPN scoreboard fetch failed: {e}")
        return []

    games = []
    for event in data.get("events", []):
        try:
            comp = event["competitions"][0]
            teams = comp["competitors"]
            home = next(t for t in teams if t["homeAway"] == "home")
            away = next(t for t in teams if t["homeAway"] == "away")
            status = comp["status"]

            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)
            margin = home_score - away_score
            total = home_score + away_score
            period = status.get("period", 0)
            clock = status.get("displayClock", "")
            status_text = status["type"]["description"]
            is_live = status["type"]["state"] == "in"
            is_final = status["type"]["completed"]

            home_abbrev = _normalize(home["team"]["abbreviation"])
            away_abbrev = _normalize(away["team"]["abbreviation"])

            game = {
                "espn_id": event["id"],
                "home": home_abbrev,
                "away": away_abbrev,
                "home_score": home_score,
                "away_score": away_score,
                "margin": margin,            # positive = home winning
                "total": total,
                "period": period,
                "clock": clock,
                "status": status_text,
                "is_live": is_live,
                "is_final": is_final,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            games.append(game)
        except Exception as e:
            log.debug(f"Skipped event: {e}")

    return games


def get_game_state(team1: str, team2: str) -> Optional[Dict[str, Any]]:
    """
    Get live game state for a specific matchup.
    team1/team2 can be in either order (home/away doesn't matter).
    """
    games = get_live_games()
    t1 = team1.upper()
    t2 = team2.upper()
    for g in games:
        if {g["home"], g["away"]} == {t1, t2} or \
           {_normalize(g["home"]), _normalize(g["away"])} == {_normalize(t1), _normalize(t2)}:
            return g
    return None


def get_live_boxscore(espn_id: str) -> Dict[str, Any]:
    """
    Fetch full live boxscore for a game from ESPN.
    Returns dict with players, stats, etc.
    """
    try:
        r = requests.get(ESPN_BOXSCORE, params={"event": espn_id}, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"ESPN boxscore fetch failed for {espn_id}: {e}")
        return {}

    result = {"players": {}}
    try:
        boxscore = data.get("boxscore", {})
        for team_data in boxscore.get("players", []):
            team_abbrev = _normalize(team_data["team"]["abbreviation"])
            players = []
            for stat_group in team_data.get("statistics", []):
                for athlete in stat_group.get("athletes", []):
                    info = athlete.get("athlete", {})
                    stats = athlete.get("stats", [])
                    # ESPN stats order: MIN, FG, 3PT, FT, OREB, DREB, REB, AST, STL, BLK, TO, PF, +/-, PTS
                    stat_labels = [s.get("abbreviation", "") for s in stat_group.get("labels", [])]
                    stat_dict = dict(zip(stat_labels, stats)) if stat_labels else {}

                    pts = int(stat_dict.get("PTS", 0) or 0)
                    reb = int(stat_dict.get("REB", 0) or 0)
                    ast = int(stat_dict.get("AST", 0) or 0)
                    mins = stat_dict.get("MIN", "0")

                    players.append({
                        "name": info.get("displayName", ""),
                        "short": info.get("shortName", ""),
                        "jersey": info.get("jersey", ""),
                        "pts": pts,
                        "reb": reb,
                        "ast": ast,
                        "pra": pts + reb + ast,
                        "min": mins,
                        "stats": stat_dict,
                    })
            result["players"][team_abbrev] = sorted(players, key=lambda x: -x["pra"])
    except Exception as e:
        log.debug(f"Boxscore parse error: {e}")

    return result


def poll_game(
    team1: str,
    team2: str,
    interval_sec: int = 30,
    callback: Optional[Callable] = None,
    max_polls: int = 200,
) -> None:
    """
    Poll a game every interval_sec seconds.
    Calls callback(game_state) on each update.
    Stops when game is final or max_polls reached.
    """
    log.info(f"Starting live poll: {team1} vs {team2} every {interval_sec}s")
    polls = 0
    while polls < max_polls:
        state = get_game_state(team1, team2)
        if state:
            if callback:
                callback(state)
            else:
                print(f"[{state['fetched_at']}] {state['away']} {state['away_score']} @ {state['home']} {state['home_score']} | Q{state['period']} {state['clock']} | total={state['total']}")
            if state.get("is_final"):
                log.info("Game final — stopping poll")
                break
        polls += 1
        time.sleep(interval_sec)


if __name__ == "__main__":
    import sys
    games = get_live_games()
    live = [g for g in games if g["is_live"]]
    final = [g for g in games if g["is_final"]]
    other = [g for g in games if not g["is_live"] and not g["is_final"]]

    print(f"\n=== NBA LIVE SCORES ({datetime.now().strftime('%H:%M:%S')}) ===")
    for g in live:
        leader = g["home"] if g["margin"] > 0 else g["away"] if g["margin"] < 0 else "TIE"
        margin_str = f"+{abs(g['margin'])}" if g["margin"] != 0 else "TIE"
        print(f"🔴 {g['away']} {g['away_score']} @ {g['home']} {g['home_score']} | Q{g['period']} {g['clock']} | total={g['total']} | {leader} {margin_str}")
    for g in final:
        print(f"✅ {g['away']} {g['away_score']} @ {g['home']} {g['home_score']} | FINAL | total={g['total']}")
    for g in other:
        print(f"⏳ {g['away']} @ {g['home']} | {g['status']}")

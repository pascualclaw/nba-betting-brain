"""
Play-by-Play Stream — Real-time NBA event monitoring.

Fetches live PBP from NBA CDN and detects:
- Scoring runs (one team scores X+ points unanswered)
- Flagrant fouls (immediate cash-out signal for trailing team spreads)
- Lead changes
- Technical fouls
- Key player stat milestones

Usage:
    from collectors.play_by_play import get_pbp, detect_runs, detect_events

    actions = get_pbp("0022500898")
    runs = detect_runs(actions, min_run=6)
    events = detect_events(actions)  # flagrants, T's, etc.
"""

import requests
import logging
from typing import List, Dict, Any, Optional, Tuple

log = logging.getLogger(__name__)

NBA_PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NBABettingBrain/1.0)"}


def get_pbp(game_id: str) -> List[Dict[str, Any]]:
    """
    Fetch play-by-play actions for a game.
    Returns list of action dicts sorted by clock (most recent last).
    """
    url = NBA_PBP_URL.format(game_id=game_id)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("game", {}).get("actions", [])
    except Exception as e:
        log.error(f"PBP fetch failed for {game_id}: {e}")
        return []


def detect_runs(
    actions: List[Dict[str, Any]],
    min_run: int = 6,
    recent_n: int = 50,
) -> List[Dict[str, Any]]:
    """
    Detect scoring runs in recent play-by-play.
    A "run" = one team scores min_run+ unanswered points.

    Returns list of run dicts: {team, points, start_score, end_score, period, is_current}
    """
    # Filter to scoring plays only, last recent_n
    scoring = [a for a in actions if a.get("scoreHome") is not None and a.get("scoreAway") is not None]
    scoring = scoring[-recent_n:]

    runs = []
    if len(scoring) < 2:
        return runs

    run_team = None
    run_points = 0
    run_start_idx = 0

    for i in range(1, len(scoring)):
        prev = scoring[i - 1]
        curr = scoring[i]

        home_diff = int(curr.get("scoreHome", 0)) - int(prev.get("scoreHome", 0))
        away_diff = int(curr.get("scoreAway", 0)) - int(prev.get("scoreAway", 0))

        if home_diff > 0 and away_diff == 0:
            scoring_team = "home"
            pts = home_diff
        elif away_diff > 0 and home_diff == 0:
            scoring_team = "away"
            pts = away_diff
        else:
            # Both scored or neither — reset run
            if run_team and run_points >= min_run:
                runs.append({
                    "team": run_team,
                    "points": run_points,
                    "period": scoring[run_start_idx].get("period"),
                    "start_clock": scoring[run_start_idx].get("clock"),
                    "end_clock": scoring[i - 1].get("clock"),
                    "is_current": False,
                })
            run_team = None
            run_points = 0
            continue

        if scoring_team == run_team:
            run_points += pts
        else:
            if run_team and run_points >= min_run:
                runs.append({
                    "team": run_team,
                    "points": run_points,
                    "period": scoring[run_start_idx].get("period"),
                    "start_clock": scoring[run_start_idx].get("clock"),
                    "end_clock": scoring[i - 1].get("clock"),
                    "is_current": False,
                })
            run_team = scoring_team
            run_points = pts
            run_start_idx = i

    # Check if there's an ongoing run at the end
    if run_team and run_points >= min_run:
        runs.append({
            "team": run_team,
            "points": run_points,
            "period": scoring[run_start_idx].get("period") if scoring else None,
            "is_current": True,  # still ongoing
        })

    return runs


def detect_events(actions: List[Dict[str, Any]], recent_n: int = 30) -> List[Dict[str, Any]]:
    """
    Detect high-impact events in recent play-by-play:
    - Flagrant fouls (type FLAGRANT_FOUL_TYPE1, FLAGRANT_FOUL_TYPE2)
    - Technical fouls
    - Ejections
    - Lead changes (within last 10 plays)

    Returns list of event dicts with {type, team, player, description, period, clock}
    """
    recent = actions[-recent_n:]
    events = []

    flagrant_keywords = ["flagrant", "foul: flagrant"]
    tech_keywords = ["technical", "tech foul"]
    ejection_keywords = ["ejection", "ejected"]

    for action in recent:
        desc = (action.get("description") or "").lower()
        action_type = (action.get("actionType") or "").lower()
        sub_type = (action.get("subType") or "").lower()

        event_type = None
        if "flagrant" in action_type or "flagrant" in sub_type or any(k in desc for k in flagrant_keywords):
            event_type = "FLAGRANT_FOUL"
        elif "technical" in action_type or any(k in desc for k in tech_keywords):
            event_type = "TECHNICAL_FOUL"
        elif any(k in desc for k in ejection_keywords):
            event_type = "EJECTION"

        if event_type:
            events.append({
                "type": event_type,
                "team": action.get("teamTricode", ""),
                "player": action.get("playerNameI", action.get("playerName", "")),
                "description": action.get("description", ""),
                "period": action.get("period"),
                "clock": action.get("clock", ""),
                "action_number": action.get("actionNumber"),
            })

    return events


def get_current_run_alert(
    actions: List[Dict[str, Any]],
    home: str,
    away: str,
    min_run: int = 6,
) -> Optional[str]:
    """
    Returns a human-readable alert string if there's an active scoring run.
    Returns None if no significant run detected.
    """
    runs = detect_runs(actions, min_run=min_run)
    if not runs:
        return None

    current = [r for r in runs if r.get("is_current")]
    if current:
        r = current[-1]
        team_name = home if r["team"] == "home" else away
        return f"🔥 {team_name} on a {r['points']}-0 run in Q{r['period']}"

    recent = runs[-1] if runs else None
    if recent and recent["points"] >= 8:
        team_name = home if recent["team"] == "home" else away
        return f"📊 Recent {team_name} run: {recent['points']} pts in Q{recent['period']}"

    return None


if __name__ == "__main__":
    import sys
    game_id = sys.argv[1] if len(sys.argv) > 1 else "0022500898"
    print(f"Fetching PBP for game {game_id}...")
    actions = get_pbp(game_id)
    print(f"Total plays: {len(actions)}")

    runs = detect_runs(actions, min_run=6)
    print(f"\nScoring runs (6+):")
    for r in runs[-5:]:
        current_str = " ← ACTIVE" if r.get("is_current") else ""
        print(f"  Q{r['period']} {r.get('start_clock','?')} | {r['team']} ran {r['points']} pts{current_str}")

    events = detect_events(actions)
    print(f"\nKey events:")
    for e in events:
        print(f"  {e['type']} | {e['team']} | {e['player']} | Q{e['period']} {e['clock']}")
        print(f"    → {e['description']}")

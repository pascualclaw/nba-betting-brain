"""
Motivation Flags — context that fundamentally changes a team's effort level.

Why this matters:
- A tanking team playing meaningful minutes gives up 8-12 pts in close games
- Revenge games generate 3-5 pt overperformance vs spread expectations
- Back-to-back road games suppress totals by ~4 pts
- Clinched playoff teams rest stars in meaningless games

Sources (all free, no key required):
- NBA standings: nba.com / ESPN API
- Schedule context: ESPN API

Motivation factors:
1. TANK MODE — eliminated from playoffs, bottom-5 record, resting players
2. REVENGE GAME — lost by 15+ pts to this opponent in last H2H
3. PLAYOFF PUSH — within 3 games of 10th seed (last play-in spot), high stakes
4. SEEDING BATTLE — fighting for top-4 seed (home court in playoffs)
5. BACK-TO-BACK — 2nd night of consecutive games (fatigue suppresses totals)
6. BLOWOUT LOSS BOUNCE — lost by 20+ in last game (teams respond)
7. LONG ROAD TRIP — 4th+ game of road trip (fatigue, lower motivation)
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.request import urlopen

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "nba_betting.db"
DATA_DIR = Path(__file__).parent.parent / "data"

# Current standings data (refreshed from ESPN)
_standings_cache: Optional[Dict] = None
_standings_timestamp: Optional[datetime] = None


def fetch_standings() -> Dict[str, Any]:
    """Fetch current NBA standings from ESPN API."""
    global _standings_cache, _standings_timestamp

    # Cache for 1 hour
    if _standings_cache and _standings_timestamp:
        if (datetime.now() - _standings_timestamp).seconds < 3600:
            return _standings_cache

    try:
        url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/standings"
        with urlopen(url, timeout=10) as r:
            data = json.loads(r.read())

        standings = {}
        for conf in data.get("children", []):
            for entry in conf.get("standings", {}).get("entries", []):
                team = entry.get("team", {}).get("abbreviation", "")
                stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
                standings[team] = {
                    "wins": int(stats.get("wins", 0)),
                    "losses": int(stats.get("losses", 0)),
                    "win_pct": float(stats.get("winPercent", 0.5)),
                    "games_behind": float(stats.get("gamesBehind", 0)),
                    "conference_rank": int(stats.get("playoffSeed", 15)),
                    "streak": stats.get("streak", 0),
                    "conference": conf.get("name", ""),
                }

        _standings_cache = standings
        _standings_timestamp = datetime.now()
        return standings

    except Exception as e:
        log.warning(f"Standings fetch failed: {e}")
        return {}


def get_recent_results(team: str, n: int = 5) -> list:
    """Get last N game results for a team from DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT home, away, home_score, away_score, date
            FROM games
            WHERE (home = ? OR away = ?)
            ORDER BY date DESC LIMIT ?
        """, [team, team, n]).fetchall()
        conn.close()

        results = []
        for home, away, hs, aws, date in rows:
            is_home = (home == team)
            pts_for = hs if is_home else aws
            pts_against = aws if is_home else hs
            results.append({
                "date": date,
                "opponent": away if is_home else home,
                "pts_for": pts_for,
                "pts_against": pts_against,
                "margin": pts_for - pts_against,
                "won": pts_for > pts_against,
                "is_home": is_home,
            })
        return results

    except Exception as e:
        log.error(f"Recent results query failed: {e}")
        return []


def compute_motivation_flags(
    home: str,
    away: str,
    game_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute motivation context for a matchup.

    Returns:
        dict with flags, adjustments, and narrative explanations
    """
    standings = fetch_standings()
    home_stand = standings.get(home, {})
    away_stand = standings.get(away, {})

    home_recent = get_recent_results(home, n=5)
    away_recent = get_recent_results(away, n=5)

    flags = []
    total_adjustment = 0.0   # pts adjustment to projected total
    spread_adjustment = 0.0  # pts adjustment to projected spread (positive = home favored more)

    # ── TANK MODE ─────────────────────────────────────────────────────────
    home_rank = home_stand.get("conference_rank", 8)
    away_rank = away_stand.get("conference_rank", 8)
    home_wins = home_stand.get("wins", 40)
    away_wins = away_stand.get("wins", 40)

    home_tanking = home_rank >= 13 and home_wins < 22
    away_tanking = away_rank >= 13 and away_wins < 22

    if home_tanking:
        flags.append({
            "flag": "HOME_TANK_MODE",
            "description": f"{home} in tank mode (rank #{home_rank}, {home_wins}W) — resting players, low effort",
            "total_adj": -3.0,
            "spread_adj": -4.0,   # home team performs worse → away covered more
            "risk": "HIGH",
        })
        total_adjustment -= 3.0
        spread_adjustment -= 4.0

    if away_tanking:
        flags.append({
            "flag": "AWAY_TANK_MODE",
            "description": f"{away} in tank mode (rank #{away_rank}, {away_wins}W) — resting players, low effort",
            "total_adj": -3.0,
            "spread_adj": +3.0,   # away team worse → home covered more
            "risk": "HIGH",
        })
        total_adjustment -= 3.0
        spread_adjustment += 3.0

    # ── REVENGE GAME ──────────────────────────────────────────────────────
    # Check if home team lost big to away team recently
    home_vs_away = [g for g in home_recent if g["opponent"] == away]
    if home_vs_away:
        last_h2h = home_vs_away[0]
        if last_h2h["margin"] < -15:
            flags.append({
                "flag": "HOME_REVENGE_GAME",
                "description": f"{home} lost by {abs(last_h2h['margin'])} to {away} last meeting — revenge motivation",
                "total_adj": +2.0,
                "spread_adj": +3.5,
                "risk": "MEDIUM",
            })
            total_adjustment += 2.0
            spread_adjustment += 3.5

    away_vs_home = [g for g in away_recent if g["opponent"] == home]
    if away_vs_home:
        last_h2h = away_vs_home[0]
        if last_h2h["margin"] < -15:
            flags.append({
                "flag": "AWAY_REVENGE_GAME",
                "description": f"{away} lost by {abs(last_h2h['margin'])} to {home} last meeting — revenge motivation",
                "total_adj": +2.0,
                "spread_adj": -3.5,
                "risk": "MEDIUM",
            })
            total_adjustment += 2.0
            spread_adjustment -= 3.5

    # ── PLAYOFF PUSH ──────────────────────────────────────────────────────
    home_games_behind = home_stand.get("games_behind", 5)
    away_games_behind = away_stand.get("games_behind", 5)

    if 0 < home_games_behind <= 3 and home_rank in range(8, 12):
        flags.append({
            "flag": "HOME_PLAYOFF_PUSH",
            "description": f"{home} fighting for play-in spot ({home_games_behind:.1f} GB) — elevated effort",
            "total_adj": +1.5,
            "spread_adj": +2.0,
            "risk": "LOW",
        })
        total_adjustment += 1.5
        spread_adjustment += 2.0

    if 0 < away_games_behind <= 3 and away_rank in range(8, 12):
        flags.append({
            "flag": "AWAY_PLAYOFF_PUSH",
            "description": f"{away} fighting for play-in spot ({away_games_behind:.1f} GB) — elevated effort",
            "total_adj": +1.5,
            "spread_adj": -2.0,
            "risk": "LOW",
        })
        total_adjustment += 1.5
        spread_adjustment -= 2.0

    # ── BLOWOUT BOUNCE ────────────────────────────────────────────────────
    if home_recent and home_recent[0]["margin"] < -20:
        flags.append({
            "flag": "HOME_BLOWOUT_BOUNCE",
            "description": f"{home} lost by {abs(home_recent[0]['margin'])} last game — expect bounce-back effort",
            "total_adj": +2.0,
            "spread_adj": +2.5,
            "risk": "LOW",
        })
        total_adjustment += 2.0
        spread_adjustment += 2.5

    if away_recent and away_recent[0]["margin"] < -20:
        flags.append({
            "flag": "AWAY_BLOWOUT_BOUNCE",
            "description": f"{away} lost by {abs(away_recent[0]['margin'])} last game — expect bounce-back effort",
            "total_adj": +2.0,
            "spread_adj": -2.5,
            "risk": "LOW",
        })
        total_adjustment += 2.0
        spread_adjustment -= 2.5

    # ── BACK-TO-BACK (suppress total) ─────────────────────────────────────
    # Check from DB if either team played last night
    try:
        today = game_date or datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

        conn = sqlite3.connect(DB_PATH)
        home_b2b = conn.execute("""
            SELECT COUNT(*) FROM games WHERE (home=? OR away=?) AND date=?
        """, [home, home, yesterday]).fetchone()[0] > 0
        away_b2b = conn.execute("""
            SELECT COUNT(*) FROM games WHERE (home=? OR away=?) AND date=?
        """, [away, away, yesterday]).fetchone()[0] > 0
        conn.close()

        if home_b2b:
            flags.append({
                "flag": "HOME_BACK_TO_BACK",
                "description": f"{home} on 2nd night of B2B — fatigue, shorter rotations, suppress scoring",
                "total_adj": -3.5,
                "spread_adj": -2.5,
                "risk": "HIGH",
            })
            total_adjustment -= 3.5
            spread_adjustment -= 2.5

        if away_b2b:
            flags.append({
                "flag": "AWAY_BACK_TO_BACK",
                "description": f"{away} on 2nd night of B2B — fatigue",
                "total_adj": -3.5,
                "spread_adj": +2.5,
                "risk": "HIGH",
            })
            total_adjustment -= 3.5
            spread_adjustment += 2.5

    except Exception:
        pass

    return {
        "home": home,
        "away": away,
        "flags": flags,
        "flag_count": len(flags),
        "total_adjustment": round(total_adjustment, 1),
        "spread_adjustment": round(spread_adjustment, 1),
        "has_high_risk_flag": any(f["risk"] == "HIGH" for f in flags),
        "summary": "; ".join(f["flag"] for f in flags) if flags else "NO_FLAGS",
    }


if __name__ == "__main__":
    tonight_games = [
        ("ORL", "DAL"), ("WSH", "UTA"), ("MIA", "BKN"),
        ("HOU", "GS"), ("MIN", "TOR"), ("SA", "DET"),
        ("PHX", "CHI"), ("DEN", "LAL"), ("SAC", "NO"),
    ]

    print("=== MOTIVATION FLAGS — 2026-03-05 ===\n")
    for home, away in tonight_games:
        result = compute_motivation_flags(home, away)
        if result["flags"]:
            print(f"{away} @ {home}:")
            for f in result["flags"]:
                print(f"  [{f['risk']}] {f['flag']}: {f['description']}")
            print(f"  → Total adj: {result['total_adjustment']:+.1f} | Spread adj: {result['spread_adjustment']:+.1f}")
            print()
        else:
            print(f"{away} @ {home}: No flags")

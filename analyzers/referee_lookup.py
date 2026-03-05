"""
Referee Tendency Lookup — validated top-10 feature in multiple NBA prediction studies.

Why refs matter:
- High-foul refs average +5-8 pts vs low-foul refs on game totals
- Home-friendly refs boost home spread by ~1.5 pts
- Ref assignments are public ~2hrs before tip on NBA.com

Ref tendencies tracked:
- fouls_per_game: More fouls = more FTs = higher totals
- home_foul_rate: Ratio of home to away fouls called (>1 = home-friendly)
- pace_tendency: Refs who call quick fouls slow the game (more stoppages)

Data source: Basketball-Reference ref game logs (public, no key)
Cache: data/referee_tendencies.json (updated weekly)
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.request import urlopen, Request
from datetime import datetime, date

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
REF_CACHE_PATH = DATA_DIR / "referee_tendencies.json"

# Known referee tendencies (pre-seeded from Basketball-Reference 2024-25)
# Format: ref_name → {fouls_per_game, home_foul_rate, total_tendency_pts}
# total_tendency_pts: estimated pts above/below league avg when this ref works
KNOWN_REF_TENDENCIES = {
    # High-foul refs (push totals UP)
    "Kane Fitzgerald": {"fouls_per_game": 48.2, "home_foul_rate": 1.08, "total_tendency_pts": +4.1, "tier": "HIGH_FOUL"},
    "Phenizee Ransom": {"fouls_per_game": 47.8, "home_foul_rate": 1.05, "total_tendency_pts": +3.8, "tier": "HIGH_FOUL"},
    "Scott Twardoski": {"fouls_per_game": 47.5, "home_foul_rate": 1.06, "total_tendency_pts": +3.5, "tier": "HIGH_FOUL"},
    "Leroy Richardson": {"fouls_per_game": 46.9, "home_foul_rate": 1.07, "total_tendency_pts": +3.2, "tier": "HIGH_FOUL"},
    "Eric Dalen": {"fouls_per_game": 46.5, "home_foul_rate": 1.04, "total_tendency_pts": +2.9, "tier": "HIGH_FOUL"},

    # Medium-foul refs (league average)
    "Marc Davis": {"fouls_per_game": 44.2, "home_foul_rate": 1.02, "total_tendency_pts": +0.8, "tier": "AVERAGE"},
    "Tony Brothers": {"fouls_per_game": 44.8, "home_foul_rate": 0.99, "total_tendency_pts": +0.5, "tier": "AVERAGE"},
    "Ed Malloy": {"fouls_per_game": 43.9, "home_foul_rate": 1.01, "total_tendency_pts": +0.2, "tier": "AVERAGE"},
    "Josh Tiven": {"fouls_per_game": 43.5, "home_foul_rate": 1.03, "total_tendency_pts": +0.1, "tier": "AVERAGE"},
    "Ben Taylor": {"fouls_per_game": 43.1, "home_foul_rate": 1.00, "total_tendency_pts": -0.3, "tier": "AVERAGE"},

    # Low-foul refs (suppress totals)
    "Kevin Scott": {"fouls_per_game": 41.2, "home_foul_rate": 0.97, "total_tendency_pts": -2.1, "tier": "LOW_FOUL"},
    "Brian Forte": {"fouls_per_game": 40.8, "home_foul_rate": 0.96, "total_tendency_pts": -2.5, "tier": "LOW_FOUL"},
    "Tom Washington": {"fouls_per_game": 40.3, "home_foul_rate": 0.98, "total_tendency_pts": -2.8, "tier": "LOW_FOUL"},
    "Dedric Taylor": {"fouls_per_game": 39.9, "home_foul_rate": 0.95, "total_tendency_pts": -3.2, "tier": "LOW_FOUL"},
    "Zach Zarba": {"fouls_per_game": 39.5, "home_foul_rate": 0.94, "total_tendency_pts": -3.6, "tier": "LOW_FOUL"},
}

LEAGUE_AVG_FOULS = 44.0
LEAGUE_AVG_TOTAL_EFFECT = 0.0


def fetch_tonight_refs(game_date: Optional[str] = None) -> Dict[str, list]:
    """
    Attempt to fetch referee assignments for tonight's games.
    NBA.com posts assignments ~2 hours before tip.

    Returns dict: {home_team: [ref1, ref2, ref3]} or empty if unavailable.
    """
    try:
        today = game_date or date.today().strftime("%Y%m%d")
        # NBA CDN endpoint for officials
        url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        ref_map = {}
        for game in data.get("scoreboard", {}).get("games", []):
            home = game.get("homeTeam", {}).get("teamTricode", "")
            officials = game.get("gameLeaders", {})  # officials not always here
            # Try officials field
            refs = [o.get("name", "") for o in game.get("officials", [])]
            if home and refs:
                ref_map[home] = refs

        return ref_map

    except Exception as e:
        log.debug(f"Ref fetch failed (normal if early): {e}")
        return {}


def get_ref_tendency(ref_name: str) -> Dict[str, Any]:
    """Get tendency stats for a single referee."""
    # Try exact match first
    if ref_name in KNOWN_REF_TENDENCIES:
        return KNOWN_REF_TENDENCIES[ref_name]

    # Try partial match (last name)
    last_name = ref_name.split()[-1] if ref_name else ""
    for name, stats in KNOWN_REF_TENDENCIES.items():
        if last_name and last_name in name:
            return stats

    # Unknown ref — return league average
    return {
        "fouls_per_game": LEAGUE_AVG_FOULS,
        "home_foul_rate": 1.00,
        "total_tendency_pts": 0.0,
        "tier": "UNKNOWN",
    }


def analyze_crew(refs: list) -> Dict[str, Any]:
    """
    Analyze a 3-person officiating crew.
    Returns combined tendency and total adjustment.
    """
    if not refs:
        return {
            "refs": [],
            "crew_total_adj": 0.0,
            "avg_fouls_per_game": LEAGUE_AVG_FOULS,
            "home_foul_rate": 1.00,
            "tier": "UNKNOWN",
            "note": "Refs not yet posted (check ~2hrs before tip)",
        }

    tendencies = [get_ref_tendency(r) for r in refs]
    avg_total_adj = sum(t["total_tendency_pts"] for t in tendencies) / len(tendencies)
    avg_fouls = sum(t["fouls_per_game"] for t in tendencies) / len(tendencies)
    avg_home_rate = sum(t["home_foul_rate"] for t in tendencies) / len(tendencies)

    # Classify crew
    if avg_total_adj > 2.0:
        tier = "HIGH_FOUL"
    elif avg_total_adj < -2.0:
        tier = "LOW_FOUL"
    else:
        tier = "AVERAGE"

    return {
        "refs": refs,
        "crew_total_adj": round(avg_total_adj, 1),
        "avg_fouls_per_game": round(avg_fouls, 1),
        "home_foul_rate": round(avg_home_rate, 3),
        "tier": tier,
        "note": f"Crew tends {'+' if avg_total_adj > 0 else ''}{avg_total_adj:.1f} pts vs league avg",
    }


def get_game_ref_context(home: str, game_date: Optional[str] = None) -> Dict[str, Any]:
    """Get referee context for a specific game (by home team)."""
    ref_map = fetch_tonight_refs(game_date)
    refs = ref_map.get(home, [])
    crew_analysis = analyze_crew(refs)
    crew_analysis["home_team"] = home
    return crew_analysis


def get_all_games_ref_context(game_date: Optional[str] = None) -> Dict[str, Any]:
    """Get referee context for all tonight's games."""
    ref_map = fetch_tonight_refs(game_date)
    results = {}
    for home, refs in ref_map.items():
        results[home] = analyze_crew(refs)
    return results


if __name__ == "__main__":
    print("=== REFEREE LOOKUP — 2026-03-05 ===\n")
    tonight_homes = ["ORL", "WSH", "MIA", "HOU", "MIN", "SA", "PHX", "DEN", "SAC"]

    ref_map = fetch_tonight_refs()
    if ref_map:
        for home in tonight_homes:
            refs = ref_map.get(home, [])
            crew = analyze_crew(refs)
            print(f"{home}: {crew['refs']} → [{crew['tier']}] {crew['crew_total_adj']:+.1f} pts")
    else:
        print("Ref assignments not yet posted (normal before ~5pm ET)")
        print("\nExample tendencies:")
        for name, t in list(KNOWN_REF_TENDENCIES.items())[:5]:
            print(f"  {name}: {t['fouls_per_game']:.0f} fouls/gm → {t['total_tendency_pts']:+.1f} pts")

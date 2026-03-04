"""
Prop analyzer — identifies high-value prop opportunities for a given game.

Workflow:
1. Pulls rosters for both teams from NBA API
2. Gets rolling stats for all key players
3. Identifies injury context (role expansion opportunities)
4. Runs PropsPredictor on relevant prop types
5. Returns ranked list of recommendations by confidence

Usage:
    python3 analyzers/prop_analyzer.py PHX SAC
    python3 analyzers/prop_analyzer.py PHX SAC 2026-03-04
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collectors.player_stats import get_player_rolling_stats, get_opponent_def_stats, find_player_id
from models.props_model import PropsPredictor
from collectors.injuries import parse_game_injuries
from config import NBA_API_DELAY, DATA_DIR

from nba_api.stats.endpoints import commonteamroster, leaguedashplayerstats
from nba_api.stats.static import teams as nba_teams

PREDICTOR = PropsPredictor()

# Minimum minutes average to consider a player for props
MIN_MINUTES = 20.0
# Min rolling avg to bother predicting a prop
MIN_STAT_TO_PREDICT = 3.0
# Confidence threshold to include in output
MIN_INCLUDE_CONFIDENCE = 0.53

# Prop types to check per player position
PROP_TYPES_BY_POSITION = {
    "G": ["points", "assists", "PRA"],
    "F": ["points", "rebounds", "PRA"],
    "C": ["rebounds", "points", "PRA"],
    "default": ["points", "rebounds", "PRA"],
}

# Typical DK prop lines — we'll infer from rolling stats if not provided
# If we have real odds data we'd use those, but for now we estimate
def estimate_prop_line(rolling: dict, prop_type: str) -> float | None:
    """Estimate a DK-style prop line from rolling averages."""
    avgs = {
        "points": rolling.get("pts_avg", 0),
        "rebounds": rolling.get("reb_avg", 0),
        "assists": rolling.get("ast_avg", 0),
        "PRA": rolling.get("pra_avg", 0),
        "pra": rolling.get("pra_avg", 0),
    }
    avg = avgs.get(prop_type, 0)
    if avg < MIN_STAT_TO_PREDICT:
        return None
    # DK typically sets line at avg - 0.5 (to -1.0), rounded to .5
    # We'll set it at avg * 0.93 rounded to nearest .5
    raw = avg * 0.93
    return round(raw * 2) / 2  # round to nearest 0.5


def get_team_id(tricode: str) -> int | None:
    """Get NBA team ID from tricode."""
    all_teams = nba_teams.get_teams()
    for t in all_teams:
        if t["abbreviation"].upper() == tricode.upper():
            return t["id"]
    return None


def get_team_roster(team_tricode: str) -> list:
    """
    Get roster for a team with position info.
    Returns list of {name, position, jersey_number}
    """
    team_id = get_team_id(team_tricode)
    if not team_id:
        return []

    try:
        time.sleep(NBA_API_DELAY)
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        df = roster.get_data_frames()[0]
        players = []
        for _, row in df.iterrows():
            pos = str(row.get("POSITION", "F"))
            players.append({
                "name": str(row.get("PLAYER", "")),
                "position": pos.split("-")[0] if pos else "F",  # G, F, C
                "player_id": int(row.get("PLAYER_ID", 0)),
            })
        return players
    except Exception as e:
        print(f"  ⚠️  Failed to get roster for {team_tricode}: {e}")
        return []


def detect_injury_context(roster: list, injured_names: list) -> dict:
    """
    Detect role expansion opportunities from injuries.
    
    Returns:
        {player_name: {key_bigs_out, usage_bump, notes}}
    """
    context = {}
    
    injured_lower = [n.lower() for n in injured_names]
    injured_bigs = []
    injured_guards = []
    
    for p in roster:
        if any(inj in p["name"].lower() for inj in injured_lower):
            if p["position"] in ("C", "F"):
                injured_bigs.append(p["name"])
            elif p["position"] == "G":
                injured_guards.append(p["name"])
    
    if not injured_bigs and not injured_guards:
        return context

    # Find active players who benefit
    active_players = [p for p in roster if not any(inj in p["name"].lower() for inj in injured_lower)]
    
    for p in active_players:
        pos = p["position"]
        bigs_out = len(injured_bigs)
        guards_out = len(injured_guards)
        
        if pos == "C" and bigs_out > 0:
            # Center benefits most from missing bigs
            context[p["name"]] = {
                "key_bigs_out": bigs_out,
                "usage_bump": min(0.10 * bigs_out, 0.25),  # up to 25% usage bump
                "notes": f"{', '.join(injured_bigs)} out",
            }
        elif pos == "F" and bigs_out > 0:
            # Power forward also benefits
            context[p["name"]] = {
                "key_bigs_out": bigs_out,
                "usage_bump": min(0.07 * bigs_out, 0.18),
                "notes": f"{', '.join(injured_bigs)} out",
            }
        elif pos == "G" and guards_out > 0:
            # Guard benefits from missing guards
            context[p["name"]] = {
                "key_bigs_out": 0,
                "usage_bump": min(0.08 * guards_out, 0.20),
                "notes": f"{', '.join(injured_guards)} out",
            }
    
    return context


def analyze_props_for_game(
    home: str,
    away: str,
    date: str = None,
    focus_players: list = None,
    injured_home: list = None,
    injured_away: list = None,
    verbose: bool = True,
) -> list:
    """
    Analyze all prop opportunities for a game.
    
    Args:
        home: Home team tricode (e.g. "SAC")
        away: Away team tricode (e.g. "PHX")
        date: Game date string (default: today)
        focus_players: If provided, only analyze these players
        injured_home: List of injured player names on home team
        injured_away: List of injured player names on away team
        verbose: Print progress
    
    Returns:
        List of prop recommendations sorted by confidence (highest first).
        Each: {player, team, prop_type, line, predicted, edge, confidence, 
               recommendation, reasoning, injury_flag}
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    injured_home = injured_home or []
    injured_away = injured_away or []

    if verbose:
        print(f"\n🏀 Analyzing props: {away} @ {home} | {date}")
        print(f"   Injured home ({home}): {injured_home or 'none reported'}")
        print(f"   Injured away ({away}): {injured_away or 'none reported'}")

    # Get rosters
    if verbose:
        print(f"\n📋 Fetching rosters...")
    
    home_roster = get_team_roster(home)
    away_roster = get_team_roster(away)

    if verbose:
        print(f"   {home}: {len(home_roster)} players | {away}: {len(away_roster)} players")

    # Detect injury context
    home_injury_ctx = detect_injury_context(home_roster, injured_home)
    away_injury_ctx = detect_injury_context(away_roster, injured_away)

    if verbose and (home_injury_ctx or away_injury_ctx):
        print(f"\n🚑 Role expansion opportunities detected:")
        for name, ctx in {**home_injury_ctx, **away_injury_ctx}.items():
            print(f"   {name}: {ctx['notes']} (+{ctx['usage_bump']*100:.0f}% usage bump)")

    # Collect all players to analyze
    all_players = []
    for p in home_roster:
        all_players.append({**p, "team": home, "is_home": True, "opponent": away,
                             "injury_ctx": home_injury_ctx.get(p["name"])})
    for p in away_roster:
        all_players.append({**p, "team": away, "is_home": False, "opponent": home,
                             "injury_ctx": away_injury_ctx.get(p["name"])})

    # Filter to focus players if specified
    if focus_players:
        focus_lower = [f.lower() for f in focus_players]
        all_players = [p for p in all_players if p["name"].lower() in focus_lower]

    recommendations = []
    analyzed_count = 0

    if verbose:
        print(f"\n🔮 Running predictions...")

    for player_info in all_players:
        name = player_info["name"]
        team = player_info["team"]
        opponent = player_info["opponent"]
        position = player_info.get("position", "F")
        is_home = player_info["is_home"]
        injury_ctx = player_info["injury_ctx"]

        # Get rolling stats (may use cache)
        rolling = get_player_rolling_stats(name, n_games=10)
        if "error" in rolling:
            continue
        
        # Skip low-minute players
        if rolling.get("min_avg", 0) < MIN_MINUTES:
            continue

        analyzed_count += 1

        # Determine which props to check for this position
        prop_types = PROP_TYPES_BY_POSITION.get(position, PROP_TYPES_BY_POSITION["default"])

        for prop_type in prop_types:
            line = estimate_prop_line(rolling, prop_type)
            if line is None:
                continue

            result = PREDICTOR.predict_prop(
                player=name,
                team=team,
                opponent=opponent,
                prop_type=prop_type,
                line=line,
                date=date,
                is_home=is_home,
                injury_context=injury_ctx,
            )

            if "error" in result:
                continue

            if result["recommendation"] == "PASS":
                continue

            if result["confidence"] < MIN_INCLUDE_CONFIDENCE:
                continue

            rec = {
                "player": name,
                "team": team,
                "opponent": opponent,
                "position": position,
                "is_home": is_home,
                "prop_type": prop_type,
                "line": line,
                "predicted": result["predicted"],
                "edge": result["edge"],
                "confidence": result["confidence"],
                "recommendation": result["recommendation"],
                "reasoning": result["reasoning"],
                "injury_flag": injury_ctx is not None,
                "injury_notes": (injury_ctx or {}).get("notes", ""),
                "features": result["features"],
            }
            recommendations.append(rec)

    # Sort by confidence descending
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)

    if verbose:
        print(f"\n   Analyzed {analyzed_count} players → {len(recommendations)} prop recs")

    return recommendations


def print_prop_recs(recs: list, title: str = "Top Prop Opportunities"):
    """Pretty-print prop recommendations."""
    print(f"\n{'='*65}")
    print(f"🎯 {title}")
    print(f"{'='*65}")
    
    if not recs:
        print("  No high-confidence props found.")
        return

    for i, r in enumerate(recs, 1):
        injury_tag = " 🚑" if r["injury_flag"] else ""
        home_tag = " (home)" if r["is_home"] else " (away)"
        print(f"\n{i}. {r['player']} ({r['team']}{home_tag}){injury_tag}")
        print(f"   {r['prop_type'].upper()} {r['recommendation']} {r['line']} "
              f"| Pred: {r['predicted']:.1f} | Edge: {r['edge']:+.1f} | "
              f"Conf: {r['confidence']*100:.1f}%")
        if r["injury_notes"]:
            print(f"   ⚕️  {r['injury_notes']}")
        # Print first 100 chars of reasoning
        reasoning_short = r["reasoning"][:120] + "..." if len(r["reasoning"]) > 120 else r["reasoning"]
        print(f"   {reasoning_short}")


if __name__ == "__main__":
    home = sys.argv[1] if len(sys.argv) > 1 else "SAC"
    away = sys.argv[2] if len(sys.argv) > 2 else "PHX"
    date = sys.argv[3] if len(sys.argv) > 3 else datetime.now().strftime("%Y-%m-%d")
    
    # For the PHX vs SAC test, add historical injury context
    injured_home = []
    injured_away = []
    
    if (home.upper() == "SAC" and away.upper() == "PHX") or \
       (home.upper() == "PHX" and away.upper() == "SAC"):
        # Historical context from the game we're testing
        if home.upper() == "SAC":
            injured_home = ["Domantas Sabonis", "Alex Cardwell"]
        else:
            injured_away = ["Domantas Sabonis", "Alex Cardwell"]

    recs = analyze_props_for_game(
        home=home,
        away=away,
        date=date,
        injured_home=injured_home,
        injured_away=injured_away,
        verbose=True,
    )
    
    print_prop_recs(recs, f"Props: {away} @ {home} | {date}")
    
    # Also show top 5 summary
    top5 = recs[:5]
    if top5:
        print(f"\n📊 Top 5 summary:")
        for r in top5:
            print(f"   {r['player']} {r['prop_type'].upper()} {r['recommendation']} {r['line']} "
                  f"({r['confidence']*100:.0f}% conf, edge {r['edge']:+.1f})")

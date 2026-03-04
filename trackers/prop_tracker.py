"""
Prop tracker — logs recommended bets and scores them after the game.
This is the self-learning core: records what we said, what happened, and why.
"""
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "props"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LESSONS_DIR = Path(__file__).parent.parent / "data" / "lessons"
LESSONS_DIR.mkdir(parents=True, exist_ok=True)


def log_prop(game_code: str, player: str, prop_type: str, line: float,
             direction: str, odds: int, confidence: float, reasoning: str) -> str:
    """
    Log a prop recommendation before the game.
    Returns a unique prop_id.
    """
    prop_id = f"{game_code}_{player.replace(' ', '_')}_{prop_type}_{datetime.now().strftime('%H%M%S')}"
    record = {
        "prop_id": prop_id,
        "game_code": game_code,
        "player": player,
        "prop_type": prop_type,  # e.g. "rebounds", "points", "PRA", "double_double"
        "line": line,
        "direction": direction,  # "over" or "under"
        "odds": odds,
        "confidence": confidence,  # 0.0 - 1.0
        "reasoning": reasoning,
        "logged_at": datetime.now().isoformat(),
        "result": None,      # filled in after game
        "actual_value": None,
        "hit": None,
        "notes": None,
    }
    path = DATA_DIR / f"{prop_id}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Logged prop: {player} {direction} {line} {prop_type} @ {odds}")
    return prop_id


def score_prop(prop_id: str, actual_value: float, notes: str = ""):
    """
    Score a prop after the game with the actual result.
    """
    # Find the prop file
    matches = list(DATA_DIR.glob(f"{prop_id}*.json"))
    if not matches:
        # Try finding by partial match
        matches = [p for p in DATA_DIR.glob("*.json") if prop_id in p.stem]
    
    if not matches:
        print(f"Prop {prop_id} not found.")
        return

    path = matches[0]
    with open(path) as f:
        record = json.load(f)

    line = record["line"]
    direction = record["direction"]
    
    if direction == "over":
        hit = actual_value > line
    elif direction == "under":
        hit = actual_value < line
    else:
        hit = None  # custom props like double-double

    record["actual_value"] = actual_value
    record["hit"] = hit
    record["result"] = "WIN" if hit else "LOSS"
    record["scored_at"] = datetime.now().isoformat()
    record["notes"] = notes

    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    
    status = "✅ HIT" if hit else "❌ MISS"
    print(f"{status}: {record['player']} {direction} {line} {record['prop_type']} — actual: {actual_value}")
    
    # Auto-log lesson if miss
    if not hit:
        log_lesson(
            game_code=record["game_code"],
            category="prop_miss",
            what_happened=f"{record['player']} {direction} {line} {record['prop_type']} — actual: {actual_value}",
            why_it_failed=notes or "No post-game analysis recorded",
            fix_for_next_time="Review player recent form and matchup-specific history",
        )


def get_hit_rates() -> dict:
    """Compute overall and per-prop-type hit rates from all scored props."""
    all_props = list(DATA_DIR.glob("*.json"))
    results = {"total": 0, "hits": 0, "by_type": {}, "by_player": {}}
    
    for path in all_props:
        with open(path) as f:
            p = json.load(f)
        if p.get("hit") is None:
            continue
        
        results["total"] += 1
        if p["hit"]:
            results["hits"] += 1
        
        ptype = p["prop_type"]
        if ptype not in results["by_type"]:
            results["by_type"][ptype] = {"total": 0, "hits": 0}
        results["by_type"][ptype]["total"] += 1
        if p["hit"]:
            results["by_type"][ptype]["hits"] += 1

        player = p["player"]
        if player not in results["by_player"]:
            results["by_player"][player] = {"total": 0, "hits": 0}
        results["by_player"][player]["total"] += 1
        if p["hit"]:
            results["by_player"][player]["hits"] += 1

    # Compute rates
    if results["total"] > 0:
        results["overall_hit_rate"] = round(results["hits"] / results["total"], 2)
    for ptype, data in results["by_type"].items():
        data["hit_rate"] = round(data["hits"] / data["total"], 2) if data["total"] > 0 else 0
    for player, data in results["by_player"].items():
        data["hit_rate"] = round(data["hits"] / data["total"], 2) if data["total"] > 0 else 0

    return results


def log_lesson(game_code: str, category: str, what_happened: str,
               why_it_failed: str, fix_for_next_time: str):
    """Log a lesson learned to the lessons database."""
    lessons_path = LESSONS_DIR / "lessons.json"
    
    if lessons_path.exists():
        with open(lessons_path) as f:
            lessons = json.load(f)
    else:
        lessons = []

    lesson = {
        "id": len(lessons) + 1,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "game_code": game_code,
        "category": category,
        "what_happened": what_happened,
        "why_it_failed": why_it_failed,
        "fix_for_next_time": fix_for_next_time,
    }
    lessons.append(lesson)
    
    with open(lessons_path, "w") as f:
        json.dump(lessons, f, indent=2)
    
    print(f"Lesson logged: [{category}] {what_happened[:60]}...")


def get_lessons(category: str = None, limit: int = 10) -> list:
    """Retrieve lessons, optionally filtered by category."""
    lessons_path = LESSONS_DIR / "lessons.json"
    if not lessons_path.exists():
        return []
    with open(lessons_path) as f:
        lessons = json.load(f)
    if category:
        lessons = [l for l in lessons if l["category"] == category]
    return lessons[-limit:]


if __name__ == "__main__":
    # Seed the lessons from tonight's PHX vs SAC game
    log_lesson(
        game_code="PHX_SAC_20260303",
        category="total_analysis",
        what_happened="Recommended Under 217.5. Game went over — ended around 220-230.",
        why_it_failed=(
            "Cited SAC team-wide Under trends without checking H2H history. "
            "Oct 22 PHX vs SAC went to 236 total. Q2 ran at 5.0 pts/min (62 combined). "
            "Didn't flag early enough when Q1 hit 52 combined."
        ),
        fix_for_next_time=(
            "1. ALWAYS pull H2H game totals for specific matchup before recommending over/under. "
            "2. When Q1 exceeds 50 combined pts, flag Under as HIGH RISK immediately. "
            "3. Do not rely on season-wide team trends — matchup-specific pace is more predictive. "
            "4. Check if either team has a tendency to run hot in specific quarter matchups."
        ),
    )
    log_lesson(
        game_code="PHX_SAC_20260303",
        category="prop_analysis",
        what_happened="Clifford 20+ PRA recommended. He had 2 pts in 24 min (PRA ~8 at that point).",
        why_it_failed=(
            "Assumed hot streak would continue without checking Clifford's stats vs PHX specifically. "
            "Did not pull live box score mid-game — assumed he was performing instead of checking."
        ),
        fix_for_next_time=(
            "1. Pull live box score every time user asks for an update — never assume. "
            "2. Check player prop hit rates in this SPECIFIC matchup, not just recent form. "
            "3. For streak-based props, note that streaks can end in any game."
        ),
    )
    log_lesson(
        game_code="PHX_SAC_20260303",
        category="process",
        what_happened="Did not flag cash-out opportunity when Under was still likely (around Q1 end).",
        why_it_failed="No systematic trigger for flagging cash-out windows.",
        fix_for_next_time=(
            "When a parlay leg (especially the total) starts trending bad, "
            "proactively mention the cash-out option and estimate EV of cashing vs riding."
        ),
    )
    print("\n✅ Initial lessons seeded from PHX vs SAC 2026-03-03")
    print(f"\nAll lessons:")
    for l in get_lessons():
        print(f"\n[{l['date']}] {l['category']}: {l['what_happened'][:80]}")
        print(f"  Fix: {l['fix_for_next_time'][:80]}")

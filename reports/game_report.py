"""
Pre-game report generator — pulls all data and generates a full analysis.
This is what gets run before you place bets.
"""
import sys
import json
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.nba_api import get_live_scoreboard, get_game_summary
from collectors.h2h_collector import get_h2h_report, load_h2h
from trackers.prop_tracker import get_lessons


def get_injuries_espn(team_tricode: str) -> list:
    """Pull injury report from ESPN for a team."""
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        r = requests.get(url, timeout=10)
        data = r.json()
        injuries = []
        for item in data.get("injuries", []):
            if item.get("team", {}).get("abbreviation", "").upper() == team_tricode.upper():
                for inj in item.get("injuries", []):
                    injuries.append({
                        "player": inj.get("athlete", {}).get("displayName", ""),
                        "status": inj.get("status", ""),
                        "detail": inj.get("type", {}).get("description", ""),
                    })
        return injuries
    except:
        return []


def generate_matchup_report(team1: str, team2: str, total_line: float = None) -> str:
    """
    Generate a full pre-game analysis report for a matchup.
    This is the main output function.
    """
    lines = [
        f"# 🏀 {team1} vs {team2} — Pre-Game Analysis",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}*",
        "",
    ]

    # === H2H HISTORY ===
    lines.append("## 📊 H2H History")
    h2h_data = load_h2h(team1, team2)
    if h2h_data["games"]:
        stats = h2h_data.get("stats", {})
        lines += [
            f"**{stats.get('games_played', 0)} games this season:**",
            f"- Avg total: **{stats.get('avg_total', 'N/A')}** | Range: {stats.get('min_total', 0)}-{stats.get('max_total', 0)}",
            f"- Last 3 avg: {stats.get('recent_3_avg', 'N/A')}",
            f"- Over 220 rate: {stats.get('over_rate', 0)*100:.0f}%",
            f"",
            f"**Quarter pace (avg combined):**",
            f"Q1: {stats.get('avg_Q1', 0)} | Q2: {stats.get('avg_Q2', 0)} | Q3: {stats.get('avg_Q3', 0)} | Q4: {stats.get('avg_Q4', 0)}",
            "",
            "**Recent games:**",
        ]
        for g in h2h_data["games"][-5:][::-1]:
            lines.append(f"- {g.get('date', 'N/A')}: Total {g.get('total', '?')} | {g.get('winner', '?')} win")
    else:
        lines.append("*No H2H data yet — this is the first tracked game.*")
        lines.append("⚠️ WARNING: Without H2H data, total analysis is less reliable.")
    lines.append("")

    # === TOTAL ANALYSIS ===
    if total_line:
        lines.append(f"## 💰 Total Analysis (Line: {total_line})")
        if h2h_data["games"]:
            stats = h2h_data.get("stats", {})
            avg = stats.get("avg_total", 0)
            diff = avg - total_line
            if diff > 5:
                rec = f"⬆️ LEAN OVER — H2H avg ({avg}) is {diff:.0f} pts above line"
            elif diff < -5:
                rec = f"⬇️ LEAN UNDER — H2H avg ({avg}) is {abs(diff):.0f} pts below line"
            else:
                rec = f"⚖️ TOSS-UP — H2H avg ({avg}) is near the line"
            lines.append(rec)
            
            # Q1 warning
            q1_avg = stats.get("avg_Q1", 0)
            if q1_avg > 52:
                lines.append(f"⚠️ HIGH PACE WARNING: Avg Q1 in this matchup is {q1_avg} combined — Under bets are higher risk")
        else:
            lines.append("*Cannot assess — no H2H history loaded*")
        lines.append("")

    # === LESSONS REMINDER ===
    lines.append("## 📚 Relevant Lessons Learned")
    total_lessons = get_lessons("total_analysis", limit=3)
    prop_lessons = get_lessons("prop_analysis", limit=2)
    process_lessons = get_lessons("process", limit=1)
    all_lessons = total_lessons + prop_lessons + process_lessons
    
    if all_lessons:
        for l in all_lessons:
            lines.append(f"**[{l['category']}]** {l['what_happened'][:100]}")
            lines.append(f"→ *{l['fix_for_next_time'][:120]}*")
            lines.append("")
    else:
        lines.append("*No lessons recorded yet.*")
        lines.append("")

    # === CHECKLIST ===
    lines += [
        "## ✅ Pre-Bet Checklist",
        "- [ ] H2H total history pulled and reviewed",
        "- [ ] Q1 pace in H2H matchups checked",
        "- [ ] Player prop — checked in THIS SPECIFIC matchup, not just recent form",
        "- [ ] Injury report verified (both teams)",
        "- [ ] If Q1 > 50 combined pts: flag Under as HIGH RISK",
        "- [ ] Cash-out trigger set: if total leg goes bad mid-game, alert user",
        "- [ ] Live box score pulled mid-game (never assume player stats)",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    team1 = sys.argv[1].upper() if len(sys.argv) > 1 else "PHX"
    team2 = sys.argv[2].upper() if len(sys.argv) > 2 else "SAC"
    total = float(sys.argv[3]) if len(sys.argv) > 3 else None
    print(generate_matchup_report(team1, team2, total))

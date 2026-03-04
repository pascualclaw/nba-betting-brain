"""
Matchup analyzer — combines ALL data sources into one pre-game intelligence report.

Order of operations (the right way to analyze a game):
1. Pull advanced stats (ORTG/DRTG/PACE) → project the total mathematically
2. Pull H2H history → validate projection against actual results
3. Pull injuries → adjust projection for missing players
4. Pull betting splits → check if public/sharp money aligns
5. Pull live odds → find best line value
6. Generate ranked bet recommendations with confidence scores
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.nba_api import get_live_scoreboard, get_game_summary
from collectors.h2h_collector import load_h2h, get_h2h_report
from collectors.rotowire import format_injury_report, get_notable_absences
from trackers.prop_tracker import get_lessons

# Try importing optional sources
try:
    from collectors.advanced_stats import project_game_total, get_team_advanced_stats
    HAS_ADVANCED = True
except:
    HAS_ADVANCED = False

try:
    from collectors.odds_api import get_live_odds, parse_game_lines, get_player_props, parse_player_props
    HAS_ODDS = True
except:
    HAS_ODDS = False

try:
    from collectors.action_network import format_splits_report
    HAS_ACTION = True
except:
    HAS_ACTION = False


def full_game_analysis(team1: str, team2: str, total_line: float = None) -> str:
    """
    The complete pre-game analysis. Run this before every bet.
    """
    t1, t2 = team1.upper(), team2.upper()
    lines = [
        f"# 🏀 FULL ANALYSIS: {t1} vs {t2}",
        f"=" * 50,
        "",
    ]

    # ── 1. MATHEMATICAL TOTAL PROJECTION ──────────────────
    lines.append("## 1️⃣ Mathematical Total Projection (ORTG/DRTG/PACE)")
    if HAS_ADVANCED:
        try:
            stats = get_team_advanced_stats()
            proj = project_game_total(t1, t2, stats)
            if "error" not in proj:
                lines += [
                    f"**Projected total: {proj['projected_total']}**",
                    f"  {t1}: {proj['team1_projected_pts']} (ORTG {proj['team1_ortg']} / DRTG {proj['team1_drtg']} / Net {proj['team1_net_rtg']})",
                    f"  {t2}: {proj['team2_projected_pts']} (ORTG {proj['team2_ortg']} / DRTG {proj['team2_drtg']} / Net {proj['team2_net_rtg']})",
                    f"  Avg pace: {proj['avg_pace']}",
                ]
                if total_line:
                    diff = proj["projected_total"] - total_line
                    if abs(diff) > 3:
                        direction = "OVER" if diff > 0 else "UNDER"
                        lines.append(f"\n  📊 Model says **{direction}** — {abs(diff):.1f} pts vs posted line {total_line}")
                    else:
                        lines.append(f"\n  📊 Model is near line ({proj['projected_total']} vs {total_line}) — no strong lean")
            else:
                lines.append(f"  ⚠️ {proj['error']}")
        except Exception as e:
            lines.append(f"  ⚠️ Could not load advanced stats: {e}")
    else:
        lines.append("  ⚠️ Advanced stats module not available")
    lines.append("")

    # ── 2. H2H HISTORY ────────────────────────────────────
    lines.append("## 2️⃣ Head-to-Head History")
    h2h = load_h2h(t1, t2)
    if h2h["games"]:
        stats_h2h = h2h.get("stats", {})
        lines += [
            f"**{stats_h2h.get('games_played', 0)} games | Avg total: {stats_h2h.get('avg_total', 0)} | Range: {stats_h2h.get('min_total', 0)}-{stats_h2h.get('max_total', 0)}**",
            f"  Q1 avg: {stats_h2h.get('avg_Q1', 0)} | Q2: {stats_h2h.get('avg_Q2', 0)} | Q3: {stats_h2h.get('avg_Q3', 0)} | Q4: {stats_h2h.get('avg_Q4', 0)}",
            f"  Over 220 rate: {stats_h2h.get('over_rate', 0)*100:.0f}% | Last 3 avg: {stats_h2h.get('recent_3_avg', 0)}",
        ]

        # Pace warning
        q1_avg = stats_h2h.get("avg_Q1", 0)
        if q1_avg > 52:
            lines.append(f"\n  ⚠️ HIGH PACE WARNING: These teams avg {q1_avg} combined in Q1 — Under bets carry elevated risk")

        lines.append("\n  **Recent results:**")
        for g in h2h["games"][-5:][::-1]:
            lines.append(f"  - {g.get('date', '?')}: {g.get('away', '?')}@{g.get('home', '?')} | Total: {g.get('total', '?')} | Winner: {g.get('winner', '?')}")
    else:
        lines.append("  ⚠️ NO H2H DATA — first tracked game. Treat total projection with extra caution.")
    lines.append("")

    # ── 3. INJURY REPORT ──────────────────────────────────
    lines.append("## 3️⃣ Injury Report")
    try:
        report = format_injury_report(t1, t2)
        lines.append(report)
        
        # Quantify impact
        for team in [t1, t2]:
            absences = get_notable_absences(team)
            if absences:
                names = ", ".join(a["player"] for a in absences)
                lines.append(f"\n  💥 {team} missing: {names}")
                lines.append(f"  → Check who absorbs their usage/minutes — those players' props have value")
    except Exception as e:
        lines.append(f"  ⚠️ Could not load injuries: {e}")
    lines.append("")

    # ── 4. BETTING SPLITS ─────────────────────────────────
    lines.append("## 4️⃣ Public Money & Sharp Signals")
    if HAS_ACTION:
        try:
            splits_report = format_splits_report(t1, t2)
            lines.append(splits_report)
        except Exception as e:
            lines.append(f"  ⚠️ Action Network unavailable: {e}")
    else:
        lines.append("  (Action Network module not configured)")
    lines.append("")

    # ── 5. LESSONS REMINDER ───────────────────────────────
    lines.append("## 5️⃣ Lessons From Past Games")
    all_lessons = (
        get_lessons("total_analysis", limit=2) +
        get_lessons("prop_analysis", limit=2) +
        get_lessons("process", limit=1)
    )
    if all_lessons:
        for l in all_lessons:
            lines.append(f"**[{l['category']}]** {l['what_happened'][:100]}")
            lines.append(f"  → *{l['fix_for_next_time'][:120]}*")
            lines.append("")
    else:
        lines.append("  No lessons logged yet.")
    lines.append("")

    # ── 6. PRE-BET CHECKLIST ─────────────────────────────
    lines += [
        "## ✅ Pre-Bet Checklist",
        f"- [ ] Mathematical projection reviewed (section 1)",
        f"- [ ] H2H history checked — is avg total near the line?",
        f"- [ ] Q1 H2H avg > 52? If yes, Under flagged as HIGH RISK",
        f"- [ ] Injury impact assessed — who absorbs missing players' usage?",
        f"- [ ] Sharp signals checked — any public/money discrepancies?",
        f"- [ ] Live box score pulled mid-game (never assume stats)",
        f"- [ ] Cash-out trigger set for any total leg",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    t1 = sys.argv[1].upper() if len(sys.argv) > 1 else "PHX"
    t2 = sys.argv[2].upper() if len(sys.argv) > 2 else "SAC"
    total = float(sys.argv[3]) if len(sys.argv) > 3 else None
    print(full_game_analysis(t1, t2, total))

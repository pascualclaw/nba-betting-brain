"""
Parlay Validator — Pre-bet reasoning checks.

Prevents systematic errors in parlay/SGP construction:
1. Never combine ML + spread on same team (redundant legs)
2. Validates that SGP legs are actually correlated, not anti-correlated
3. B2B fatigue flag for pace-control assumptions
4. Spread math: correctly calculates swing needed for +/- spread bets
5. Pre-bet checklist before any recommendation

Usage:
    from analyzers.parlay_validator import validate_parlay, spread_cover_status, sgp_correlation

Author: Jarvis (corrective build after NYK@LAC SGP error, 2026-03-09)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

log = logging.getLogger(__name__)


# ── Spread math (CORRECTED) ────────────────────────────────────────────────

def spread_cover_status(
    team_score: int,
    opponent_score: int,
    spread: float,
    is_favorite: bool = None,
) -> dict:
    """
    Calculate spread cover status with correct math.

    Args:
        team_score: current score of the team whose spread we're checking
        opponent_score: opponent's current score
        spread: the spread value (NEGATIVE = team is favorite, POSITIVE = underdog)
        is_favorite: optional override; inferred from spread sign if None

    Returns dict with:
        - covering: bool — whether team is currently covering
        - margin: current score margin (positive = team leading)
        - cover_threshold: how much team needs to win/lose by to cover
        - swing_needed: how many points the team needs to swing (0 if already covering)
        - description: human-readable status
    """
    margin = team_score - opponent_score  # positive = team leading

    if spread < 0:
        # Team is FAVORITE (e.g., -2.5 means team must win by 3+)
        cover_threshold = abs(spread)  # must win by more than this
        covering = margin > cover_threshold
        swing_needed = max(0, cover_threshold - margin + 0.5)  # +0.5 for half-point
        desc = f"Need to WIN by >{cover_threshold:.1f}. Current margin: {margin:+d}."
    else:
        # Team is UNDERDOG (e.g., +2.5 means team can lose by up to 2)
        cover_threshold = spread  # can lose by up to this amount
        covering = margin > -cover_threshold  # covering if deficit < spread
        # Swing needed = how much the team needs to CLOSE the gap
        # If down 9 and getting +2.5 → need deficit to drop to < 2.5 → swing = 9 - 2.5 = 6.5
        if margin < 0:
            swing_needed = max(0, abs(margin) - cover_threshold)
        else:
            swing_needed = 0  # already winning = definitely covers
        desc = f"Need to LOSE by <{cover_threshold:.1f} (or win). Current margin: {margin:+d}."

    return {
        "covering": covering,
        "margin": margin,
        "spread": spread,
        "cover_threshold": cover_threshold,
        "swing_needed": swing_needed,
        "description": desc,
        "is_favorite": spread < 0,
    }


# ── Leg definitions ────────────────────────────────────────────────────────

@dataclass
class ParalayLeg:
    """Represents a single bet leg."""
    game: str           # e.g. "NYK @ LAC"
    team: str           # team abbreviation this leg is about
    bet_type: str       # "ml", "spread", "total_over", "total_under", "h1_spread", etc.
    line: float         # the line value (spread or total)
    odds: int           # American odds (e.g., -110, +150)
    on_b2b: bool = False       # is this team on back-to-back?
    pace_tier: str = "avg"     # "fast", "avg", "slow" based on team's avg total


# ── SGP Correlation Engine ─────────────────────────────────────────────────

def sgp_correlation(legs: List[ParalayLeg]) -> dict:
    """
    Determines if SGP legs are correlated, anti-correlated, or neutral.

    RULES:
    - Favorite spread (-) + Under on SAME game: POSITIVE (team controls pace = Under)
    - Underdog spread (+) + Under on SAME game: ANTI-CORRELATED (upset = chaos = usually Over)
    - Favorite spread (-) + Over on SAME game: ANTI-CORRELATED (blowout = garbage time = Under)
    - Underdog spread (+) + Over on SAME game: NEUTRAL/WEAK POSITIVE (competitive game)
    - B2B team + pace-control assumption: FLAG (tired team can't control pace)
    - ML + spread on same team: REDUNDANT (forbidden)

    Returns: {
        "valid": bool,
        "correlation": "positive" | "anti" | "neutral",
        "score": float (-1 to 1),
        "flags": list of warning strings,
        "recommendation": str
    }
    """
    flags = []
    correlation_score = 0.0

    # Group legs by game
    games = {}
    for leg in legs:
        games.setdefault(leg.game, []).append(leg)

    for game, game_legs in games.items():
        bet_types = {l.bet_type for l in game_legs}
        teams = {l.team for l in game_legs}

        # Rule 1: ML + spread same team = FORBIDDEN
        ml_teams = {l.team for l in game_legs if l.bet_type == "ml"}
        spread_teams = {l.team for l in game_legs if l.bet_type == "spread"}
        redundant = ml_teams & spread_teams
        if redundant:
            flags.append(f"❌ FORBIDDEN: ML + spread on same team ({redundant}) in {game}")
            return {"valid": False, "correlation": "invalid", "score": -2.0,
                    "flags": flags, "recommendation": "Remove redundant ML/spread leg."}

        # Rule 2: Spread + Under/Over correlation check
        spread_legs = [l for l in game_legs if l.bet_type == "spread"]
        total_legs = [l for l in game_legs if l.bet_type in ("total_over", "total_under")]

        for s_leg in spread_legs:
            for t_leg in total_legs:
                is_fav = s_leg.line < 0  # negative spread = favorite
                is_under = t_leg.bet_type == "total_under"

                if is_fav and is_under:
                    # POSITIVE correlation: favorite controlling game = low scoring
                    correlation_score += 0.6
                    if s_leg.on_b2b:
                        flags.append(f"⚠️ B2B: {s_leg.team} is on back-to-back — pace control is REDUCED. "
                                     f"Favorite + Under correlation weakened.")
                        correlation_score -= 0.3
                    else:
                        flags.append(f"✅ POSITIVE: {s_leg.team} favorite ({s_leg.line}) + Under = correlated (controls pace)")

                elif not is_fav and is_under:
                    # ANTI-CORRELATED: underdog winning = chaos = usually Over
                    correlation_score -= 0.5
                    flags.append(f"⚠️ ANTI-CORRELATED: {s_leg.team} underdog ({s_leg.line:+.1f}) + Under "
                                 f"= these fight each other. Underdog winning = competitive game = usually OVER.")

                elif is_fav and not is_under:
                    # Favorite + Over: anti-correlated (blowout = garbage time = Under)
                    correlation_score -= 0.3
                    flags.append(f"⚠️ WEAK ANTI: {s_leg.team} favorite ({s_leg.line}) + Over "
                                 f"= if team dominates, garbage time reduces scoring.")

                elif not is_fav and not is_under:
                    # Underdog + Over: neutral/slightly positive (close game = more possessions)
                    correlation_score += 0.2
                    flags.append(f"ℹ️ NEUTRAL: {s_leg.team} underdog ({s_leg.line:+.1f}) + Over "
                                 f"= close competitive game expected.")

        # Rule 3: B2B flag for any pace-assumption legs
        for leg in game_legs:
            if leg.on_b2b and leg.bet_type in ("total_under", "spread"):
                flags.append(f"🔴 B2B FLAG: {leg.team} on back-to-back. "
                              f"Do NOT assume they control pace. Under/spread assumptions weakened.")

    # Determine overall correlation label
    if correlation_score >= 0.4:
        corr_label = "positive"
        recommendation = "Legs are correlated. SGP is reasonable."
    elif correlation_score <= -0.3:
        corr_label = "anti"
        recommendation = "⛔ ANTI-CORRELATED: These legs work AGAINST each other. Avoid this SGP."
    else:
        corr_label = "neutral"
        recommendation = "Legs are weakly correlated. Proceed with caution."

    return {
        "valid": correlation_score > -0.3,
        "correlation": corr_label,
        "score": round(correlation_score, 2),
        "flags": flags,
        "recommendation": recommendation,
    }


# ── Pre-bet Checklist ──────────────────────────────────────────────────────

def validate_parlay(legs: List[ParalayLeg], min_ev_pct: float = 3.0) -> dict:
    """
    Full pre-bet validation checklist.

    Runs all checks and returns a structured report.
    """
    report = {"passed": True, "checks": [], "flags": [], "recommendation": ""}

    def check(name: str, passed: bool, detail: str):
        report["checks"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            report["passed"] = False
            report["flags"].append(f"❌ {name}: {detail}")

    # Check 1: No ML + spread same team
    ml_teams = {l.team for l in legs if l.bet_type == "ml"}
    spread_teams = {l.team for l in legs if l.bet_type == "spread"}
    redundant = ml_teams & spread_teams
    check(
        "No ML+spread same team",
        len(redundant) == 0,
        f"Redundant legs: {redundant}" if redundant else "OK"
    )

    # Check 2: SGP correlation
    same_game_legs = {}
    for leg in legs:
        same_game_legs.setdefault(leg.game, []).append(leg)

    for game, glgs in same_game_legs.items():
        if len(glgs) > 1:
            corr = sgp_correlation(glgs)
            check(
                f"SGP correlation ({game})",
                corr["valid"],
                corr["recommendation"]
            )
            report["flags"].extend(corr["flags"])

    # Check 3: B2B pace control warning
    for leg in legs:
        if leg.on_b2b and leg.bet_type in ("total_under",):
            check(
                f"B2B+Under warning ({leg.team})",
                False,
                f"{leg.team} is on B2B — tired teams don't control pace. Under risky."
            )

    # Check 4: Spread direction sanity
    for leg in legs:
        if leg.bet_type == "spread":
            direction = "FAVORITE" if leg.line < 0 else "UNDERDOG"
            covers_if = (f"wins by >{abs(leg.line)}" if leg.line < 0
                         else f"loses by <{leg.line} OR wins")
            check(
                f"Spread direction ({leg.team} {leg.line:+.1f})",
                True,  # informational only
                f"{leg.team} is {direction}. Covers if: {covers_if}"
            )

    # Summary
    if report["passed"]:
        report["recommendation"] = "✅ Parlay passed all checks."
    else:
        report["recommendation"] = "⛔ Parlay FAILED checks. Review flags before betting."

    return report


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with the NYK@LAC mistake
    print("=== NYK@LAC SGP Reconstruction (the bad bet) ===\n")

    legs = [
        ParalayLeg(game="NYK@LAC", team="NYK", bet_type="spread",
                   line=+2.5, odds=-105, on_b2b=True, pace_tier="slow"),
        ParalayLeg(game="NYK@LAC", team="NYK", bet_type="total_under",
                   line=220.5, odds=-110, on_b2b=True, pace_tier="slow"),
    ]

    result = validate_parlay(legs)
    print("VALIDATION REPORT:")
    for c in result["checks"]:
        icon = "✅" if c["passed"] else "❌"
        print(f"  {icon} {c['name']}: {c['detail']}")
    print(f"\nFlags:")
    for f in result["flags"]:
        print(f"  {f}")
    print(f"\nRecommendation: {result['recommendation']}")

    print("\n\n=== Spread math examples ===")
    # Correct example
    status = spread_cover_status(team_score=41, opponent_score=47, spread=+2.5)
    print(f"\nNYK +2.5, score 41-47: {status['description']}")
    print(f"  Currently covering: {status['covering']} | Swing needed: {status['swing_needed']:.1f}")

    status2 = spread_cover_status(team_score=109, opponent_score=98, spread=-2.5)
    print(f"\nBOS -2.5, score 109-98: {status2['description']}")
    print(f"  Currently covering: {status2['covering']} | Swing needed: {status2['swing_needed']:.1f}")

    print("\n\n=== CORRECT bet (BOS + Under) for comparison ===")
    good_legs = [
        ParalayLeg(game="BOS@CLE", team="BOS", bet_type="spread",
                   line=-2.5, odds=-118, on_b2b=False, pace_tier="slow"),
        ParalayLeg(game="BOS@CLE", team="BOS", bet_type="total_under",
                   line=223.5, odds=-110, on_b2b=False, pace_tier="slow"),
    ]
    good_result = validate_parlay(good_legs)
    for f in good_result["flags"]:
        print(f"  {f}")
    print(f"\nRecommendation: {good_result['recommendation']}")

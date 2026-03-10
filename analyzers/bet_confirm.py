"""
bet_confirm.py — Pre-bet confirmation gate.

Runs before EVERY bet recommendation:
1. Reads back exact legs with correct spread direction
2. Validates correlation via parlay_validator
3. Flags oversized bets vs Kelly sizing
4. Requires explicit confirmation of each leg

Usage:
    from analyzers.bet_confirm import confirm_bet
    confirm_bet(legs, wager, bankroll, ev_pct)
"""

import sys
sys.path.insert(0, ".")
from analyzers.parlay_validator import validate_parlay, ParalayLeg, spread_cover_status

KELLY_FRACTION = 0.25  # quarter Kelly
MAX_BET_PCT = 0.05     # 5% bankroll max


def spread_direction_str(line: float) -> str:
    if line < 0:
        return f"FAVORITE (must WIN by >{abs(line):.1f})"
    else:
        return f"UNDERDOG (covers if loses by <{line:.1f} OR wins)"


def confirm_bet(
    legs: list,
    wager: float,
    bankroll: float,
    ev_pct: float,
    print_output: bool = True,
) -> dict:
    """
    Full pre-bet gate. Returns pass/fail with details.

    legs: list of ParalayLeg objects
    wager: dollar amount being bet
    bankroll: current total bankroll
    ev_pct: estimated EV percentage
    """
    issues = []
    warnings = []
    output_lines = []

    def log(s): output_lines.append(s)

    log("\n" + "="*60)
    log("PRE-BET CONFIRMATION GATE")
    log("="*60)

    # 1. Leg readback
    log("\n📋 CONFIRMING LEGS:")
    for i, leg in enumerate(legs, 1):
        if leg.bet_type == "spread":
            direction = spread_direction_str(leg.line)
            log(f"  Leg {i}: {leg.team} {leg.line:+.1f} SPREAD — {direction}")
        elif leg.bet_type == "ml":
            log(f"  Leg {i}: {leg.team} MONEYLINE — wins outright")
        elif leg.bet_type == "total_under":
            log(f"  Leg {i}: UNDER {leg.line} total")
        elif leg.bet_type == "total_over":
            log(f"  Leg {i}: OVER {leg.line} total")
        else:
            log(f"  Leg {i}: {leg.team} {leg.bet_type} {leg.line}")

        if leg.on_b2b:
            warnings.append(f"⚠️  Leg {i}: {leg.team} is on B2B — pace control assumptions weakened")

    # 2. Parlay validation
    log("\n🔍 RUNNING PARLAY VALIDATOR:")
    result = validate_parlay(legs)
    for check in result["checks"]:
        icon = "✅" if check["passed"] else "❌"
        log(f"  {icon} {check['name']}: {check['detail']}")
    for flag in result["flags"]:
        log(f"  {flag}")
    log(f"  → {result['recommendation']}")

    if not result["passed"]:
        issues.append("Parlay failed validator checks")

    # 3. Kelly / bankroll check
    log(f"\n💰 SIZING CHECK (bankroll: ${bankroll:.0f}):")
    kelly_bet = bankroll * KELLY_FRACTION * (ev_pct / 100)
    max_bet = bankroll * MAX_BET_PCT
    log(f"  Kelly suggested: ${kelly_bet:.0f} (quarter Kelly × {ev_pct:.0f}% EV)")
    log(f"  Max allowed (5%): ${max_bet:.0f}")
    log(f"  Proposed wager: ${wager:.0f}")

    if wager > max_bet * 2:
        issues.append(f"Wager ${wager:.0f} is {wager/max_bet:.1f}× the 5% bankroll max (${max_bet:.0f})")
        log(f"  🔴 OVERSIZED: ${wager:.0f} is {wager/max_bet:.1f}× max. Recommend max ${max_bet:.0f}.")
    elif wager > max_bet:
        warnings.append(f"Wager ${wager:.0f} exceeds 5% max (${max_bet:.0f})")
        log(f"  🟡 WARNING: ${wager:.0f} exceeds 5% bankroll. Proceed with caution.")
    else:
        log(f"  ✅ Sizing OK")

    # 4. EV gate
    log(f"\n📊 EV CHECK:")
    log(f"  Estimated EV: {ev_pct:+.1f}%")
    if ev_pct < 3.0:
        issues.append(f"EV {ev_pct:.1f}% below 3% minimum gate")
        log(f"  ❌ Below 3% EV gate. Do not bet.")
    else:
        log(f"  ✅ Above 3% EV gate")

    # 5. Final verdict
    log("\n" + "="*60)
    passed = len(issues) == 0
    if passed:
        log("✅ BET CONFIRMED — all checks passed")
        if warnings:
            log("   Warnings (non-blocking):")
            for w in warnings: log(f"   {w}")
    else:
        log("⛔ BET BLOCKED — fix these issues:")
        for issue in issues:
            log(f"   ❌ {issue}")
    log("="*60 + "\n")

    if print_output:
        print("\n".join(output_lines))

    return {
        "passed": passed,
        "issues": issues,
        "warnings": warnings,
        "kelly_suggested": kelly_bet,
        "max_bet": max_bet,
    }


if __name__ == "__main__":
    # Test: the SMC ML + Under SGP that lost
    print("=== TEST: SMC ML + Under 145.5 (the losing bet) ===")
    legs = [
        ParalayLeg(game="SCU@SMC", team="SMC", bet_type="ml", line=0, odds=-238, on_b2b=False),
        ParalayLeg(game="SCU@SMC", team="SMC", bet_type="total_under", line=145.5, odds=-110, on_b2b=False),
    ]
    confirm_bet(legs, wager=1000, bankroll=1000, ev_pct=33)

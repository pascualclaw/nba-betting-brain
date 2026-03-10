"""
live_decision.py — Real-time cash out advisor.

Pulls live score from ESPN, calculates EV of holding vs cashing out,
outputs a clear recommendation WITH the current score shown.

RULE: Always call this before advising on any cash out decision.

Usage:
    python3 analyzers/live_decision.py --sport ncaab --team1 SMC --team2 SCU \
        --bet-type sgp --payout 2420 --wager 1000 --cashout 200 \
        --legs "SMC_ml" "under_145.5"
"""

import argparse
import requests
import json
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from scipy import stats

SPORT_ENDPOINTS = {
    "nba": "basketball/nba",
    "ncaab": "basketball/mens-college-basketball",
}

def fetch_live_score(sport: str, team1: str, team2: str) -> Optional[dict]:
    """Pull current live score from ESPN API."""
    endpoint = SPORT_ENDPOINTS.get(sport, "basketball/nba")
    for days_back in [0, 1]:
        date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        url = f"http://site.api.espn.com/apis/site/v2/sports/{endpoint}/scoreboard?dates={date}&limit=200"
        try:
            r = requests.get(url, timeout=8)
            data = r.json()
            for event in data.get("events", []):
                name = event.get("name", "").upper()
                if team1.upper() in name or team2.upper() in name:
                    comps = event.get("competitions", [{}])[0]
                    competitors = comps.get("competitors", [])
                    scores = {}
                    linescores = {}
                    for c in competitors:
                        abbr = c.get("team", {}).get("abbreviation", "?")
                        scores[abbr] = int(c.get("score", 0) or 0)
                        linescores[abbr] = [q.get("value", 0) for q in c.get("linescores", [])]
                    status = event.get("status", {}).get("type", {})
                    return {
                        "status": status.get("shortDetail", "?"),
                        "completed": status.get("completed", False),
                        "scores": scores,
                        "linescores": linescores,
                        "total": sum(scores.values()),
                        "event_name": event.get("name", "?"),
                        "date": date,
                    }
        except Exception as e:
            print(f"  ESPN fetch error: {e}", file=sys.stderr)
    return None


def parse_time_remaining(status_str: str, sport: str) -> float:
    """Parse time remaining in minutes from ESPN status string."""
    # e.g. "3:31 - 2nd" or "6:38 - 2nd" or "Final"
    if "Final" in status_str or "final" in status_str:
        return 0.0
    try:
        time_part = status_str.split(" - ")[0].strip()
        parts = time_part.split(":")
        mins = float(parts[0]) + float(parts[1]) / 60
        period = status_str.split(" - ")[-1].strip() if " - " in status_str else "1"

        if sport == "ncaab":
            # Two 20-min halves
            if "2nd" in period:
                return mins  # just remainder of 2nd half
            else:
                return mins + 20  # remainder of 1st + full 2nd
        else:
            # NBA: four 12-min quarters
            period_map = {"1st": 3, "2nd": 2, "3rd": 1, "4th": 0}
            quarters_remaining = period_map.get(period, 0)
            return mins + quarters_remaining * 12
    except:
        return 0.0


def project_total(current_total: int, time_elapsed_min: float,
                  time_remaining_min: float) -> float:
    """Project final total based on current pace."""
    if time_elapsed_min <= 0:
        return float(current_total)
    pace = current_total / time_elapsed_min
    return current_total + pace * time_remaining_min


def ev_hold(win_prob: float, payout: float, wager: float) -> float:
    """Expected value of holding the bet."""
    return win_prob * payout - (1 - win_prob) * wager


def analyze_cashout(
    sport: str,
    team1: str,
    team2: str,
    wager: float,
    payout: float,
    cashout: float,
    legs: list,
    game_total_line: Optional[float] = None,
    spread_line: Optional[float] = None,
    spread_team: Optional[str] = None,
    bankroll: Optional[float] = None,
):
    print(f"\n{'='*60}")
    print(f"LIVE DECISION ENGINE — {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    # 1. Pull live score
    print(f"\n📡 Fetching live score ({sport.upper()}: {team1} vs {team2})...")
    game = fetch_live_score(sport, team1, team2)

    if not game:
        print("  ❌ Could not fetch live score. Cannot make informed recommendation.")
        print("  → DO NOT give cash out advice without live score data.")
        return

    print(f"\n🏀 LIVE: {game['event_name']}")
    print(f"   Status: {game['status']}")
    print(f"   Score: {game['scores']}")
    print(f"   Total: {game['total']}")
    if game["linescores"]:
        for team, ls in game["linescores"].items():
            print(f"   {team} by period: {ls}")

    if game["completed"]:
        print("\n  ✅ Game is FINAL. Bet result should be settled.")
        return

    # 2. Calculate time and pace
    sport_full_time = 40.0 if sport == "ncaab" else 48.0
    time_remaining = parse_time_remaining(game["status"], sport)
    time_elapsed = sport_full_time - time_remaining

    if time_elapsed > 0:
        pace = game["total"] / time_elapsed
        projected_final = project_total(game["total"], time_elapsed, time_remaining)
    else:
        pace = 0
        projected_final = game["total"]

    print(f"\n📊 PACE ANALYSIS:")
    print(f"   Time elapsed: {time_elapsed:.1f} min | Remaining: {time_remaining:.1f} min")
    print(f"   Current pace: {pace:.2f} pts/min")
    print(f"   Projected final total: {projected_final:.0f}")

    # 3. Evaluate each leg
    print(f"\n🎯 LEG ANALYSIS (market line: {game_total_line}):")
    under_prob = None
    ml_prob = None

    if game_total_line:
        needed_h2 = game_total_line - game["total"]
        print(f"   Total needed to hit Under {game_total_line}: ≤{needed_h2:.1f} more pts in {time_remaining:.1f} min")
        print(f"   At current pace, projects: {projected_final:.0f} ({'OVER' if projected_final > game_total_line else 'UNDER'})")
        margin_from_line = game_total_line - projected_final
        # P(Under) using normal distribution
        sigma = 14 if sport == "ncaab" else 18
        under_prob = stats.norm.cdf(game_total_line, loc=projected_final, scale=sigma * (time_remaining / sport_full_time) ** 0.5)
        print(f"   P(Under {game_total_line}) = {under_prob:.1%}  [{margin_from_line:+.1f} pt buffer]")

    if spread_team and spread_line is not None:
        team_scores = {k.upper(): v for k, v in game["scores"].items()}
        team_score = team_scores.get(spread_team.upper(), 0)
        opp_score = sum(v for k, v in team_scores.items() if k.upper() != spread_team.upper())
        current_margin = team_score - opp_score
        if spread_line < 0:  # favorite
            covering = current_margin > abs(spread_line)
            swing_needed = max(0, abs(spread_line) - current_margin + 0.5)
            print(f"\n   {spread_team} {spread_line:+.1f} (FAVORITE): margin {current_margin:+d} | covering={covering} | swing_needed={swing_needed:.0f}")
            ml_prob = stats.norm.cdf(0, loc=-(current_margin), scale=8 * (time_remaining / sport_full_time) ** 0.5)
            ml_prob = 1 - ml_prob if current_margin > 0 else ml_prob
        else:  # underdog
            covering = current_margin > -spread_line
            swing_needed = max(0, -spread_line - current_margin)
            print(f"\n   {spread_team} {spread_line:+.1f} (UNDERDOG): margin {current_margin:+d} | covering={covering} | swing_needed={swing_needed:.0f}")

    # 4. Combined probability & EV
    probs = [p for p in [under_prob, ml_prob] if p is not None]
    combined_prob = 1.0
    for p in probs:
        combined_prob *= p
    if not probs:
        combined_prob = cashout / payout  # fallback: use DK's implied

    ev_holding = ev_hold(combined_prob, payout, wager)
    ev_cashout = cashout - wager  # net of cashout
    ev_hold_net = ev_holding - wager  # net EV of hold position

    print(f"\n💰 EV ANALYSIS:")
    print(f"   Combined probability: {combined_prob:.1%}")
    print(f"   DK implied probability: {cashout/payout:.1%}")
    print(f"   EV of HOLDING: ${ev_holding:.0f} (net: {ev_hold_net:+.0f})")
    print(f"   EV of CASHING: ${cashout:.0f} (net: {ev_cashout:+.0f})")
    print(f"   Edge to hold: ${ev_holding - cashout:+.0f}")

    # 5. Recommendation
    print(f"\n{'='*60}")
    edge = ev_holding - cashout
    if edge > 100:
        rec = f"✅ HOLD — ${edge:.0f} EV advantage. Both legs tracking."
    elif edge > 0:
        rec = f"⚖️  MARGINAL HOLD — only ${edge:.0f} EV edge. Your call."
    else:
        rec = f"💸 CASH OUT — DK pricing is fair or better. Take the guaranteed ${cashout:.0f}."

    print(f"RECOMMENDATION: {rec}")
    print(f"{'='*60}\n")

    return {
        "live_score": game["scores"],
        "total": game["total"],
        "projected_final": projected_final,
        "combined_prob": combined_prob,
        "ev_hold": ev_holding,
        "ev_cash": cashout,
        "edge": edge,
        "recommendation": rec,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live cash out decision engine")
    p.add_argument("--sport", default="nba", choices=["nba", "ncaab"])
    p.add_argument("--team1", required=True)
    p.add_argument("--team2", required=True)
    p.add_argument("--wager", type=float, required=True)
    p.add_argument("--payout", type=float, required=True)
    p.add_argument("--cashout", type=float, required=True)
    p.add_argument("--total-line", type=float, dest="total_line")
    p.add_argument("--spread-line", type=float, dest="spread_line")
    p.add_argument("--spread-team", dest="spread_team")
    p.add_argument("--bankroll", type=float)
    args = p.parse_args()

    analyze_cashout(
        sport=args.sport,
        team1=args.team1,
        team2=args.team2,
        wager=args.wager,
        payout=args.payout,
        cashout=args.cashout,
        legs=[],
        game_total_line=args.total_line,
        spread_line=args.spread_line,
        spread_team=args.spread_team,
        bankroll=args.bankroll,
    )

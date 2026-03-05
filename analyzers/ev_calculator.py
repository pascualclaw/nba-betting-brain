"""
Expected Value & Kelly Criterion Calculator

The most important module we were missing. Every bet recommendation
MUST pass through an EV gate before being made.

Philosophy:
- Only bet when EV >= 3% (0.03)
- Size bets using fractional Kelly (25% of full Kelly)
- Cap single bets at 5% of bankroll
- Track CLV after the fact to validate edge

Key insight from research: An 18-month quant study showed Kelly sizing
improved returns by +14% over flat sizing on the same model.

Usage:
    from analyzers.ev_calculator import EVCalculator

    calc = EVCalculator(bankroll=1000)
    result = calc.evaluate_bet(
        market_odds=-110,       # DraftKings American odds
        model_probability=0.55, # our model's win probability
        bet_type="spread",
        description="LAC -5.5 vs IND"
    )
    print(result)
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple

log = logging.getLogger(__name__)

# Thresholds (from literature)
MIN_EV_THRESHOLD = 0.03        # 3% minimum edge to bet
KELLY_FRACTION = 0.25          # Quarter Kelly (safer than full Kelly)
MAX_BET_PCT = 0.05             # Max 5% of bankroll per bet
MAX_BET_ABS = 500              # Hard cap in dollars
MIN_BET_ABS = 10               # Minimum meaningful bet


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal >= 2.0:
        return (decimal - 1) * 100
    else:
        return -100 / (decimal - 1)


def implied_probability(american_odds: float) -> float:
    """
    Convert American odds to implied probability (without vig).
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(home_odds: float, away_odds: float) -> Tuple[float, float]:
    """
    Remove the vig from a two-sided market and return true probabilities.
    """
    p_home_raw = implied_probability(home_odds)
    p_away_raw = implied_probability(away_odds)
    total = p_home_raw + p_away_raw
    return p_home_raw / total, p_away_raw / total


def calculate_ev(
    model_probability: float,
    market_odds_american: float,
) -> float:
    """
    Calculate Expected Value of a bet.

    EV = (p × b) - (1 - p)
    where b = decimal odds - 1

    Returns: EV per $1 wagered (positive = +EV bet)
    """
    decimal_odds = american_to_decimal(market_odds_american)
    b = decimal_odds - 1
    p = model_probability
    q = 1 - p
    return (p * b) - q


def calculate_kelly(
    model_probability: float,
    market_odds_american: float,
    fraction: float = KELLY_FRACTION,
) -> float:
    """
    Calculate fractional Kelly bet size as fraction of bankroll.

    f* = (bp - q) / b   [Full Kelly]
    f_bet = fraction × f*  [Fractional Kelly]

    Returns: fraction of bankroll to bet (e.g., 0.02 = 2%)
    """
    decimal_odds = american_to_decimal(market_odds_american)
    b = decimal_odds - 1
    p = model_probability
    q = 1 - p

    if b <= 0:
        return 0.0

    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return 0.0  # No edge — don't bet

    return min(full_kelly * fraction, MAX_BET_PCT)


def model_probability_from_spread(
    model_projected_spread: float,
    market_spread: float,
    sigma: float = 12.0,
) -> float:
    """
    Convert our model's spread projection into a win probability.

    Uses Normal distribution: margin ~ N(projected_spread, sigma)
    P(cover) = P(actual_margin > market_spread) for favorite,
               P(actual_margin < -market_spread) for underdog

    Args:
        model_projected_spread: our model's projected home margin (positive = home wins)
        market_spread: DK spread for home team (negative = home is favorite)
        sigma: historical standard deviation of margin residuals (~12 pts for NBA)
    """
    from scipy.stats import norm
    # Home covers if actual margin > -market_spread (since market_spread is negative for favorites)
    # e.g., market_spread = -5.5 means home is -5.5 favorite → home covers if wins by 6+
    required_margin = -market_spread  # positive = home needs to win by this
    # P(home covers) = P(margin > required_margin)
    return 1 - norm.cdf(required_margin, loc=model_projected_spread, scale=sigma)


def model_probability_from_total(
    model_projected_total: float,
    market_total: float,
    direction: str = "over",
    sigma: float = 18.0,
) -> float:
    """
    Convert our model's total projection into an over/under probability.

    sigma ~ 18-20 pts for NBA game totals (wider than spread)
    """
    from scipy.stats import norm
    if direction.lower() == "over":
        return 1 - norm.cdf(market_total, loc=model_projected_total, scale=sigma)
    else:
        return norm.cdf(market_total, loc=model_projected_total, scale=sigma)


class EVCalculator:
    """
    Full EV + Kelly calculator for NBA bets.
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        min_ev: float = MIN_EV_THRESHOLD,
        kelly_fraction: float = KELLY_FRACTION,
    ):
        self.bankroll = bankroll
        self.min_ev = min_ev
        self.kelly_fraction = kelly_fraction

    def evaluate_spread(
        self,
        home: str,
        away: str,
        market_spread: float,          # DK spread for home team (e.g., -5.5)
        model_projected_spread: float, # our model's projected home margin
        home_odds: float = -110,       # DK odds on home spread
        away_odds: float = -110,       # DK odds on away spread
        sigma: float = 12.0,
    ) -> Dict[str, Any]:
        """
        Evaluate a spread bet. Returns full EV analysis.
        """
        try:
            home_prob = model_probability_from_spread(
                model_projected_spread, market_spread, sigma
            )
            away_prob = 1 - home_prob

            home_ev = calculate_ev(home_prob, home_odds)
            away_ev = calculate_ev(away_prob, away_odds)

            # Determine which side (if any) has edge
            if home_ev >= self.min_ev and home_ev >= away_ev:
                bet_side = "home"
                bet_ev = home_ev
                bet_odds = home_odds
                bet_prob = home_prob
                bet_team = home
            elif away_ev >= self.min_ev:
                bet_side = "away"
                bet_ev = away_ev
                bet_odds = away_odds
                bet_prob = away_prob
                bet_team = away
            else:
                # No edge
                return {
                    "recommendation": "PASS",
                    "reason": f"Insufficient edge. Home EV: {home_ev:.1%}, Away EV: {away_ev:.1%}. Threshold: {self.min_ev:.1%}",
                    "home_ev": round(home_ev, 4),
                    "away_ev": round(away_ev, 4),
                    "home_prob": round(home_prob, 3),
                    "away_prob": round(away_prob, 3),
                    "market_spread": market_spread,
                    "model_spread": model_projected_spread,
                    "spread_diff": round(model_projected_spread - market_spread, 1),
                }

            kelly_pct = calculate_kelly(bet_prob, bet_odds, self.kelly_fraction)
            bet_size = max(MIN_BET_ABS, min(self.bankroll * kelly_pct, MAX_BET_ABS))

            return {
                "recommendation": "BET",
                "bet_side": bet_side,
                "bet_team": bet_team,
                "bet_line": market_spread if bet_side == "home" else -market_spread,
                "bet_odds": bet_odds,
                "ev": round(bet_ev, 4),
                "ev_pct": f"{bet_ev:.1%}",
                "model_probability": round(bet_prob, 3),
                "kelly_pct": round(kelly_pct, 4),
                "suggested_bet": round(bet_size, 0),
                "home_ev": round(home_ev, 4),
                "away_ev": round(away_ev, 4),
                "model_spread": model_projected_spread,
                "market_spread": market_spread,
                "spread_diff": round(model_projected_spread - market_spread, 1),
                "home": home,
                "away": away,
            }
        except ImportError:
            log.warning("scipy not available — using simplified probability estimate")
            return self._evaluate_spread_simple(home, away, market_spread, model_projected_spread, home_odds, away_odds)

    def _evaluate_spread_simple(self, home, away, market_spread, model_spread, home_odds, away_odds):
        """Simple spread evaluation without scipy."""
        diff = model_spread - (-market_spread)  # how much we beat the spread
        if diff > 3:
            home_prob = min(0.62, 0.50 + diff * 0.02)
        elif diff > 1:
            home_prob = 0.54
        else:
            home_prob = 0.50 + diff * 0.02

        home_ev = calculate_ev(home_prob, home_odds)
        away_ev = calculate_ev(1 - home_prob, away_odds)

        if home_ev >= self.min_ev:
            kelly_pct = calculate_kelly(home_prob, home_odds, self.kelly_fraction)
            bet_size = max(MIN_BET_ABS, min(self.bankroll * kelly_pct, MAX_BET_ABS))
            return {"recommendation": "BET", "bet_team": home, "ev": round(home_ev, 4),
                    "ev_pct": f"{home_ev:.1%}", "suggested_bet": round(bet_size, 0),
                    "model_probability": round(home_prob, 3)}
        elif away_ev >= self.min_ev:
            kelly_pct = calculate_kelly(1 - home_prob, away_odds, self.kelly_fraction)
            bet_size = max(MIN_BET_ABS, min(self.bankroll * kelly_pct, MAX_BET_ABS))
            return {"recommendation": "BET", "bet_team": away, "ev": round(away_ev, 4),
                    "ev_pct": f"{away_ev:.1%}", "suggested_bet": round(bet_size, 0),
                    "model_probability": round(1 - home_prob, 3)}
        return {"recommendation": "PASS", "home_ev": round(home_ev, 4), "away_ev": round(away_ev, 4)}

    def evaluate_total(
        self,
        market_total: float,
        model_projected_total: float,
        over_odds: float = -110,
        under_odds: float = -110,
        sigma: float = 18.0,
    ) -> Dict[str, Any]:
        """
        Evaluate an over/under total bet.
        """
        try:
            over_prob = model_probability_from_total(
                model_projected_total, market_total, "over", sigma
            )
            under_prob = 1 - over_prob

            over_ev = calculate_ev(over_prob, over_odds)
            under_ev = calculate_ev(under_prob, under_odds)

            if over_ev >= self.min_ev and over_ev >= under_ev:
                bet_dir = "OVER"
                bet_ev = over_ev
                bet_odds = over_odds
                bet_prob = over_prob
            elif under_ev >= self.min_ev:
                bet_dir = "UNDER"
                bet_ev = under_ev
                bet_odds = under_odds
                bet_prob = under_prob
            else:
                return {
                    "recommendation": "PASS",
                    "reason": f"No edge. Over EV: {over_ev:.1%}, Under EV: {under_ev:.1%}",
                    "over_ev": round(over_ev, 4),
                    "under_ev": round(under_ev, 4),
                    "market_total": market_total,
                    "model_total": model_projected_total,
                    "total_diff": round(model_projected_total - market_total, 1),
                }

            kelly_pct = calculate_kelly(bet_prob, bet_odds, self.kelly_fraction)
            bet_size = max(MIN_BET_ABS, min(self.bankroll * kelly_pct, MAX_BET_ABS))

            return {
                "recommendation": "BET",
                "direction": bet_dir,
                "line": market_total,
                "bet_odds": bet_odds,
                "ev": round(bet_ev, 4),
                "ev_pct": f"{bet_ev:.1%}",
                "model_probability": round(bet_prob, 3),
                "kelly_pct": round(kelly_pct, 4),
                "suggested_bet": round(bet_size, 0),
                "model_total": model_projected_total,
                "market_total": market_total,
                "total_diff": round(model_projected_total - market_total, 1),
                "over_ev": round(over_ev, 4),
                "under_ev": round(under_ev, 4),
            }
        except ImportError:
            # Simplified without scipy
            diff = model_projected_total - market_total
            if abs(diff) < 2:
                return {"recommendation": "PASS", "reason": "Edge too small (<2 pts)"}
            direction = "OVER" if diff > 0 else "UNDER"
            prob = min(0.60, 0.50 + abs(diff) * 0.02)
            odds = over_odds if direction == "OVER" else under_odds
            ev = calculate_ev(prob, odds)
            if ev >= self.min_ev:
                kelly_pct = calculate_kelly(prob, odds, self.kelly_fraction)
                bet_size = max(MIN_BET_ABS, min(self.bankroll * kelly_pct, MAX_BET_ABS))
                return {"recommendation": "BET", "direction": direction, "ev": round(ev, 4),
                        "ev_pct": f"{ev:.1%}", "suggested_bet": round(bet_size, 0)}
            return {"recommendation": "PASS", "reason": f"EV below threshold: {ev:.1%}"}

    def format_recommendation(self, result: Dict[str, Any]) -> str:
        """Format EV analysis as readable output."""
        if result["recommendation"] == "PASS":
            return f"⛔ PASS — {result.get('reason', 'No edge')}"

        lines = [
            f"✅ BET RECOMMENDED",
            f"   Team/Direction: {result.get('bet_team', result.get('direction', '?'))}",
            f"   EV: {result.get('ev_pct', '?')} (threshold: {self.min_ev:.0%})",
            f"   Model probability: {result.get('model_probability', '?')}",
            f"   Kelly bet size: ${result.get('suggested_bet', '?')} (on ${self.bankroll:.0f} bankroll)",
            f"   Kelly %: {result.get('kelly_pct', 0):.1%} of bankroll",
        ]
        if "model_spread" in result:
            lines.append(f"   Model spread: {result['model_spread']:+.1f} | Market: {result['market_spread']:+.1f} | Edge: {result['spread_diff']:+.1f}")
        if "model_total" in result:
            lines.append(f"   Model total: {result['model_total']:.0f} | Market: {result['market_total']} | Edge: {result['total_diff']:+.1f}")
        return "\n".join(lines)


# ── Quick EV check function for use in briefings ───────────────────────────

def quick_ev_check(
    model_projection: float,
    market_line: float,
    market_odds: float = -110,
    bet_type: str = "total",
    bankroll: float = 500.0,
) -> Dict[str, Any]:
    """
    Quick EV check for use in game briefings.
    Returns simplified result with recommendation.
    """
    calc = EVCalculator(bankroll=bankroll)
    if bet_type == "total":
        diff = model_projection - market_line
        direction = "over" if diff > 0 else "under"
        odds = market_odds if direction == "over" else market_odds
        return calc.evaluate_total(market_line, model_projection, market_odds, market_odds)
    elif bet_type == "spread":
        return calc.evaluate_spread("HOME", "AWAY", market_line, model_projection,
                                    market_odds, market_odds)


if __name__ == "__main__":
    # Test cases
    calc = EVCalculator(bankroll=1000.0)

    print("=== EV CALCULATOR TEST CASES ===\n")

    # Case 1: Good Over bet (model projects 230, market 224.5)
    result = calc.evaluate_total(224.5, 230.0, -110, -110)
    print("Case 1: Model 230 vs Market 224.5")
    print(calc.format_recommendation(result))
    print()

    # Case 2: Good Under bet (model projects 210, market 224.5)
    result = calc.evaluate_total(224.5, 210.0, -110, -110)
    print("Case 2: Model 210 vs Market 224.5")
    print(calc.format_recommendation(result))
    print()

    # Case 3: No edge (model 225 vs market 224.5)
    result = calc.evaluate_total(224.5, 225.0, -110, -110)
    print("Case 3: Model 225 vs Market 224.5 (no edge)")
    print(calc.format_recommendation(result))
    print()

    # Case 4: American odds conversion check
    print("Odds conversion checks:")
    for odds in [-110, -150, +120, +200, -200]:
        decimal = american_to_decimal(odds)
        imp_prob = implied_probability(odds)
        print(f"  {odds:+d} → decimal {decimal:.3f} → implied prob {imp_prob:.3f}")

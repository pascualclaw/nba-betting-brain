"""
Closing Line Value (CLV) Tracker

The gold standard for validating betting edge.
If you consistently beat the closing line, you have real edge.
If you don't, wins are luck and losses will catch up.

CLV = how much better your bet was vs where the market closed.

Usage:
    from analyzers.clv_tracker import CLVTracker

    tracker = CLVTracker()
    tracker.log_opening(bet_id="BET001", market="total", our_odds=-110,
                        line=224.5, direction="under")
    # ... after game closes ...
    tracker.log_closing(bet_id="BET001", closing_odds=-130, closing_line=222.5)
    report = tracker.get_clv_report()
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "clv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CLV_LOG = DATA_DIR / "clv_log.json"


def _load_log() -> List[Dict[str, Any]]:
    if CLV_LOG.exists():
        try:
            return json.loads(CLV_LOG.read_text())
        except Exception:
            return []
    return []


def _save_log(entries: List[Dict[str, Any]]) -> None:
    CLV_LOG.write_text(json.dumps(entries, indent=2))


def calculate_clv(
    our_odds_american: float,
    closing_odds_american: float,
) -> float:
    """
    Calculate CLV in percentage terms.
    Positive = we got a better price than the market closed at.
    Negative = market moved against us.

    CLV in percentage = (our_prob - closing_prob) / closing_prob
    where prob = implied probability (with vig)
    """
    def implied_prob(odds):
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    our_prob = implied_prob(our_odds_american)
    closing_prob = implied_prob(closing_odds_american)

    # CLV: positive means closing odds are WORSE for bettors (we got good number)
    return closing_prob - our_prob


def calculate_clv_points(
    our_line: float,
    closing_line: float,
    bet_side: str,  # "over", "under", "home", "away"
) -> float:
    """
    Calculate CLV in points (spread/total movement).
    Positive = line moved in our favor after we bet.
    """
    if bet_side in ("over", "home"):
        return our_line - closing_line  # we got a higher number = better for over
    else:
        return closing_line - our_line  # we got a lower number = better for under


class CLVTracker:
    """
    Tracks closing line value for all bets to validate long-term edge.
    """

    def __init__(self):
        self.entries = _load_log()

    def log_bet(
        self,
        bet_id: str,
        game: str,
        market: str,           # "spread", "total", "ml"
        direction: str,        # "home", "away", "over", "under"
        our_line: float,       # line when we bet
        our_odds: float,       # odds when we bet (American)
        game_date: Optional[str] = None,
        ev_at_bet: Optional[float] = None,
        model_projection: Optional[float] = None,
        closing_line: Optional[float] = None,    # fill in after game
        closing_odds: Optional[float] = None,    # fill in after game
        result: Optional[str] = None,            # "WIN" / "LOSS" / "PUSH"
        pnl: Optional[float] = None,             # actual P&L
    ) -> Dict[str, Any]:
        """
        Log a bet with opening info. Update later with closing line + result.
        """
        entry = {
            "bet_id": bet_id,
            "game": game,
            "game_date": game_date or str(date.today()),
            "market": market,
            "direction": direction,
            "our_line": our_line,
            "our_odds": our_odds,
            "ev_at_bet": ev_at_bet,
            "model_projection": model_projection,
            "closing_line": closing_line,
            "closing_odds": closing_odds,
            "clv_odds": None,
            "clv_points": None,
            "result": result,
            "pnl": pnl,
            "logged_at": datetime.now().isoformat(),
            "status": "open" if not result else "closed",
        }

        # If closing info already provided, compute CLV now
        if closing_odds is not None:
            entry["clv_odds"] = round(calculate_clv(our_odds, closing_odds), 4)
        if closing_line is not None:
            entry["clv_points"] = round(calculate_clv_points(our_line, closing_line, direction), 2)

        # Update if exists, append if new
        existing = next((e for e in self.entries if e["bet_id"] == bet_id), None)
        if existing:
            existing.update({k: v for k, v in entry.items() if v is not None})
        else:
            self.entries.append(entry)

        _save_log(self.entries)
        return entry

    def update_closing(
        self,
        bet_id: str,
        closing_line: float,
        closing_odds: float,
        result: Optional[str] = None,
        pnl: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a bet with closing line and result."""
        entry = next((e for e in self.entries if e["bet_id"] == bet_id), None)
        if not entry:
            log.warning(f"Bet {bet_id} not found in CLV log")
            return None

        entry["closing_line"] = closing_line
        entry["closing_odds"] = closing_odds
        entry["clv_odds"] = round(calculate_clv(entry["our_odds"], closing_odds), 4)
        entry["clv_points"] = round(calculate_clv_points(entry["our_line"], closing_line, entry["direction"]), 2)

        if result:
            entry["result"] = result
            entry["status"] = "closed"
        if pnl is not None:
            entry["pnl"] = pnl

        _save_log(self.entries)
        return entry

    def get_clv_report(
        self,
        market_filter: Optional[str] = None,
        n_recent: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate CLV performance report.
        The key metric: are we consistently beating the closing line?
        """
        entries = self.entries
        if market_filter:
            entries = [e for e in entries if e.get("market") == market_filter]
        if n_recent:
            entries = entries[-n_recent:]

        closed = [e for e in entries if e.get("status") == "closed"]
        with_clv = [e for e in closed if e.get("clv_odds") is not None]
        winners = [e for e in closed if e.get("result") == "WIN"]

        if not closed:
            return {"status": "no_closed_bets", "total_bets": len(entries)}

        avg_clv_odds = sum(e["clv_odds"] for e in with_clv) / len(with_clv) if with_clv else 0
        avg_clv_pts = sum(e["clv_points"] for e in with_clv if e.get("clv_points") is not None)
        avg_clv_pts /= len([e for e in with_clv if e.get("clv_points") is not None]) if with_clv else 1

        win_rate = len(winners) / len(closed) if closed else 0
        total_pnl = sum(e.get("pnl", 0) or 0 for e in closed)

        # CLV interpretation
        if avg_clv_odds > 0.02:
            clv_grade = "EXCELLENT — strong evidence of real edge"
        elif avg_clv_odds > 0:
            clv_grade = "POSITIVE — good but need more sample size"
        elif avg_clv_odds > -0.01:
            clv_grade = "NEUTRAL — no clear edge detected"
        else:
            clv_grade = "NEGATIVE — model is behind the market"

        return {
            "total_bets": len(entries),
            "closed_bets": len(closed),
            "open_bets": len(entries) - len(closed),
            "win_rate": round(win_rate, 3),
            "total_pnl": round(total_pnl, 2),
            "avg_clv_odds": round(avg_clv_odds, 4),
            "avg_clv_points": round(avg_clv_pts, 2),
            "clv_grade": clv_grade,
            "bets_with_positive_clv": len([e for e in with_clv if e.get("clv_odds", 0) > 0]),
            "bets_with_negative_clv": len([e for e in with_clv if e.get("clv_odds", 0) < 0]),
            "positive_clv_rate": round(
                len([e for e in with_clv if e.get("clv_odds", 0) > 0]) / len(with_clv), 3
            ) if with_clv else 0,
        }

    def print_report(self, **kwargs) -> None:
        """Print formatted CLV report."""
        r = self.get_clv_report(**kwargs)
        print("\n=== CLV PERFORMANCE REPORT ===")
        print(f"Total bets: {r['total_bets']} | Closed: {r['closed_bets']} | Open: {r['open_bets']}")
        if r['closed_bets'] > 0:
            print(f"Win rate: {r['win_rate']:.1%} | Total P&L: ${r['total_pnl']:+.2f}")
            print(f"Avg CLV (odds): {r['avg_clv_odds']:+.3f} | Avg CLV (pts): {r['avg_clv_points']:+.2f}")
            print(f"CLV Grade: {r['clv_grade']}")
            print(f"Positive CLV rate: {r['positive_clv_rate']:.1%} ({r['bets_with_positive_clv']}/{r['closed_bets']})")
        print()


if __name__ == "__main__":
    tracker = CLVTracker()

    # Log today's bets from our data
    # PHI SGP (WON)
    tracker.log_bet(
        bet_id="SGP_20260304_UTA_PHI",
        game="UTA@PHI",
        market="total",
        direction="under",
        our_line=237.5,
        our_odds=-110,
        game_date="2026-03-04",
        ev_at_bet=None,  # didn't compute EV before
        result="WIN",
        pnl=132.0,
        closing_line=237.5,  # placeholder — need real closing line
        closing_odds=-110,
    )

    tracker.print_report()

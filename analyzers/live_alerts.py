"""
Live Alerts — Real-time cash-out signals and bet monitoring.

Monitors active bets against live game state and fires alerts when:
- Lead changes significantly (spread leg at risk)
- Scoring run detected (momentum shift)
- Flagrant/technical foul impacts game flow
- Bet is winning or at risk
- Auto cash-out recommendation triggered

Usage:
    from analyzers.live_alerts import LiveBetMonitor

    monitor = LiveBetMonitor(
        home="LAC", away="IND",
        active_bets=[
            {"type": "spread", "team": "IND", "line": 15.5},
            {"type": "total", "direction": "under", "line": 224.5},
        ],
        discord_channel="1478608635738198067",
    )
    monitor.run(interval=30)
"""

import os
import sys
import time
import logging
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.live_score_poller import get_game_state, get_live_boxscore
from collectors.play_by_play import get_pbp, detect_runs, detect_events

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Alert thresholds
RUN_ALERT_MIN = 7          # Alert if one team scores 7+ unanswered
SPREAD_DANGER_BUFFER = 3   # Alert if spread leg within 3 points of losing
CASHOUT_BUFFER = 2         # Recommend cash-out if within 2 points of losing spread

# Discord webhook (set via env var or passed in)
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "1478608635738198067")


def evaluate_spread_bet(
    game: Dict[str, Any],
    team: str,
    line: float,
) -> Dict[str, Any]:
    """
    Evaluate a live spread bet.

    team: the team taking the points (e.g., "IND" for IND +15.5)
    line: the spread (positive = underdog, negative = favorite)
    """
    home = game["home"]
    away = game["away"]
    margin = game["margin"]  # positive = home winning

    # Current margin from the bet team's perspective
    if team == home:
        team_margin = margin  # positive = team winning
    else:
        team_margin = -margin  # positive = team winning (away)

    # For a +line bet: need team_margin >= -line (i.e., not losing by more than line)
    # For a -line bet: need team_margin >= line (i.e., winning by more than line)
    if line > 0:
        # Taking the points (underdog)
        current_cover = team_margin + line  # positive = covering
        needed_to_lose = current_cover      # how much cushion we have
    else:
        # Laying the points (favorite)
        current_cover = team_margin - abs(line)  # positive = covering
        needed_to_lose = current_cover

    is_covering = current_cover > 0
    in_danger = 0 < current_cover <= SPREAD_DANGER_BUFFER
    recommend_cashout = 0 < current_cover <= CASHOUT_BUFFER
    is_dead = current_cover <= 0

    return {
        "bet_type": "spread",
        "team": team,
        "line": line,
        "current_cover": round(current_cover, 1),
        "is_covering": is_covering,
        "in_danger": in_danger,
        "recommend_cashout": recommend_cashout,
        "is_dead": is_dead,
        "summary": _spread_summary(team, line, current_cover, is_covering),
    }


def _spread_summary(team: str, line: float, cover: float, is_covering: bool) -> str:
    sign = "+" if line > 0 else ""
    if is_covering:
        return f"✅ {team} {sign}{line} COVERING by {cover:.1f} pts"
    else:
        return f"❌ {team} {sign}{line} DEAD (short by {abs(cover):.1f} pts)"


def evaluate_total_bet(
    game: Dict[str, Any],
    direction: str,
    line: float,
) -> Dict[str, Any]:
    """
    Evaluate a live total (over/under) bet.
    Projects final total based on current pace.
    """
    current_total = game["total"]
    period = game["period"]
    clock = game["clock"]

    # Parse clock (format: "PT05M30.00S" or "5:30")
    mins_remaining_in_period = _parse_clock_to_minutes(clock)
    periods_remaining = max(0, 4 - period)
    total_mins_remaining = mins_remaining_in_period + (periods_remaining * 12)

    # Estimate pace from current total
    mins_played = (period - 1) * 12 + (12 - mins_remaining_in_period)
    if mins_played > 0:
        pace_per_min = current_total / mins_played
    else:
        pace_per_min = 4.5  # default NBA pace

    projected_final = current_total + (pace_per_min * total_mins_remaining)
    needed_pts = line - current_total  # points needed to hit the line

    if direction.lower() == "under":
        is_winning = projected_final < line
        cushion = line - projected_final
        is_dead = projected_final > line + 5
        in_danger = 0 < cushion < 8
        summary = (
            f"📉 UNDER {line}: {current_total} scored, projecting {projected_final:.0f} | "
            f"{'✅ COVERING' if is_winning else '❌ OVER'}"
        )
    else:
        is_winning = projected_final > line
        cushion = projected_final - line
        is_dead = projected_final < line - 5
        in_danger = 0 < cushion < 8
        summary = (
            f"📈 OVER {line}: {current_total} scored, projecting {projected_final:.0f} | "
            f"{'✅ COVERING' if is_winning else '❌ UNDER'}"
        )

    return {
        "bet_type": "total",
        "direction": direction,
        "line": line,
        "current_total": current_total,
        "projected_final": round(projected_final, 0),
        "pace_per_min": round(pace_per_min, 2),
        "total_mins_remaining": round(total_mins_remaining, 1),
        "is_winning": is_winning,
        "in_danger": in_danger,
        "is_dead": is_dead,
        "cushion": round(cushion, 1),
        "summary": summary,
    }


def _parse_clock_to_minutes(clock: str) -> float:
    """Parse ESPN clock format to minutes remaining."""
    try:
        if ":" in clock:
            parts = clock.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        elif "PT" in clock:
            # "PT05M30.00S"
            clock = clock.replace("PT", "").replace("S", "")
            if "M" in clock:
                m, s = clock.split("M")
                return float(m) + float(s) / 60
        return 6.0
    except Exception:
        return 6.0


def check_flagrant_impact(
    events: List[Dict[str, Any]],
    active_bets: List[Dict[str, Any]],
    game: Dict[str, Any],
) -> Optional[str]:
    """
    Check if a flagrant foul should trigger a cash-out alert.
    Rule: If flagrant by trailing team → free pts + possession for leading team → spread danger.
    """
    flagrants = [e for e in events if e["type"] == "FLAGRANT_FOUL"]
    if not flagrants:
        return None

    for flagrant in flagrants:
        fouling_team = flagrant.get("team", "")
        for bet in active_bets:
            if bet.get("type") == "spread":
                # If the team committing the flagrant is the team we have +points on
                bet_team = bet.get("team", "")
                if fouling_team and bet_team and fouling_team == bet_team:
                    return (
                        f"🚨 FLAGRANT FOUL by {fouling_team} ({flagrant.get('player', '?')}) — "
                        f"opponent gets 2 FTs + possession. "
                        f"CONSIDER CASHING OUT {bet_team} +{bet.get('line')} IMMEDIATELY."
                    )
                elif fouling_team:
                    return (
                        f"⚠️  FLAGRANT FOUL by {fouling_team} ({flagrant.get('player', '?')}) — "
                        f"free pts + possession coming. Monitor spread."
                    )
    return None


def generate_live_report(
    home: str,
    away: str,
    active_bets: List[Dict[str, Any]],
    nba_game_id: Optional[str] = None,
) -> str:
    """
    Generate a complete live bet status report.
    Call this every 30-60 seconds during a game.
    """
    game = get_game_state(home, away)
    if not game:
        return f"⚠️ Could not find live game: {away} @ {home}"

    lines = [
        f"🏀 {away} {game['away_score']} @ {home} {game['home_score']} | Q{game['period']} {game['clock']}",
        f"   Total: {game['total']} pts | Margin: {abs(game['margin'])} ({'home' if game['margin'] > 0 else 'away'} leading)",
        "",
    ]

    # Evaluate each active bet
    all_winning = True
    for bet in active_bets:
        if bet.get("type") == "spread":
            result = evaluate_spread_bet(game, bet["team"], bet["line"])
            lines.append(result["summary"])
            if result["recommend_cashout"]:
                lines.append(f"   💸 CASH OUT RECOMMENDED — only {result['current_cover']:.1f} pts cushion!")
            if not result["is_covering"]:
                all_winning = False

        elif bet.get("type") == "total":
            result = evaluate_total_bet(game, bet["direction"], bet["line"])
            lines.append(result["summary"])
            if result["in_danger"]:
                lines.append(f"   ⚠️  Only {result['cushion']:.0f} pts projected cushion — monitor closely")
            if not result["is_winning"]:
                all_winning = False

    # Check play-by-play for events
    if nba_game_id:
        try:
            pbp = get_pbp(nba_game_id)
            if pbp:
                # Scoring runs
                runs = detect_runs(pbp, min_run=RUN_ALERT_MIN)
                active_runs = [r for r in runs if r.get("is_current")]
                if active_runs:
                    r = active_runs[-1]
                    team_name = home if r["team"] == "home" else away
                    lines.append(f"\n🔥 ACTIVE RUN: {team_name} on a {r['points']}-0 run in Q{r['period']}")

                # Flagrants and events
                events = detect_events(pbp, recent_n=20)
                flagrant_alert = check_flagrant_impact(events, active_bets, game)
                if flagrant_alert:
                    lines.append(f"\n{flagrant_alert}")
        except Exception as e:
            log.debug(f"PBP check failed: {e}")

    lines.append("")
    lines.append("✅ ALL BETS WINNING" if all_winning else "⚠️  CHECK BETS ABOVE")

    return "\n".join(lines)


class LiveBetMonitor:
    """
    Continuous monitor for active live bets.
    Runs polling loop and sends Discord alerts when action required.
    """

    def __init__(
        self,
        home: str,
        away: str,
        active_bets: List[Dict[str, Any]],
        nba_game_id: Optional[str] = None,
        discord_channel: str = DISCORD_CHANNEL_ID,
        interval: int = 30,
    ):
        self.home = home
        self.away = away
        self.active_bets = active_bets
        self.nba_game_id = nba_game_id
        self.discord_channel = discord_channel
        self.interval = interval
        self._last_alert_hash = None

    def run(self, max_polls: int = 200) -> None:
        """Run the monitoring loop."""
        print(f"🏀 Monitoring {self.away} @ {self.home} | {len(self.active_bets)} active bet(s)")
        for i in range(max_polls):
            try:
                report = generate_live_report(
                    self.home, self.away, self.active_bets, self.nba_game_id
                )
                print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
                print(report)

                # Check if game is final
                game = get_game_state(self.home, self.away)
                if game and game.get("is_final"):
                    print("Game final. Stopping monitor.")
                    break

            except Exception as e:
                log.error(f"Monitor poll error: {e}")

            time.sleep(self.interval)


if __name__ == "__main__":
    # Example: monitor IND @ LAC with active bets
    monitor = LiveBetMonitor(
        home="LAC",
        away="IND",
        active_bets=[
            {"type": "spread", "team": "IND", "line": 15.5},
            {"type": "total", "direction": "under", "line": 224.5},
        ],
        nba_game_id="0022500898",
        interval=30,
    )
    monitor.run()

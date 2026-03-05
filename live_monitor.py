"""
Live Game Monitor — Real-time bet tracking with Discord alerts.

Polls ESPN every 30 seconds, NBA play-by-play every 60 seconds.
Sends Discord alerts when:
  - Bet is in danger (within 3 pts of losing)
  - Cash-out recommended (within 2 pts)
  - Flagrant foul detected (immediate alert)
  - Active scoring run detected (7+ unanswered)
  - Game is final (result reported)

Usage:
    python live_monitor.py LAC IND --nba-id 0022500898 --bets spread:IND:+15.5 total:under:224.5
    python live_monitor.py MIL ATL --bets ml:MIL prop:Giannis:PRA:40

Bot sends alerts to Discord channel 1478608635738198067
"""

import os
import sys
import time
import json
import logging
import argparse
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from collectors.live_score_poller import get_game_state, get_live_boxscore
from collectors.play_by_play import get_pbp, detect_runs, detect_events
from analyzers.live_alerts import (
    evaluate_spread_bet, evaluate_total_bet,
    check_flagrant_impact, generate_live_report,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DISCORD_CHANNEL = os.environ.get("DISCORD_CHANNEL_ID", "1478608635738198067")
OPENCLAW_WEBHOOK = os.environ.get("OPENCLAW_WEBHOOK_URL", "")  # set if webhook available
POLL_INTERVAL = 30    # seconds between ESPN score checks
PBP_INTERVAL  = 60    # seconds between play-by-play checks

# Global state
_last_score_report = ""
_last_pbp_alert = ""
_alert_cooldown = {}  # type -> last_alert_time, to avoid spamming


def _send_discord_alert(message: str, urgent: bool = False):
    """Send alert to Discord. Uses OpenClaw webhook if available, else logs."""
    prefix = "🚨 " if urgent else "🏀 "
    full_msg = prefix + message
    print(f"\n[DISCORD ALERT] {full_msg}")

    # Try OpenClaw webhook
    if OPENCLAW_WEBHOOK:
        try:
            r = requests.post(OPENCLAW_WEBHOOK, json={
                "channel": DISCORD_CHANNEL,
                "message": full_msg,
            }, timeout=5)
            if r.status_code == 200:
                log.info("Discord alert sent via webhook")
                return
        except Exception as e:
            log.debug(f"Webhook send failed: {e}")

    # Fallback: log to file for pickup
    alert_log = Path(__file__).parent / "data" / "live_alerts_pending.json"
    pending = []
    if alert_log.exists():
        try:
            pending = json.loads(alert_log.read_text())
        except Exception:
            pass
    pending.append({
        "message": full_msg,
        "channel": DISCORD_CHANNEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "urgent": urgent,
    })
    alert_log.write_text(json.dumps(pending, indent=2))


def _cooldown_ok(alert_type: str, cooldown_sec: int = 120) -> bool:
    """Return True if we haven't sent this alert type in the last cooldown_sec seconds."""
    now = time.time()
    last = _alert_cooldown.get(alert_type, 0)
    if now - last > cooldown_sec:
        _alert_cooldown[alert_type] = now
        return True
    return False


def parse_bets(bet_args: List[str]) -> List[Dict[str, Any]]:
    """
    Parse bet arguments from command line.
    Format: type:params
      spread:TEAM:+15.5   → {"type": "spread", "team": "TEAM", "line": 15.5}
      total:under:224.5   → {"type": "total", "direction": "under", "line": 224.5}
      ml:TEAM             → {"type": "ml", "team": "TEAM"}
      prop:NAME:PRA:40    → {"type": "prop", "player": "NAME", "stat": "PRA", "line": 40}
    """
    bets = []
    for arg in bet_args:
        parts = arg.split(":")
        if not parts:
            continue
        bet_type = parts[0].lower()

        if bet_type == "spread" and len(parts) >= 3:
            bets.append({
                "type": "spread",
                "team": parts[1].upper(),
                "line": float(parts[2]),
            })
        elif bet_type == "total" and len(parts) >= 3:
            bets.append({
                "type": "total",
                "direction": parts[1].lower(),
                "line": float(parts[2]),
            })
        elif bet_type == "ml" and len(parts) >= 2:
            bets.append({
                "type": "ml",
                "team": parts[1].upper(),
            })
        elif bet_type == "prop" and len(parts) >= 4:
            bets.append({
                "type": "prop",
                "player": parts[1],
                "stat": parts[2].upper(),
                "line": float(parts[3]),
            })

    return bets


def monitor_loop(
    home: str,
    away: str,
    active_bets: List[Dict[str, Any]],
    nba_game_id: Optional[str] = None,
    interval: int = POLL_INTERVAL,
    pbp_interval: int = PBP_INTERVAL,
    max_polls: int = 300,
) -> None:
    """
    Main monitoring loop. Runs until game is final or max_polls reached.
    """
    global _last_score_report, _last_pbp_alert

    print(f"\n{'='*50}")
    print(f"🏀 LIVE MONITOR STARTED: {away} @ {home}")
    print(f"   Active bets: {len(active_bets)}")
    for b in active_bets:
        if b["type"] == "spread":
            print(f"   - {b['team']} {'+' if b['line'] > 0 else ''}{b['line']}")
        elif b["type"] == "total":
            print(f"   - {b['direction'].upper()} {b['line']}")
        elif b["type"] == "ml":
            print(f"   - {b['team']} ML")
    print(f"   Poll interval: {interval}s | PBP interval: {pbp_interval}s")
    print(f"{'='*50}\n")

    last_pbp_check = 0
    polls = 0

    while polls < max_polls:
        now = time.time()
        game = get_game_state(home, away)

        if not game:
            log.warning(f"No game data for {away}@{home} — retrying...")
            time.sleep(interval)
            polls += 1
            continue

        score_str = f"Q{game['period']} {game['clock']} | {away} {game['away_score']} @ {home} {game['home_score']}"
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {score_str}", end="", flush=True)

        # Evaluate each bet
        alerts = []
        for bet in active_bets:
            if bet["type"] == "spread":
                result = evaluate_spread_bet(game, bet["team"], bet["line"])

                if result["is_dead"] and _cooldown_ok(f"spread_dead_{bet['team']}", 300):
                    alerts.append((f"❌ {result['summary']}", True))

                elif result["recommend_cashout"] and _cooldown_ok(f"cashout_{bet['team']}", 180):
                    alerts.append((
                        f"💸 CASH OUT NOW — {result['summary']} — only {result['current_cover']:.1f} pts cushion!",
                        True
                    ))

                elif result["in_danger"] and _cooldown_ok(f"danger_{bet['team']}", 120):
                    alerts.append((f"⚠️  SPREAD DANGER: {result['summary']}", False))

            elif bet["type"] == "total":
                result = evaluate_total_bet(game, bet["direction"], bet["line"])
                if result["is_dead"] and _cooldown_ok("total_dead", 300):
                    alerts.append((f"❌ {result['summary']}", True))
                elif result["in_danger"] and _cooldown_ok("total_danger", 120):
                    alerts.append((f"⚠️  TOTAL DANGER: {result['summary']}", False))

        # Check PBP periodically
        if nba_game_id and (now - last_pbp_check > pbp_interval):
            try:
                pbp = get_pbp(nba_game_id)
                if pbp:
                    runs = detect_runs(pbp, min_run=7)
                    active_runs = [r for r in runs if r.get("is_current")]
                    if active_runs:
                        r = active_runs[-1]
                        team_name = home if r["team"] == "home" else away
                        run_key = f"run_{team_name}_{r['points']}"
                        if _cooldown_ok(run_key, 90):
                            alerts.append((
                                f"🔥 {team_name} on a {r['points']}-0 run in Q{r['period']} — spread at risk!",
                                r["points"] >= 10
                            ))

                    events = detect_events(pbp, recent_n=15)
                    flagrant_msg = check_flagrant_impact(events, active_bets, game)
                    if flagrant_msg and _cooldown_ok("flagrant", 60):
                        alerts.append((flagrant_msg, True))

                last_pbp_check = now
            except Exception as e:
                log.debug(f"PBP check error: {e}")

        # Send alerts
        for alert_msg, urgent in alerts:
            print(f"\n  → {alert_msg}")
            _send_discord_alert(
                f"{away} @ {home} Q{game['period']} {game['clock']} | {alert_msg}",
                urgent=urgent
            )

        # Game final
        if game.get("is_final"):
            margin = game["margin"]
            final_total = game["total"]
            print(f"\n\n{'='*50}")
            print(f"🏁 FINAL: {away} {game['away_score']} @ {home} {game['home_score']}")
            print(f"   Margin: {abs(margin)} ({'home' if margin > 0 else 'away'}) | Total: {final_total}")

            # Final bet results
            results = []
            for bet in active_bets:
                if bet["type"] == "spread":
                    r = evaluate_spread_bet(game, bet["team"], bet["line"])
                    results.append(r["summary"])
                elif bet["type"] == "total":
                    r = evaluate_total_bet(game, bet["direction"], bet["line"])
                    results.append(r["summary"])

            for res in results:
                print(f"   {res}")

            final_msg = (
                f"FINAL: {away} {game['away_score']} @ {home} {game['home_score']} | "
                + " | ".join(results)
            )
            _send_discord_alert(final_msg, urgent=False)
            print(f"{'='*50}\n")
            break

        polls += 1
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="NBA Live Bet Monitor")
    parser.add_argument("home", help="Home team abbreviation (e.g. LAC)")
    parser.add_argument("away", help="Away team abbreviation (e.g. IND)")
    parser.add_argument("--nba-id", help="NBA game ID (e.g. 0022500898)")
    parser.add_argument("--bets", nargs="+", help="Active bets (spread:TEAM:+15.5, total:under:224.5, ml:TEAM, prop:PLAYER:STAT:LINE)")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="Poll interval in seconds")
    parser.add_argument("--status", action="store_true", help="Just print current status and exit")
    args = parser.parse_args()

    bets = parse_bets(args.bets or [])

    if args.status:
        game = get_game_state(args.home, args.away)
        if game:
            print(f"{args.away} {game['away_score']} @ {args.home} {game['home_score']} | Q{game['period']} {game['clock']} | total={game['total']}")
            for bet in bets:
                if bet["type"] == "spread":
                    r = evaluate_spread_bet(game, bet["team"], bet["line"])
                    print(f"  {r['summary']}")
                elif bet["type"] == "total":
                    r = evaluate_total_bet(game, bet["direction"], bet["line"])
                    print(f"  {r['summary']}")
        else:
            print(f"Game not found: {args.away} @ {args.home}")
        return

    monitor_loop(
        home=args.home,
        away=args.away,
        active_bets=bets,
        nba_game_id=args.nba_id,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()

"""
collectors/opening_line_logger.py — Opening Line & Movement Tracker
====================================================================
Based on: "Line Movement as a Signal for Sharp Action" (betting market literature)

Key insight: Opening lines are set hours before confirmed lineups.
Sharp money moves lines in final 2-4 hours — BEFORE then is our edge window.

Functions:
- log_opening_line(game_id, market, line, odds, timestamp) — capture first available
- log_current_line(game_id, market, line, odds) — update to latest
- get_line_movement(game_id, market) — returns movement analysis
- detect_sharp_action(movement) — classify SHARP CONFIRM / SHARP FADE / NEUTRAL
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "nba_betting.db"

logger = logging.getLogger(__name__)

# Sharp action thresholds
SHARP_MOVEMENT_THRESHOLD = 1.5  # points — movement > this = "sharp action"
ODDS_SHARP_THRESHOLD     = 10   # American odds units

MOVEMENT_CLASSIFICATIONS = {
    "strong_confirm": "🟢 SHARP CONFIRM — smart money agrees, line moved our way",
    "confirm":        "✅ CONFIRM — line moved our direction",
    "neutral":        "⚪ NEUTRAL — minimal line movement",
    "fade":           "🔴 SHARP FADE — line moved AGAINST us (reassess before betting)",
    "strong_fade":    "❌ STRONG FADE — line significantly moved against us",
}


def ensure_line_tables(conn: sqlite3.Connection):
    """Create line tracking tables if not exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS line_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            market TEXT NOT NULL,
            snapshot_type TEXT NOT NULL,   -- 'opening' or 'current'
            line REAL,
            odds INTEGER,
            bookmaker TEXT DEFAULT 'draftkings',
            timestamp TEXT NOT NULL,
            UNIQUE(game_id, market, snapshot_type, bookmaker)
        );

        CREATE TABLE IF NOT EXISTS line_movement_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            market TEXT NOT NULL,
            opening_line REAL,
            current_line REAL,
            movement REAL,
            opening_odds INTEGER,
            current_odds INTEGER,
            odds_movement INTEGER,
            classification TEXT,
            our_side TEXT,        -- which side we're betting
            logged_at TEXT NOT NULL
        );
    """)
    conn.commit()


def log_opening_line(game_id: str, market: str, line: float, odds: int,
                      timestamp: str = None, bookmaker: str = "draftkings",
                      db_path: Path = None):
    """
    Capture the first-available opening line for a game/market.
    Only stores if no opening line exists yet for this game+market+bookmaker.
    """
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    ensure_line_tables(conn)

    ts = timestamp or datetime.now(timezone.utc).isoformat()

    try:
        conn.execute("""
            INSERT OR IGNORE INTO line_snapshots
            (game_id, market, snapshot_type, line, odds, bookmaker, timestamp)
            VALUES (?, ?, 'opening', ?, ?, ?, ?)
        """, (game_id, market, float(line) if line is not None else None,
              int(odds) if odds is not None else None, bookmaker, ts))
        conn.commit()
        logger.debug(f"Opening line logged: {game_id} {market} {line} ({odds})")
    except Exception as e:
        logger.warning(f"Line log failed: {e}")
    finally:
        conn.close()


def log_current_line(game_id: str, market: str, line: float, odds: int,
                      bookmaker: str = "draftkings", db_path: Path = None):
    """Update the current line snapshot (upsert)."""
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    ensure_line_tables(conn)
    ts = datetime.now(timezone.utc).isoformat()

    try:
        conn.execute("""
            INSERT OR REPLACE INTO line_snapshots
            (game_id, market, snapshot_type, line, odds, bookmaker, timestamp)
            VALUES (?, ?, 'current', ?, ?, ?, ?)
        """, (game_id, market, float(line) if line is not None else None,
              int(odds) if odds is not None else None, bookmaker, ts))
        conn.commit()
    except Exception as e:
        logger.warning(f"Current line update failed: {e}")
    finally:
        conn.close()


def get_line_movement(game_id: str, market: str,
                       bookmaker: str = "draftkings",
                       db_path: Path = None) -> Dict[str, Any]:
    """
    Get opening vs current line movement for a game/market.
    Returns movement analysis dict.
    """
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    ensure_line_tables(conn)

    opening = conn.execute("""
        SELECT line, odds, timestamp FROM line_snapshots
        WHERE game_id=? AND market=? AND snapshot_type='opening' AND bookmaker=?
    """, (game_id, market, bookmaker)).fetchone()

    current = conn.execute("""
        SELECT line, odds, timestamp FROM line_snapshots
        WHERE game_id=? AND market=? AND snapshot_type='current' AND bookmaker=?
    """, (game_id, market, bookmaker)).fetchone()

    conn.close()

    result = {
        "game_id":      game_id,
        "market":       market,
        "opening_line": None,
        "current_line": None,
        "movement":     None,
        "opening_odds": None,
        "current_odds": None,
        "odds_movement":None,
        "status":       "no_data",
        "classification": "neutral",
        "label":        MOVEMENT_CLASSIFICATIONS["neutral"],
    }

    if opening:
        result["opening_line"] = opening[0]
        result["opening_odds"] = opening[1]
        result["opening_ts"]   = opening[2]

    if current:
        result["current_line"] = current[0]
        result["current_odds"] = current[1]
        result["current_ts"]   = current[2]

    if opening and current and opening[0] is not None and current[0] is not None:
        movement = current[0] - opening[0]
        odds_move = (current[1] or 0) - (opening[1] or 0)
        result["movement"]      = movement
        result["odds_movement"] = odds_move
        result["status"]        = "tracked"

    return result


def classify_movement(movement: Optional[float], our_side: str = "home",
                       market: str = "spread") -> Dict[str, str]:
    """
    Classify line movement relative to which side we're betting.

    our_side: "home" or "away" for spread/ML; "over" or "under" for totals

    For spread bets:
    - Betting HOME (home favored): our_side = "home"
      - Line moved from -3.5 to -5.5 (more negative = home MORE favored) → BAD for us
        Wait: if we bet home +4.5 and line moves to +3.5, home dog gets fewer points → FADE
      - Line moved from -3.5 to -1.5 (less negative = home LESS favored) → GOOD for home dog

    For totals:
    - Betting OVER: if line moved up (e.g., 225 → 228) → SHARP FADE (market agrees total is higher)
    - Betting UNDER: if line moved up → SHARP CONFIRM (we get better number)

    Simplified: positive movement = line went up. Classify based on direction + our_side.
    """
    if movement is None:
        return {"classification": "neutral", "label": MOVEMENT_CLASSIFICATIONS["neutral"]}

    abs_move = abs(movement)

    # For spread/ML: line moving toward us = confirm, away from us = fade
    if market in ("spread", "ml"):
        if our_side == "home":
            # Positive movement = line moved to favor home more = we got worse price (fade)
            direction = -movement  # flip: more favorable home spread = worse for home bettor
        else:
            direction = movement

    elif market == "total_over":
        # Betting over: line going UP is bad (need more pts to cover)
        direction = -movement
    elif market == "total_under":
        # Betting under: line going UP is good (more points to go under from)
        direction = movement
    else:
        direction = movement

    if direction > SHARP_MOVEMENT_THRESHOLD * 1.5:
        key = "strong_confirm"
    elif direction > SHARP_MOVEMENT_THRESHOLD:
        key = "confirm"
    elif direction < -SHARP_MOVEMENT_THRESHOLD * 1.5:
        key = "strong_fade"
    elif direction < -SHARP_MOVEMENT_THRESHOLD:
        key = "fade"
    else:
        key = "neutral"

    return {
        "classification": key,
        "label": MOVEMENT_CLASSIFICATIONS[key],
        "movement_pts": movement,
        "direction": "confirm" if direction > 0 else ("fade" if direction < 0 else "neutral"),
    }


def log_all_nba_opening_lines(api_key: str, game_date: str = None,
                                db_path: Path = None):
    """
    Pull current DK NBA lines and log them as opening lines.
    Should be called early (2-3 hrs before tip) to capture pre-sharp lines.
    """
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "bookmakers": "draftkings",
                "oddsFormat": "american",
            },
            timeout=15
        )
        games = r.json()
        remaining = r.headers.get("x-requests-remaining", "?")
        logger.info(f"Opening lines: {len(games)} NBA games | API remaining: {remaining}")
    except Exception as e:
        logger.error(f"Opening line fetch failed: {e}")
        return 0

    logged = 0
    for game in games:
        game_id = game.get("id", "")
        home    = game.get("home_team", "")
        away    = game.get("away_team", "")
        dk = next((b for b in game.get("bookmakers", []) if b.get("key") == "draftkings"), {})

        for market in dk.get("markets", []):
            mkt_key = market.get("key", "")
            for outcome in market.get("outcomes", []):
                name  = outcome.get("name", "")
                point = outcome.get("point")
                price = outcome.get("price")

                if point is not None:
                    market_label = f"{mkt_key}_{name.lower().replace(' ', '_')}"
                    log_opening_line(game_id, market_label, point, price,
                                     db_path=db_path)
                    log_current_line(game_id, market_label, point, price,
                                     db_path=db_path)
                    logged += 1

    logger.info(f"Opening lines logged: {logged} market entries")
    return logged


def format_movement_for_output(movement_data: Dict) -> str:
    """Format line movement for Discord/console output."""
    if movement_data.get("status") == "no_data":
        return "⚪ Line movement: Not tracked (first run)"

    opening = movement_data.get("opening_line")
    current = movement_data.get("current_line")
    movement = movement_data.get("movement")
    label = movement_data.get("label", "")

    if opening is None or current is None:
        return "⚪ Line movement: Insufficient data"

    move_str = f"{movement:+.1f} pts" if movement else "±0"
    return f"📊 Line: {opening:+g} → {current:+g} ({move_str}) | {label}"


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        creds = Path.home() / ".openclaw/credentials/jarvis-github.env"
        if creds.exists():
            for line in creds.read_text().splitlines():
                if line.startswith("ODDS_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()

    if api_key:
        n = log_all_nba_opening_lines(api_key)
        print(f"Logged {n} opening line entries")
    else:
        print("No API key — set ODDS_API_KEY")

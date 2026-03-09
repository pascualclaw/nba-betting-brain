#!/usr/bin/env bash
# daily_retrain.sh — NBA Betting Brain daily pipeline
#
# Runs every night (or manually) to:
#   1. Activate the Python venv
#   2. Fetch yesterday's NBA game results from ESPN into the DB
#   3. Regenerate the training features CSV
#   4. Retrain the main total model (training/train.py)
#   5. Retrain the residual market model (training/train_residual.py)
#
# Logs everything to logs/retrain_YYYY-MM-DD.log
#
# Usage:
#   chmod +x scripts/daily_retrain.sh
#   ./scripts/daily_retrain.sh
#   # or via cron:
#   # 30 3 * * * /path/to/nba-betting-brain/scripts/daily_retrain.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_DIR/venv"
LOG_DIR="$REPO_DIR/logs"
DATE_TAG="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/retrain_${DATE_TAG}.log"
YESTERDAY="$(date -v-1d +%Y-%m-%d 2>/dev/null || date --date='yesterday' +%Y-%m-%d)"

# ── Setup ─────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

exec >> "$LOG_FILE" 2>&1

echo "========================================"
echo "  NBA Betting Brain — Daily Retrain"
echo "  Date: $DATE_TAG  (fetching: $YESTERDAY)"
echo "========================================"
echo "Start: $(date)"
echo ""

# ── Activate venv ─────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] venv not found at $VENV_DIR"
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"

echo "[INFO] Python: $(python3 --version)"
echo "[INFO] Repo  : $REPO_DIR"
echo ""

# ── Step 1: Fetch yesterday's results ────────────────────────────────────
echo "── Step 1: Fetching results for $YESTERDAY ──────────────────────────"
python3 - <<'PYEOF'
import sys, sqlite3, time, json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, '.')

try:
    from collectors.nba_api import NBAApiCollector
    collector = NBAApiCollector()
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"[INFO] Fetching NBA games for {yesterday} via nba_api...")
    collector.collect_daily_games(yesterday)
    print("[INFO] nba_api fetch complete")
except Exception as e:
    print(f"[WARN] nba_api collector failed: {e}")
    print("[INFO] Trying ESPN scoreboard fallback...")
    try:
        import requests
        from database.db import get_connection, upsert_game
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        date_compact = yesterday.replace('-', '')  # YYYYMMDD
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_compact}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        events = data.get('events', [])
        conn = get_connection()
        count = 0
        for ev in events:
            try:
                competitions = ev.get('competitions', [{}])
                comp = competitions[0]
                competitors = comp.get('competitors', [])
                if len(competitors) < 2:
                    continue
                home_comp = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                away_comp = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
                home_abbr = home_comp['team']['abbreviation']
                away_abbr = away_comp['team']['abbreviation']
                home_score = int(home_comp.get('score', 0) or 0)
                away_score = int(away_comp.get('score', 0) or 0)
                if home_score == 0 and away_score == 0:
                    continue
                game = {
                    'game_id': ev['id'],
                    'date': yesterday,
                    'season': '2024-25',
                    'home': home_abbr,
                    'away': away_abbr,
                    'home_score': home_score,
                    'away_score': away_score,
                    'total': home_score + away_score,
                    'winner': home_abbr if home_score > away_score else away_abbr,
                    'home_margin': home_score - away_score,
                }
                upsert_game(conn, game)
                count += 1
            except Exception as ge:
                print(f"  [WARN] Skipping game: {ge}")
        conn.commit()
        conn.close()
        print(f"[INFO] ESPN fallback: inserted/updated {count} games for {yesterday}")
    except Exception as e2:
        print(f"[WARN] ESPN fallback also failed: {e2}")
        print("[INFO] Continuing with existing data...")
PYEOF
echo ""

# ── Step 2: Regenerate training features CSV ─────────────────────────────
echo "── Step 2: Regenerating training features CSV ───────────────────────"
python3 training/historical_loader.py --seasons 3
echo ""

# ── Step 3: Retrain main total model ─────────────────────────────────────
echo "── Step 3: Retraining main total model ──────────────────────────────"
python3 training/train.py
echo ""

# ── Step 4: Retrain residual model ───────────────────────────────────────
echo "── Step 4: Retraining market residual model ─────────────────────────"
python3 training/train_residual.py
echo ""

# ── Done ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Retrain complete: $(date)"
echo "  Log: $LOG_FILE"
echo "========================================"

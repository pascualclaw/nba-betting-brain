"""
NCAAB Daily Picks CLI

Fetches today's college basketball games, runs model predictions,
and outputs picks with EV >= 3%.

Usage:
    python3 college/daily_picks.py [--date YYYY-MM-DD] [--discord]

Options:
    --date      Date to fetch games for (default: today)
    --discord   Format output with emoji for Discord

Requirements:
    - Run espn_loader.py to load game history into DB
    - Run train.py to generate models
"""

import sys
import os
import pickle
import logging
import requests
import argparse
from datetime import datetime, date
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

from college.config import (
    NCAAB_DB_PATH, NCAAB_MODELS_DIR, ESPN_SCOREBOARD,
    ODDS_API_KEY, ODDS_API_NCAAB, HOME_COURT_ADVANTAGE
)
from college.espn_loader import get_db_conn, init_db, get_all_games_df
from college.features import build_game_features, FEATURE_COLS
from analyzers.ev_calculator import EVCalculator, calculate_ev

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Sigma values for NCAAB (wider distributions than NBA)
NCAAB_SPREAD_SIGMA = 14.0   # pts std dev for spread
NCAAB_TOTAL_SIGMA = 20.0    # pts std dev for total


# ── Model Loading ──────────────────────────────────────────────────────────

def load_models():
    """Load trained total and spread models."""
    total_path = os.path.join(NCAAB_MODELS_DIR, 'total_latest.pkl')
    spread_path = os.path.join(NCAAB_MODELS_DIR, 'spread_latest.pkl')

    if not os.path.exists(total_path) or not os.path.exists(spread_path):
        raise FileNotFoundError(
            f"Models not found in {NCAAB_MODELS_DIR}. "
            "Run college/train.py first."
        )

    with open(total_path, 'rb') as f:
        total_model = pickle.load(f)
    with open(spread_path, 'rb') as f:
        spread_model = pickle.load(f)

    log.info("Loaded NCAAB total and spread models")
    return total_model, spread_model


# ── ESPN Scoreboard ────────────────────────────────────────────────────────

def fetch_todays_games(game_date: str) -> list:
    """
    Fetch today's NCAAB games from ESPN scoreboard.
    game_date: YYYY-MM-DD
    Returns list of game dicts with team IDs and names.
    """
    date_str = game_date.replace('-', '')  # YYYYMMDD
    params = {"dates": date_str, "limit": 100}

    try:
        resp = requests.get(ESPN_SCOREBOARD, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch scoreboard: {e}")
        return []

    games = []
    for event in data.get('events', []):
        try:
            competition = event['competitions'][0]
            competitors = competition.get('competitors', [])

            if len(competitors) < 2:
                continue

            home_comp = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_comp = next((c for c in competitors if c.get('homeAway') == 'away'), None)

            if not home_comp or not away_comp:
                continue

            neutral = competition.get('neutralSite', False)

            games.append({
                'game_id': str(event['id']),
                'home_team_id': str(home_comp['team']['id']),
                'away_team_id': str(away_comp['team']['id']),
                'home_team_name': home_comp['team'].get('displayName', home_comp['team'].get('name', '')),
                'away_team_name': away_comp['team'].get('displayName', away_comp['team'].get('name', '')),
                'home_abbr': home_comp['team'].get('abbreviation', ''),
                'away_abbr': away_comp['team'].get('abbreviation', ''),
                'neutral_site': 1 if neutral else 0,
                'date': game_date,
                'season': 2026,  # Current season
            })
        except Exception as e:
            log.debug(f"Error parsing event: {e}")
            continue

    log.info(f"Found {len(games)} games on {game_date}")
    return games


# ── Odds Fetching ──────────────────────────────────────────────────────────

def fetch_ncaab_odds() -> dict:
    """
    Fetch NCAAB odds from The Odds API.
    Returns dict keyed by team name: {spread, total, spread_odds, total_odds}
    """
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(ODDS_API_NCAAB, params=params, timeout=15)
        resp.raise_for_status()
        odds_data = resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch odds: {e}")
        return {}

    odds_map = {}

    for game in odds_data:
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        # Look for DraftKings first, then any bookmaker
        bookmakers = game.get('bookmakers', [])
        dk = next((b for b in bookmakers if b['key'] == 'draftkings'), None)
        bookmaker = dk or (bookmakers[0] if bookmakers else None)

        if not bookmaker:
            continue

        markets = {m['key']: m for m in bookmaker.get('markets', [])}
        game_key = f"{away_team}@{home_team}"

        spread_data = {'home_spread': None, 'home_spread_odds': -110, 'away_spread_odds': -110}
        total_data = {'total': None, 'over_odds': -110, 'under_odds': -110}

        # Parse spreads
        if 'spreads' in markets:
            for outcome in markets['spreads'].get('outcomes', []):
                if outcome['name'] == home_team:
                    spread_data['home_spread'] = outcome.get('point', None)
                    spread_data['home_spread_odds'] = outcome.get('price', -110)
                elif outcome['name'] == away_team:
                    spread_data['away_spread_odds'] = outcome.get('price', -110)

        # Parse totals
        if 'totals' in markets:
            for outcome in markets['totals'].get('outcomes', []):
                if outcome['name'] == 'Over':
                    total_data['total'] = outcome.get('point', None)
                    total_data['over_odds'] = outcome.get('price', -110)
                elif outcome['name'] == 'Under':
                    total_data['under_odds'] = outcome.get('price', -110)

        odds_map[game_key] = {
            'home_team': home_team,
            'away_team': away_team,
            **spread_data,
            **total_data,
        }

    log.info(f"Fetched odds for {len(odds_map)} NCAAB games")
    return odds_map


def match_game_odds(game: dict, odds_map: dict) -> dict:
    """
    Try to match a game with odds data using team name fuzzy matching.
    Returns odds dict or empty dict if no match.
    """
    home_name = game['home_team_name'].lower()
    away_name = game['away_team_name'].lower()

    for key, odds in odds_map.items():
        odds_home = odds['home_team'].lower()
        odds_away = odds['away_team'].lower()

        # Direct match or partial match
        if (home_name in odds_home or odds_home in home_name) and \
           (away_name in odds_away or odds_away in away_name):
            return odds

        # Abbreviation match
        home_abbr = game.get('home_abbr', '').lower()
        away_abbr = game.get('away_abbr', '').lower()
        if home_abbr and away_abbr:
            if (home_abbr in odds_home or odds_home in home_abbr) and \
               (away_abbr in odds_away or odds_away in away_abbr):
                return odds

    return {}


# ── Prediction Pipeline ────────────────────────────────────────────────────

def predict_game(
    game: dict,
    games_df: pd.DataFrame,
    total_model,
    spread_model,
) -> dict:
    """
    Build features and generate predictions for a single game.
    Returns prediction dict or None if insufficient history.
    """
    row = pd.Series({
        'game_id': game['game_id'],
        'date': game['date'],
        'home_team_id': str(game['home_team_id']),
        'away_team_id': str(game['away_team_id']),
        'home_score': 0,
        'away_score': 0,
        'neutral_site': game.get('neutral_site', 0),
        'season': game.get('season', 2026),
    })

    features = build_game_features(row, games_df)
    if features is None:
        return None

    X = np.array([[features[col] for col in FEATURE_COLS]])

    pred_total = float(total_model.predict(X)[0])
    pred_margin = float(spread_model.predict(X)[0])

    return {
        **features,
        'pred_total': round(pred_total, 1),
        'pred_margin': round(pred_margin, 1),
    }


# ── EV Calculation ─────────────────────────────────────────────────────────

def compute_picks(predictions: list, odds_map: dict) -> list:
    """
    Compute EV for each game and return picks with EV >= 3%.
    """
    calc = EVCalculator(bankroll=1000.0)
    picks = []

    for pred in predictions:
        game = pred['game']
        p = pred['prediction']

        odds = match_game_odds(game, odds_map)
        if not odds:
            log.debug(f"No odds for {game['away_team_name']} @ {game['home_team_name']}")
            continue

        home_name = game['home_team_name']
        away_name = game['away_team_name']

        result_total = None
        result_spread = None

        # Evaluate total
        if odds.get('total') is not None:
            result_total = calc.evaluate_total(
                market_total=float(odds['total']),
                model_projected_total=p['pred_total'],
                over_odds=float(odds.get('over_odds', -110)),
                under_odds=float(odds.get('under_odds', -110)),
                sigma=NCAAB_TOTAL_SIGMA,
            )

        # Evaluate spread
        if odds.get('home_spread') is not None:
            result_spread = calc.evaluate_spread(
                home=home_name,
                away=away_name,
                market_spread=float(odds['home_spread']),
                model_projected_spread=p['pred_margin'],
                home_odds=float(odds.get('home_spread_odds', -110)),
                away_odds=float(odds.get('away_spread_odds', -110)),
                sigma=NCAAB_SPREAD_SIGMA,
            )

        # Collect picks with EV >= 3%
        if result_total and result_total.get('recommendation') == 'BET':
            ev = result_total.get('ev', 0)
            picks.append({
                'game': f"{away_name} @ {home_name}",
                'bet_type': 'TOTAL',
                'direction': result_total.get('direction', ''),
                'line': result_total.get('line', odds.get('total')),
                'bet_odds': result_total.get('bet_odds', -110),
                'model_projection': p['pred_total'],
                'market_line': odds.get('total'),
                'total_diff': result_total.get('total_diff', 0),
                'ev': ev,
                'ev_pct': result_total.get('ev_pct', ''),
                'model_probability': result_total.get('model_probability', 0),
                'suggested_bet': result_total.get('suggested_bet', 0),
            })

        if result_spread and result_spread.get('recommendation') == 'BET':
            ev = result_spread.get('ev', 0)
            picks.append({
                'game': f"{away_name} @ {home_name}",
                'bet_type': 'SPREAD',
                'direction': result_spread.get('bet_side', '').upper(),
                'team': result_spread.get('bet_team', ''),
                'line': result_spread.get('bet_line', odds.get('home_spread')),
                'bet_odds': result_spread.get('bet_odds', -110),
                'model_projection': p['pred_margin'],
                'market_spread': odds.get('home_spread'),
                'spread_diff': result_spread.get('spread_diff', 0),
                'ev': ev,
                'ev_pct': result_spread.get('ev_pct', ''),
                'model_probability': result_spread.get('model_probability', 0),
                'suggested_bet': result_spread.get('suggested_bet', 0),
            })

    # Sort by EV descending
    picks.sort(key=lambda x: x['ev'], reverse=True)
    return picks


# ── Output Formatting ──────────────────────────────────────────────────────

def format_picks_text(picks: list, game_date: str) -> str:
    """Format picks as plain text."""
    lines = [
        f"NCAAB DAILY PICKS — {game_date}",
        f"EV threshold: ≥3% | Model: NCAAB v1",
        "=" * 50,
    ]

    if not picks:
        lines.append("No picks today (no games with EV >= 3%)")
        return '\n'.join(lines)

    for i, pick in enumerate(picks, 1):
        lines.append(f"\nPick #{i}: {pick['game']}")
        if pick['bet_type'] == 'TOTAL':
            lines.append(f"  Bet: {pick['direction']} {pick['line']}")
            lines.append(f"  Model: {pick['model_projection']:.0f} pts vs Market: {pick['market_line']}")
            lines.append(f"  Edge: {pick['total_diff']:+.1f} pts")
        else:
            lines.append(f"  Bet: {pick.get('team', '')} {pick['line']:+.1f}")
            lines.append(f"  Model margin: {pick['model_projection']:+.1f} vs Market: {pick['market_spread']:+.1f}")
            lines.append(f"  Edge: {pick.get('spread_diff', 0):+.1f} pts")
        lines.append(f"  Odds: {int(pick['bet_odds']):+d}")
        lines.append(f"  EV: {pick['ev_pct']}")
        lines.append(f"  Win prob: {pick['model_probability']:.1%}")
        lines.append(f"  Kelly bet: ${pick['suggested_bet']:.0f}")

    return '\n'.join(lines)


def format_picks_discord(picks: list, game_date: str) -> str:
    """Format picks with Discord-friendly emoji."""
    lines = [
        f"🏀 **NCAAB DAILY PICKS — {game_date}**",
        f"📊 EV threshold: ≥3% | Model: NCAAB v1",
        "─" * 40,
    ]

    if not picks:
        lines.append("⛔ No picks today (no games with EV ≥ 3%)")
        return '\n'.join(lines)

    for i, pick in enumerate(picks, 1):
        ev_emoji = "🔥" if pick['ev'] >= 0.08 else "✅"
        lines.append(f"\n{ev_emoji} **Pick #{i}:** {pick['game']}")

        if pick['bet_type'] == 'TOTAL':
            dir_emoji = "📈" if pick['direction'] == 'OVER' else "📉"
            lines.append(f"  {dir_emoji} **{pick['direction']} {pick['line']}** (`{int(pick['bet_odds']):+d}`)")
            lines.append(f"  🤖 Model: `{pick['model_projection']:.0f}` vs Market: `{pick['market_line']}`  |  Edge: `{pick['total_diff']:+.1f} pts`")
        else:
            lines.append(f"  📋 **{pick.get('team', '')} {pick['line']:+.1f}** (`{int(pick['bet_odds']):+d}`)")
            lines.append(f"  🤖 Model: `{pick['model_projection']:+.1f}` vs Market: `{pick['market_spread']:+.1f}`  |  Edge: `{pick.get('spread_diff', 0):+.1f} pts`")

        lines.append(f"  💰 EV: **{pick['ev_pct']}** | Win prob: `{pick['model_probability']:.1%}` | Kelly: `${pick['suggested_bet']:.0f}`")

    lines.append("\n─" * 40)
    lines.append(f"⚠️ *Bet responsibly. Model-based picks only — always verify lines.*")

    return '\n'.join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCAAB Daily Picks")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to fetch games for (YYYY-MM-DD). Default: today.")
    parser.add_argument("--discord", action="store_true",
                        help="Format output for Discord with emoji.")
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    log.info(f"Running NCAAB picks for {game_date}")

    # Load models
    try:
        total_model, spread_model = load_models()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load historical games from DB
    try:
        init_db()
        games_df = get_all_games_df()
        if games_df.empty:
            print("WARNING: No historical games in DB. Run espn_loader.py first.")
            games_df = pd.DataFrame(columns=[
                'game_id', 'date', 'home_team_id', 'away_team_id',
                'home_score', 'away_score', 'neutral_site', 'season'
            ])
        else:
            games_df['home_team_id'] = games_df['home_team_id'].astype(str)
            games_df['away_team_id'] = games_df['away_team_id'].astype(str)
            log.info(f"Loaded {len(games_df)} historical games from DB")
    except Exception as e:
        log.error(f"DB error: {e}")
        games_df = pd.DataFrame()

    # Fetch today's games
    todays_games = fetch_todays_games(game_date)
    if not todays_games:
        msg = f"No NCAAB games found for {game_date}"
        if args.discord:
            print(f"🏀 **NCAAB DAILY PICKS — {game_date}**\n⛔ {msg}")
        else:
            print(msg)
        return

    # Generate predictions
    predictions = []
    skipped = 0
    for game in todays_games:
        pred = predict_game(game, games_df, total_model, spread_model)
        if pred is not None:
            predictions.append({'game': game, 'prediction': pred})
        else:
            skipped += 1
            log.debug(f"Skipped {game['home_team_name']} (insufficient history)")

    log.info(f"Generated predictions for {len(predictions)}/{len(todays_games)} games ({skipped} skipped)")

    # Fetch odds
    odds_map = fetch_ncaab_odds()

    # Compute picks
    picks = compute_picks(predictions, odds_map)
    log.info(f"Found {len(picks)} picks with EV >= 3%")

    # Output
    if args.discord:
        print(format_picks_discord(picks, game_date))
    else:
        print(format_picks_text(picks, game_date))


if __name__ == "__main__":
    main()

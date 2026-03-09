"""
NCAAB Feature Engineering

Builds rolling team stats and game-level features for model training.

Usage:
    from college.features import build_training_dataset
    import pandas as pd
    df = pd.read_sql_query("SELECT * FROM ncaab_games ORDER BY date", conn)
    features_df = build_training_dataset(df)
"""

import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

sys.path.insert(0, '.')

from college.config import (
    NCAAB_FEATURES_CSV, ROLLING_WINDOW, MIN_GAMES_REQUIRED,
    HOME_COURT_ADVANTAGE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

FEATURE_COLS = [
    'home_avg_pts_for',
    'home_avg_pts_against',
    'home_last5_wins',
    'home_home_wins_pct',
    'home_away_wins_pct',
    'home_avg_margin',
    'away_avg_pts_for',
    'away_avg_pts_against',
    'away_last5_wins',
    'away_home_wins_pct',
    'away_away_wins_pct',
    'away_avg_margin',
    'home_court_flag',
    'implied_total',
    'efficiency_diff',
]


def build_team_rolling_stats(
    games_df: pd.DataFrame,
    team_id: str,
    before_date: str,
    window: int = ROLLING_WINDOW
) -> Optional[Dict[str, float]]:
    """
    Build rolling stats for a team before a given date.

    Returns dict with:
        avg_pts_for, avg_pts_against, home_wins_pct, away_wins_pct,
        last5_wins, avg_margin
    Returns None if team has fewer than MIN_GAMES_REQUIRED games.
    """
    team_id = str(team_id)

    # Get all games where this team participated, before the date
    mask = (
        ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
        (games_df['date'] < before_date)
    )
    team_games = games_df[mask].sort_values('date')

    if len(team_games) < MIN_GAMES_REQUIRED:
        return None

    # Build per-game stats from team's perspective
    records = []
    for _, row in team_games.iterrows():
        is_home = (row['home_team_id'] == team_id)
        if is_home:
            pts_for = row['home_score']
            pts_against = row['away_score']
            won = pts_for > pts_against
            location = 'home'
        else:
            pts_for = row['away_score']
            pts_against = row['home_score']
            won = pts_for > pts_against
            location = 'away'

        records.append({
            'pts_for': pts_for,
            'pts_against': pts_against,
            'won': int(won),
            'margin': pts_for - pts_against,
            'location': location,
        })

    tdf = pd.DataFrame(records)

    # Use last N games for rolling window
    recent = tdf.tail(window)
    last5 = tdf.tail(5)

    home_games = tdf[tdf['location'] == 'home']
    away_games = tdf[tdf['location'] == 'away']

    return {
        'avg_pts_for': float(recent['pts_for'].mean()),
        'avg_pts_against': float(recent['pts_against'].mean()),
        'home_wins_pct': float(home_games['won'].mean()) if len(home_games) > 0 else 0.5,
        'away_wins_pct': float(away_games['won'].mean()) if len(away_games) > 0 else 0.5,
        'last5_wins': int(last5['won'].sum()),
        'avg_margin': float(recent['margin'].mean()),
    }


def build_game_features(
    game_row: pd.Series,
    games_df: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Build full feature dict for a game.

    Returns None if either team lacks sufficient history.
    """
    home_id = str(game_row['home_team_id'])
    away_id = str(game_row['away_team_id'])
    game_date = game_row['date']

    home_stats = build_team_rolling_stats(games_df, home_id, game_date)
    away_stats = build_team_rolling_stats(games_df, away_id, game_date)

    if home_stats is None or away_stats is None:
        return None

    neutral = int(game_row.get('neutral_site', 0))
    home_court_flag = 0 if neutral else 1

    implied_total = home_stats['avg_pts_for'] + away_stats['avg_pts_for']
    efficiency_diff = (
        (home_stats['avg_pts_for'] - home_stats['avg_pts_against']) -
        (away_stats['avg_pts_for'] - away_stats['avg_pts_against'])
    )

    return {
        # Home team features
        'home_avg_pts_for': home_stats['avg_pts_for'],
        'home_avg_pts_against': home_stats['avg_pts_against'],
        'home_last5_wins': home_stats['last5_wins'],
        'home_home_wins_pct': home_stats['home_wins_pct'],
        'home_away_wins_pct': home_stats['away_wins_pct'],
        'home_avg_margin': home_stats['avg_margin'],
        # Away team features
        'away_avg_pts_for': away_stats['avg_pts_for'],
        'away_avg_pts_against': away_stats['avg_pts_against'],
        'away_last5_wins': away_stats['last5_wins'],
        'away_home_wins_pct': away_stats['home_wins_pct'],
        'away_away_wins_pct': away_stats['away_wins_pct'],
        'away_avg_margin': away_stats['avg_margin'],
        # Context
        'home_court_flag': home_court_flag,
        'implied_total': implied_total,
        'efficiency_diff': efficiency_diff,
        # Targets
        'target_total': int(game_row['home_score']) + int(game_row['away_score']),
        'target_margin': int(game_row['home_score']) - int(game_row['away_score']),
        # Metadata
        'game_id': game_row['game_id'],
        'date': game_date,
        'season': game_row['season'],
        'home_team_id': home_id,
        'away_team_id': away_id,
    }


def build_training_dataset(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build training dataset from all games.
    Saves to NCAAB_FEATURES_CSV.

    Args:
        games_df: DataFrame of all games from DB

    Returns:
        DataFrame of game features (one row per game)
    """
    log.info(f"Building training dataset from {len(games_df)} games...")

    # Sort by date for proper temporal ordering
    games_df = games_df.sort_values('date').reset_index(drop=True)

    # Ensure string IDs
    games_df['home_team_id'] = games_df['home_team_id'].astype(str)
    games_df['away_team_id'] = games_df['away_team_id'].astype(str)

    feature_rows = []
    skipped = 0

    for i, (_, row) in enumerate(games_df.iterrows()):
        if i % 500 == 0 and i > 0:
            log.info(f"  Processed {i}/{len(games_df)} games ({len(feature_rows)} valid, {skipped} skipped)...")

        feats = build_game_features(row, games_df)
        if feats is not None:
            feature_rows.append(feats)
        else:
            skipped += 1

    if not feature_rows:
        log.error("No features built — check data")
        return pd.DataFrame()

    df = pd.DataFrame(feature_rows)

    # Save to CSV
    import os
    os.makedirs(os.path.dirname(NCAAB_FEATURES_CSV), exist_ok=True)
    df.to_csv(NCAAB_FEATURES_CSV, index=False)
    log.info(f"Saved {len(df)} feature rows to {NCAAB_FEATURES_CSV}")
    log.info(f"Skipped {skipped} games (insufficient history)")

    return df


if __name__ == "__main__":
    import sqlite3
    from college.espn_loader import get_db_conn

    conn = get_db_conn()
    games_df = pd.read_sql_query("SELECT * FROM ncaab_games ORDER BY date ASC", conn)
    conn.close()

    print(f"Loaded {len(games_df)} games from DB")

    df = build_training_dataset(games_df)
    print(f"Feature dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

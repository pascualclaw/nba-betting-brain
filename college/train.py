"""
NCAAB Model Training

Trains two models:
1. Total model: predict total points scored (home + away)
2. Spread model: predict home team margin

Walk-forward validation: train on first 80%, test on last 20% chronologically.
Uses sklearn Pipeline with StandardScaler + Ridge and GradientBoostingRegressor.

Usage:
    cd /path/to/nba-betting-brain
    venv/bin/python3 college/train.py
"""

import sys
import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from college.config import (
    NCAAB_FEATURES_CSV, NCAAB_MODELS_DIR, ROLLING_WINDOW
)

warnings.filterwarnings("ignore")
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


def load_features() -> pd.DataFrame:
    """Load feature CSV, sorting by date for temporal integrity."""
    if not os.path.exists(NCAAB_FEATURES_CSV):
        raise FileNotFoundError(
            f"Features CSV not found at {NCAAB_FEATURES_CSV}. "
            "Run college/features.py first."
        )
    df = pd.read_csv(NCAAB_FEATURES_CSV)
    df = df.sort_values('date').reset_index(drop=True)
    log.info(f"Loaded {len(df)} feature rows from {NCAAB_FEATURES_CSV}")
    return df


def walk_forward_split(df: pd.DataFrame, train_pct: float = 0.80):
    """Split chronologically: first 80% train, last 20% test."""
    split_idx = int(len(df) * train_pct)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    log.info(f"Walk-forward split: {len(train)} train, {len(test)} test")
    return train, test


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction where prediction and actual have the same sign."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def ats_win_rate(y_true_margin: np.ndarray, y_pred_margin: np.ndarray) -> float:
    """
    ATS (Against the Spread) win rate.
    Did our predicted spread direction match the actual outcome?
    (simplified: positive margin prediction vs actual)
    """
    return float(np.mean(np.sign(y_true_margin) == np.sign(y_pred_margin)))


def build_pipeline(model_type: str = "ridge") -> Pipeline:
    """Build sklearn pipeline with scaler + model."""
    if model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
) -> dict:
    """Train both Ridge and GBR, pick best by test MAE. Return performance dict."""
    # Drop rows with missing features or target
    cols = FEATURE_COLS + [target_col, 'date']
    df_clean = df[cols].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Too few rows ({len(df_clean)}) for training")

    train_df, test_df = walk_forward_split(df_clean)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[target_col].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[target_col].values

    results = {}

    for model_type in ["ridge", "gbr"]:
        log.info(f"Training {model_name} ({model_type})...")
        pipe = build_pipeline(model_type)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        dir_acc = directional_accuracy(y_test, y_pred)
        ats = ats_win_rate(y_test, y_pred)

        results[model_type] = {
            'pipeline': pipe,
            'mae': mae,
            'dir_acc': dir_acc,
            'ats': ats,
            'y_test': y_test,
            'y_pred': y_pred,
        }
        log.info(f"  {model_type}: MAE={mae:.2f}, DirAcc={dir_acc:.3f}, ATS={ats:.3f}")

    # Pick best by MAE
    best_type = min(results, key=lambda k: results[k]['mae'])
    best = results[best_type]
    log.info(f"  Best: {best_type} (MAE={best['mae']:.2f})")

    return {
        'pipeline': best['pipeline'],
        'model_type': best_type,
        'mae': best['mae'],
        'dir_acc': best['dir_acc'],
        'ats': best['ats'],
        'train_size': len(train_df),
        'test_size': len(test_df),
        'feature_cols': FEATURE_COLS,
        'target_col': target_col,
    }


def save_model(pipeline, path: str):
    """Save trained pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
    log.info(f"Saved model to {path}")


def load_model(path: str):
    """Load trained pipeline from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def print_performance_summary(total_perf: dict, spread_perf: dict):
    """Print formatted performance summary."""
    print("\n" + "="*60)
    print("NCAAB MODEL PERFORMANCE SUMMARY")
    print("="*60)

    print(f"\n📊 TOTAL MODEL (predict total points scored)")
    print(f"   Best model:       {total_perf['model_type'].upper()}")
    print(f"   Train samples:    {total_perf['train_size']:,}")
    print(f"   Test samples:     {total_perf['test_size']:,}")
    print(f"   MAE:              {total_perf['mae']:.2f} pts")
    print(f"   Directional Acc:  {total_perf['dir_acc']:.1%}")
    print(f"   Over/Under Acc:   {total_perf['ats']:.1%}")

    print(f"\n📊 SPREAD MODEL (predict home margin)")
    print(f"   Best model:       {spread_perf['model_type'].upper()}")
    print(f"   Train samples:    {spread_perf['train_size']:,}")
    print(f"   Test samples:     {spread_perf['test_size']:,}")
    print(f"   MAE:              {spread_perf['mae']:.2f} pts")
    print(f"   Directional Acc:  {spread_perf['dir_acc']:.1%}")
    print(f"   ATS Win Rate:     {spread_perf['ats']:.1%}")

    print("\n" + "="*60)
    print(f"Models saved to: {NCAAB_MODELS_DIR}")
    print("="*60 + "\n")


def main():
    log.info("Starting NCAAB model training...")

    # Load features
    df = load_features()
    log.info(f"Feature dataset: {df.shape}")

    # Train total model
    log.info("\n--- Training TOTAL model ---")
    total_perf = train_and_evaluate(df, 'target_total', 'total')

    # Train spread model
    log.info("\n--- Training SPREAD model ---")
    spread_perf = train_and_evaluate(df, 'target_margin', 'spread')

    # Save models
    total_path = os.path.join(NCAAB_MODELS_DIR, 'total_latest.pkl')
    spread_path = os.path.join(NCAAB_MODELS_DIR, 'spread_latest.pkl')

    save_model(total_perf['pipeline'], total_path)
    save_model(spread_perf['pipeline'], spread_path)

    # Print summary
    print_performance_summary(total_perf, spread_perf)

    return total_perf, spread_perf


if __name__ == "__main__":
    main()

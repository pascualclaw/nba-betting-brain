"""
Market Residual Model
=====================
Target  : actual_total - math_total_projection
          (the gap between what actually happened and what the pure math model expected)
Features: all columns EXCEPT math_total_projection (to avoid circular reasoning)

Walk-forward validation:
  - Split data chronologically by date
  - Train on first 70 %, evaluate on rolling 20 % blocks, final 10 % as holdout
  - Report MAE / RMSE per fold

Two models trained:
  1. Ridge regression (linear baseline)
  2. HistGradientBoostingRegressor (handles missing values natively)

The better model (lower holdout MAE) is saved to:
  models/saved/residual_latest.pkl

Usage:
    cd /path/to/nba-betting-brain
    python training/train_residual.py
    python training/train_residual.py --csv data/training_features.csv
"""

import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import MODEL_SAVE_DIR, DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

TARGET_COL = "residual"  # actual_total - math_total_projection
# Columns to drop from feature set (targets, leakage, metadata)
DROP_COLS = [
    "math_total_projection",   # ← the variable we're correcting; must not be a feature
    "actual_total",            # leakage (is the target indirectly)
    "actual_home_margin",      # leakage
    "home_won",                # leakage
    "game_id",                 # identifier
    "date",                    # will be used for walk-forward, not a feature
    "season",                  # categorical metadata
    "home",                    # categorical team name
    "away",                    # categorical team name
]

N_WALK_FOLDS = 5             # number of walk-forward evaluation folds
MIN_TRAIN_GAMES = 500        # minimum games needed to start training


# ── Data Loading ────────────────────────────────────────────────────────────

def load_features(csv_path: Path) -> pd.DataFrame:
    """Load feature CSV, compute residual target, return cleaned DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training features not found at {csv_path}.\n"
            "Run: python training/historical_loader.py first."
        )

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path}")

    # Require these columns
    required = ["actual_total", "math_total_projection", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute target: the residual the market gets wrong
    df[TARGET_COL] = df["actual_total"] - df["math_total_projection"]

    # Sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log.info(f"Residual stats: mean={df[TARGET_COL].mean():.2f}, "
             f"std={df[TARGET_COL].std():.2f}, "
             f"range=[{df[TARGET_COL].min():.1f}, {df[TARGET_COL].max():.1f}]")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Build X (features) and y (residual target).
    Drops non-numeric, leakage, and metadata columns.
    Returns (X, y, feature_names).
    """
    y = df[TARGET_COL].values

    # Drop unwanted columns
    drop = [c for c in DROP_COLS + [TARGET_COL] if c in df.columns]
    X_df = df.drop(columns=drop)

    # Drop remaining non-numeric columns
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.info(f"Dropping non-numeric columns: {non_numeric}")
        X_df = X_df.drop(columns=non_numeric)

    feature_names = X_df.columns.tolist()
    X = X_df.values
    log.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, feature_names


# ── Walk-Forward Validation ─────────────────────────────────────────────────

def walk_forward_eval(X: np.ndarray, y: np.ndarray, n_folds: int = N_WALK_FOLDS):
    """
    Walk-forward validation: train on expanding window, test on next block.

    Returns:
        ridge_maes, hgb_maes (lists of MAE per fold)
        ridge_rmses, hgb_rmses (lists of RMSE per fold)
    """
    n = len(X)
    fold_size = n // (n_folds + 1)
    min_train = max(MIN_TRAIN_GAMES, fold_size)

    ridge_maes, hgb_maes = [], []
    ridge_rmses, hgb_rmses = [], []

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = train_end + fold_size

        if test_end > n:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]

        # Ridge pipeline (impute + scale + ridge)
        ridge_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        ridge_pipe.fit(X_tr, y_tr)
        ridge_preds = ridge_pipe.predict(X_te)
        ridge_mae = mean_absolute_error(y_te, ridge_preds)
        ridge_rmse = np.sqrt(mean_squared_error(y_te, ridge_preds))
        ridge_maes.append(ridge_mae)
        ridge_rmses.append(ridge_rmse)

        # HistGradientBoosting (handles NaN natively, no imputation needed)
        hgb = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=20,
            random_state=42,
        )
        hgb.fit(X_tr, y_tr)
        hgb_preds = hgb.predict(X_te)
        hgb_mae = mean_absolute_error(y_te, hgb_preds)
        hgb_rmse = np.sqrt(mean_squared_error(y_te, hgb_preds))
        hgb_maes.append(hgb_mae)
        hgb_rmses.append(hgb_rmse)

        log.info(
            f"Fold {fold + 1}/{n_folds}  "
            f"train={train_end}  test={fold_size}  "
            f"Ridge MAE={ridge_mae:.3f}  HGB MAE={hgb_mae:.3f}"
        )

    return ridge_maes, hgb_maes, ridge_rmses, hgb_rmses


# ── Final Training ───────────────────────────────────────────────────────────

def train_final_models(X_train: np.ndarray, y_train: np.ndarray):
    """Train final Ridge and HGB models on the full training set."""
    ridge_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    ridge_pipe.fit(X_train, y_train)

    hgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
    )
    hgb.fit(X_train, y_train)

    return ridge_pipe, hgb


# ── Main ────────────────────────────────────────────────────────────────────

def main(csv_path: Path, save_dir: Path):
    df = load_features(csv_path)
    X, y, feature_names = build_feature_matrix(df)

    if len(X) < MIN_TRAIN_GAMES:
        log.error(
            f"Only {len(X)} games available; need at least {MIN_TRAIN_GAMES}. "
            "Run historical_loader.py to fetch more data."
        )
        return

    # ── Walk-forward validation ──────────────────────────────────────────
    log.info(f"\nRunning {N_WALK_FOLDS}-fold walk-forward validation...")
    ridge_maes, hgb_maes, ridge_rmses, hgb_rmses = walk_forward_eval(X, y)

    avg_ridge_mae = float(np.mean(ridge_maes)) if ridge_maes else float("inf")
    avg_hgb_mae = float(np.mean(hgb_maes)) if hgb_maes else float("inf")
    avg_ridge_rmse = float(np.mean(ridge_rmses)) if ridge_rmses else float("inf")
    avg_hgb_rmse = float(np.mean(hgb_rmses)) if hgb_rmses else float("inf")

    log.info(f"\nWalk-Forward Summary:")
    log.info(f"  Ridge  — Avg MAE: {avg_ridge_mae:.3f}  Avg RMSE: {avg_ridge_rmse:.3f}")
    log.info(f"  HGB    — Avg MAE: {avg_hgb_mae:.3f}  Avg RMSE: {avg_hgb_rmse:.3f}")

    # ── Holdout eval (last 10 %) ─────────────────────────────────────────
    holdout_start = int(len(X) * 0.90)
    X_train_full = X[:holdout_start]
    y_train_full = y[:holdout_start]
    X_holdout = X[holdout_start:]
    y_holdout = y[holdout_start:]

    ridge_final, hgb_final = train_final_models(X_train_full, y_train_full)

    ridge_holdout_mae = mean_absolute_error(y_holdout, ridge_final.predict(X_holdout))
    hgb_holdout_mae = mean_absolute_error(y_holdout, hgb_final.predict(X_holdout))

    log.info(f"\nHoldout MAE (last 10% of data):")
    log.info(f"  Ridge  : {ridge_holdout_mae:.3f}")
    log.info(f"  HGB    : {hgb_holdout_mae:.3f}")

    # ── Pick best model ──────────────────────────────────────────────────
    if hgb_holdout_mae <= ridge_holdout_mae:
        best_model = hgb_final
        best_name = "HistGradientBoostingRegressor"
        best_mae = hgb_holdout_mae
        # Retrain on all data for deployment
        best_model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=20, random_state=42,
        )
        best_model.fit(X, y)
    else:
        best_model = ridge_final
        best_name = "Ridge"
        best_mae = ridge_holdout_mae
        # Retrain on all data for deployment
        best_model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        best_model.fit(X, y)

    log.info(f"\nBest model: {best_name}  (holdout MAE={best_mae:.3f})")

    # ── Save ─────────────────────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "residual_latest.pkl"

    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "target_col": TARGET_COL,
        "trained_at": datetime.now().isoformat(),
        "n_train_games": len(X),
        "holdout_mae": best_mae,
        "walk_forward": {
            "ridge_avg_mae": avg_ridge_mae,
            "hgb_avg_mae": avg_hgb_mae,
            "ridge_avg_rmse": avg_ridge_rmse,
            "hgb_avg_rmse": avg_hgb_rmse,
            "n_folds": len(ridge_maes),
        },
    }

    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"✅ Saved residual model → {save_path}")

    # ── Write performance JSON ───────────────────────────────────────────
    perf_path = save_dir.parent.parent / "data" / "residual_model_performance.json"
    perf = {
        "saved_at": datetime.now().isoformat(),
        "best_model": best_name,
        "holdout_mae": round(best_mae, 4),
        "n_train_games": len(X),
        "n_features": len(feature_names),
        "walk_forward_ridge_mae": round(avg_ridge_mae, 4),
        "walk_forward_hgb_mae": round(avg_hgb_mae, 4),
    }
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(perf_path, "w") as f:
        json.dump(perf, f, indent=2)
    log.info(f"✅ Performance saved → {perf_path}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train market residual model")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "training_features.csv",
        help="Path to training features CSV (default: data/training_features.csv)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(MODEL_SAVE_DIR),
        help="Directory to save model (default: models/saved/)",
    )
    args = parser.parse_args()

    main(args.csv, args.save_dir)

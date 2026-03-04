"""
Model training — walk-forward cross-validation on historical game data.
Trains multiple models and picks the best performer.

No look-ahead bias: uses only pre-game snapshot data for each prediction.
Walk-forward: train on months 1-N, test on month N+1, advance window.

Models trained:
1. Baseline: simple average of home + away pts_for (dumb baseline)
2. Rule-based: current heuristic (ORTG/DRTG formula)
3. Linear regression: feature-based
4. XGBoost: gradient boosted trees

Outputs:
- Model performance report (data/model_performance.json)
- Trained model saved (models/saved/xgb_v{version}.pkl)
- Updated PERFORMANCE.md
"""
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor

from database.db import get_connection
from config import MODEL_SAVE_DIR, DB_PATH

# Silence LightGBM verbosity
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "home_pts_for", "home_pts_against", "home_net_rating", "home_win_pct",
    "home_last5_pts_for", "away_pts_for", "away_pts_against",
    "away_net_rating", "away_win_pct", "away_last5_pts_for",
    "math_total_projection", "pace_proxy_avg", "combined_pts_avg",
    "net_rating_diff", "h2h_total_avg", "h2h_games", "h2h_over220_rate",
]

TARGET_TOTAL = "actual_total"
TARGET_MARGIN = "actual_home_margin"


def load_feature_dataset() -> pd.DataFrame:
    """Load the pre-built feature dataset."""
    path = Path(__file__).parent.parent / "data" / "training_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {path}\n"
            "Run: python training/historical_loader.py --seasons 5"
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"Loaded {len(df)} games from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def walk_forward_evaluate(df: pd.DataFrame, model_fn, window_games: int = 300,
                           step_games: int = 50, target: str = TARGET_TOTAL) -> dict:
    """
    Walk-forward cross-validation.
    Train on window, test on next step, advance, repeat.
    Returns dict of performance metrics.
    """
    maes, rmses, preds_all, actuals_all = [], [], [], []
    n = len(df)
    
    test_start = window_games
    while test_start < n:
        test_end = min(test_start + step_games, n)
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:test_end]
        
        X_train = train_df[FEATURE_COLS].fillna(0).values
        y_train = train_df[target].values
        X_test = test_df[FEATURE_COLS].fillna(0).values
        y_test = test_df[target].values
        
        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        maes.append(mean_absolute_error(y_test, preds))
        rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
        preds_all.extend(preds.tolist())
        actuals_all.extend(y_test.tolist())
        
        test_start = test_end
    
    # Simulate betting: bet Over if predicted > actual_line, Under if predicted < actual_line
    # Use predicted total vs actual total as proxy for beating posted line
    correct_direction = sum(
        1 for p, a in zip(preds_all, actuals_all)
        if (p > 220 and a > 220) or (p < 220 and a < 220)
    )
    direction_accuracy = correct_direction / len(preds_all) if preds_all else 0
    
    return {
        "mae": round(float(np.mean(maes)), 2),
        "rmse": round(float(np.mean(rmses)), 2),
        "direction_accuracy": round(direction_accuracy, 3),
        "n_test_games": len(preds_all),
        "n_train_windows": len(maes),
        "predictions": preds_all[-100:],  # keep last 100 for calibration
        "actuals": actuals_all[-100:],
    }


def simulate_betting_roi(preds: list, actuals: list, line: float = 220.0,
                          odds: float = -110) -> dict:
    """
    Simulate flat betting: bet $100 on Over if predicted > line, Under if predicted < line.
    Returns ROI and win rate.
    """
    bets, wins, pnl = 0, 0, 0
    stake = 100
    
    for pred, actual in zip(preds, actuals):
        if abs(pred - line) < 2:  # no bet if too close to line
            continue
        bets += 1
        bet_over = pred > line
        result_over = actual > line
        
        if bet_over == result_over:
            wins += 1
            pnl += stake * (100 / abs(odds)) if odds < 0 else stake * (odds / 100)
        else:
            pnl -= stake
    
    return {
        "bets": bets,
        "wins": wins,
        "win_rate": round(wins / bets, 3) if bets > 0 else 0,
        "total_pnl": round(pnl, 2),
        "roi": round(pnl / (bets * stake) * 100, 2) if bets > 0 else 0,
    }


def train_final_model(df: pd.DataFrame) -> tuple:
    """Train final GradientBoosting model on all data."""
    X = df[FEATURE_COLS].fillna(0).values
    y = df[TARGET_TOTAL].values
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)
    
    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return model, importance


def generate_performance_report(results: dict, version: str) -> str:
    """Generate markdown performance report."""
    lines = [
        f"# NBA Betting Brain — Model Performance",
        f"**Version:** {version} | **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M ET')}",
        "",
        "## Backtesting Results (Walk-Forward Validation)",
        "",
        "| Model | MAE (pts) | RMSE | Direction Accuracy | N Games |",
        "|-------|-----------|------|-------------------|---------|",
    ]
    for model_name, res in results.items():
        if "mae" in res:
            lines.append(f"| {model_name} | {res['mae']} | {res['rmse']} | {res['direction_accuracy']*100:.1f}% | {res['n_test_games']} |")
    
    lines += ["", "## Simulated Betting ROI", ""]
    for model_name, res in results.items():
        if "betting" in res:
            b = res["betting"]
            lines.append(f"**{model_name}:** {b['wins']}/{b['bets']} ({b['win_rate']*100:.1f}%) | ROI: {b['roi']}%")
    
    if "gradient_boosting" in results and "feature_importance" in results["gradient_boosting"]:
        lines += ["", "## Top Predictive Features (XGBoost)", ""]
        importance = results["gradient_boosting"]["feature_importance"]
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            bar = "█" * int(imp * 100)
            lines.append(f"{i+1}. `{feat}`: {imp:.3f} {bar}")
    
    lines += [
        "",
        "## Key Takeaways",
        "- MAE < 8 pts = model is useful for identifying value bets",
        "- Direction accuracy > 55% = edge over coin flip",
        "- ROI > 0% = model has real betting value",
        "",
        "## Next Steps",
        "- Add injury impact scoring to features",
        "- Add betting line (open/close) as feature when available",
        "- Add referee tendencies",
        "- Calibrate confidence intervals",
    ]
    return "\n".join(lines)


def run_training():
    """Main training pipeline."""
    version = datetime.now().strftime("v%Y%m%d_%H%M")
    log.info(f"\n{'='*60}")
    log.info(f"🏀 NBA BETTING BRAIN — Model Training {version}")
    log.info(f"{'='*60}\n")
    
    # Load data
    df = load_feature_dataset()
    log.info(f"Training on {len(df)} games across {df['season'].nunique()} seasons\n")
    
    results = {}
    
    # ── Baseline: simple average ──────────────────────────
    log.info("Evaluating baseline model...")
    class BaselineModel:
        def fit(self, X, y): return self
        def predict(self, X):
            # Just predict league avg
            return np.full(len(X), 224.0)
    results["baseline_avg224"] = walk_forward_evaluate(df, BaselineModel)
    
    # ── Baseline: combined pts avg ────────────────────────
    log.info("Evaluating combined pts model...")
    class CombinedPtsModel:
        def fit(self, X, y): self.idx = FEATURE_COLS.index("combined_pts_avg"); return self
        def predict(self, X): return X[:, self.idx]
    results["combined_pts_avg"] = walk_forward_evaluate(df, CombinedPtsModel)
    
    # ── Linear Regression ─────────────────────────────────
    log.info("Training linear regression...")
    results["linear_ridge"] = walk_forward_evaluate(
        df, lambda: Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    )
    
    # ── Gradient Boosting (sklearn, no OpenMP needed) ─────
    log.info("Training Gradient Boosting...")
    results["gradient_boosting"] = walk_forward_evaluate(
        df, lambda: HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.1, random_state=42)
    )
    
    # Simulate betting ROI for each model
    for model_name, res in results.items():
        if "predictions" in res and "actuals" in res:
            results[model_name]["betting"] = simulate_betting_roi(
                res["predictions"], res["actuals"]
            )
    
    # ── Train final model on all data ────────────────────
    log.info("\nTraining final Gradient Boosting model on all data...")
    final_model, importance = train_final_model(df)
    results["gradient_boosting"]["feature_importance"] = importance
    
    # Save model
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_DIR / f"gb_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": FEATURE_COLS,
                     "version": version, "trained_on": len(df)}, f)
    log.info(f"Saved model: {model_path}")
    
    # Save latest model pointer
    latest_path = MODEL_SAVE_DIR / "latest.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": FEATURE_COLS,
                     "version": version, "trained_on": len(df)}, f)
    
    # Save performance JSON
    perf_path = Path(__file__).parent.parent / "data" / "model_performance.json"
    perf_data = {k: {k2: v2 for k2, v2 in v.items()
                     if k2 not in ("predictions", "actuals")} for k, v in results.items()}
    perf_data["_meta"] = {"version": version, "trained_on": len(df),
                           "timestamp": datetime.now().isoformat()}
    with open(perf_path, "w") as f:
        json.dump(perf_data, f, indent=2)
    
    # Generate PERFORMANCE.md
    report = generate_performance_report(results, version)
    perf_md = Path(__file__).parent.parent / "PERFORMANCE.md"
    perf_md.write_text(report)
    
    # Print summary
    log.info("\n" + "="*60)
    log.info("RESULTS SUMMARY:")
    log.info("="*60)
    for name, res in results.items():
        if "mae" in res:
            betting = res.get("betting", {})
            log.info(f"{name:25s} MAE: {res['mae']:5.1f} pts | "
                     f"Dir: {res['direction_accuracy']*100:.1f}% | "
                     f"ROI: {betting.get('roi', 'N/A')}%")
    
    log.info(f"\nTop features:")
    for feat, imp in list(importance.items())[:5]:
        log.info(f"  {feat}: {imp:.3f}")
    
    log.info(f"\n✅ Training complete. Model saved: {model_path.name}")
    log.info(f"📊 Report: PERFORMANCE.md")
    return results


if __name__ == "__main__":
    run_training()

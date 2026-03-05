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
    # Team quality
    "home_pts_for", "home_pts_against", "home_net_rating", "home_win_pct",
    "home_last5_pts_for", "away_pts_for", "away_pts_against",
    "away_net_rating", "away_win_pct", "away_last5_pts_for",
    # Matchup projections
    "math_total_projection", "pace_proxy_avg", "combined_pts_avg",
    "net_rating_diff",
    # H2H
    "h2h_total_avg", "h2h_games", "h2h_over220_rate",
    # Rest / fatigue (b2b = biggest fatigue signal)
    "home_rest_days", "away_rest_days", "home_b2b", "away_b2b",
    # Injury impact (key signal — missing star players reduce total)
    "home_injury_impact", "away_injury_impact",
    # Betting lines — NaN when unavailable; HistGradientBoosting handles natively
    "open_total_line", "close_total_line",
    # ── Four Factors (Dean Oliver) — biggest model gap ──────────────────
    # eFG%: shooting efficiency accounting for 3-pointers (weight: ~40% of wins)
    "home_efg", "away_efg", "efg_diff",
    # TOV%: turnover rate = wasted possessions (weight: ~25% of wins)
    "home_tov_rate", "away_tov_rate", "tov_diff",
    # ORB%: extra possessions from offensive rebounds (weight: ~20% of wins)
    "home_orb_rate", "away_orb_rate", "orb_diff",
    # FT Rate: free throw opportunities (weight: ~15% of wins)
    "home_ft_rate", "away_ft_rate", "ft_rate_diff",
    # Net Rating: ORTG - DRTG (opponent-adjusted efficiency)
    "home_net_rtg", "away_net_rtg", "net_rtg_diff_ff",
    # ── Variance metrics ─────────────────────────────────────────────────
    "home_pts_std", "away_pts_std",
    "home_pts_floor", "home_pts_ceiling",
    "away_pts_floor", "away_pts_ceiling",
    "home_momentum", "away_momentum",
    "upset_risk_score",
    # ── 3-point era features ─────────────────────────────────────────────
    "home_3pa_rate", "away_3pa_rate",    # 3-point attempt rate
    "home_3p_pct", "away_3p_pct",        # 3-point make %
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


def compute_edge_accuracy(
    preds: list[float],
    actuals: list[float],
    lines: list[float],
) -> float:
    """
    Edge accuracy = % of games where sign(predicted - line) == sign(actual - line).

    In other words: when we think the game will go Over the line, does it?
    Only counts games where a real betting line was available.

    Args:
        preds:   Model's total predictions.
        actuals: Actual final totals.
        lines:   Opening betting line (NaN if unavailable).

    Returns:
        Edge accuracy in [0, 1], or NaN if no games have line data.
    """
    correct = 0
    total = 0
    for p, a, line in zip(preds, actuals, lines):
        if line is None or (isinstance(line, float) and np.isnan(line)):
            continue
        # Direction: positive = Over, negative = Under
        pred_direction = p - line
        actual_direction = a - line
        if pred_direction * actual_direction > 0:  # same sign
            correct += 1
        total += 1

    return round(correct / total, 3) if total > 0 else float("nan")


def walk_forward_evaluate(df: pd.DataFrame, model_fn, window_games: int = 300,
                           step_games: int = 50, target: str = TARGET_TOTAL) -> dict:
    """
    Walk-forward cross-validation.
    Train on window, test on next step, advance, repeat.
    Returns dict of performance metrics including edge accuracy when lines available.
    """
    maes, rmses, preds_all, actuals_all, lines_all = [], [], [], [], []
    n = len(df)

    # Determine which FEATURE_COLS actually exist in the dataframe
    available_cols = [c for c in FEATURE_COLS if c in df.columns]

    test_start = window_games
    while test_start < n:
        test_end = min(test_start + step_games, n)
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:test_end]

        X_train = train_df[available_cols].fillna(0).values
        y_train = train_df[target].values
        X_test = test_df[available_cols].fillna(0).values
        y_test = test_df[target].values

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, preds))
        rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
        preds_all.extend(preds.tolist())
        actuals_all.extend(y_test.tolist())

        # Collect lines for edge accuracy (NaN if column missing)
        if "open_total_line" in df.columns:
            lines_all.extend(test_df["open_total_line"].tolist())
        else:
            lines_all.extend([float("nan")] * len(y_test))

        test_start = test_end

    # Simulate betting: bet Over if predicted > actual_line, Under if predicted < actual_line
    # Use predicted total vs actual total as proxy for beating posted line
    correct_direction = sum(
        1 for p, a in zip(preds_all, actuals_all)
        if (p > 220 and a > 220) or (p < 220 and a < 220)
    )
    direction_accuracy = correct_direction / len(preds_all) if preds_all else 0

    # Edge accuracy against real betting lines
    edge_acc = compute_edge_accuracy(preds_all, actuals_all, lines_all)
    n_with_lines = sum(1 for ln in lines_all if ln is not None and not (isinstance(ln, float) and np.isnan(ln)))

    return {
        "mae": round(float(np.mean(maes)), 2),
        "rmse": round(float(np.mean(rmses)), 2),
        "direction_accuracy": round(direction_accuracy, 3),
        "edge_accuracy": edge_acc,          # vs real posted line (NaN if no lines)
        "n_games_with_lines": n_with_lines,
        "n_test_games": len(preds_all),
        "n_train_windows": len(maes),
        "predictions": preds_all[-100:],    # keep last 100 for calibration
        "actuals": actuals_all[-100:],
        "lines": lines_all[-100:],
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
    """Train final GradientBoosting model on all data.
    
    Uses only feature columns that actually exist in df, so the model
    gracefully degrades if odds columns (open_total_line, close_total_line)
    are absent from older datasets.
    """
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0).values
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
    importance = dict(zip(available_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return model, importance, available_cols


def generate_performance_report(results: dict, version: str) -> str:
    """Generate markdown performance report."""
    lines = [
        f"# NBA Betting Brain — Model Performance",
        f"**Version:** {version} | **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M ET')}",
        "",
        "## Backtesting Results (Walk-Forward Validation)",
        "",
        "| Model | MAE (pts) | RMSE | Direction Acc | Edge Acc (vs line) | N Games |",
        "|-------|-----------|------|---------------|--------------------|---------|",
    ]
    for model_name, res in results.items():
        if "mae" in res:
            edge = res.get("edge_accuracy")
            n_lines = res.get("n_games_with_lines", 0)
            edge_str = f"{edge*100:.1f}% ({n_lines}g)" if edge is not None and not (isinstance(edge, float) and np.isnan(edge)) else "N/A"
            lines.append(
                f"| {model_name} | {res['mae']} | {res['rmse']} "
                f"| {res['direction_accuracy']*100:.1f}% | {edge_str} | {res['n_test_games']} |"
            )

    lines += ["", "## Simulated Betting ROI", ""]
    for model_name, res in results.items():
        if "betting" in res:
            b = res["betting"]
            lines.append(f"**{model_name}:** {b['wins']}/{b['bets']} ({b['win_rate']*100:.1f}%) | ROI: {b['roi']}%")

    if "gradient_boosting" in results and "feature_importance" in results["gradient_boosting"]:
        lines += ["", "## Top Predictive Features (Gradient Boosting)", ""]
        importance = results["gradient_boosting"]["feature_importance"]
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            bar = "█" * int(imp * 100)
            lines.append(f"{i+1}. `{feat}`: {imp:.3f} {bar}")

    lines += [
        "",
        "## Key Takeaways",
        "- MAE < 8 pts = model is useful for identifying value bets",
        "- Direction accuracy > 55% = edge over coin flip",
        "- Edge accuracy > 55% vs posted line = real betting value",
        "- ROI > 0% = model has real betting value",
        "",
        "## Next Steps",
        "- Accumulate more games with odds data to improve edge accuracy signal",
        "- Add referee tendencies",
        "- Calibrate confidence intervals",
        "- Track line movement as a feature (sharp vs. square money signal)",
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
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    combined_idx = available_cols.index("combined_pts_avg") if "combined_pts_avg" in available_cols else 0
    class CombinedPtsModel:
        def fit(self, X, y): return self
        def predict(self, X): return X[:, combined_idx]
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
    final_model, importance, used_cols = train_final_model(df)
    results["gradient_boosting"]["feature_importance"] = importance

    # Save model
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_DIR / f"gb_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": used_cols,
                     "version": version, "trained_on": len(df)}, f)
    log.info(f"Saved model: {model_path}")

    # Save latest model pointer
    latest_path = MODEL_SAVE_DIR / "latest.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": used_cols,
                     "version": version, "trained_on": len(df)}, f)
    
    # Save performance JSON
    perf_path = Path(__file__).parent.parent / "data" / "model_performance.json"
    _skip = {"predictions", "actuals", "lines"}
    perf_data = {k: {k2: v2 for k2, v2 in v.items() if k2 not in _skip}
                 for k, v in results.items()}
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
            edge = res.get("edge_accuracy")
            edge_str = f"{edge*100:.1f}%" if edge is not None and not (isinstance(edge, float) and np.isnan(edge)) else "N/A"
            log.info(f"{name:25s} MAE: {res['mae']:5.1f} pts | "
                     f"Dir: {res['direction_accuracy']*100:.1f}% | "
                     f"Edge: {edge_str} | "
                     f"ROI: {betting.get('roi', 'N/A')}%")
    
    log.info(f"\nTop features:")
    for feat, imp in list(importance.items())[:5]:
        log.info(f"  {feat}: {imp:.3f}")
    
    log.info(f"\n✅ Training complete. Model saved: {model_path.name}")
    log.info(f"📊 Report: PERFORMANCE.md")
    return results


if __name__ == "__main__":
    run_training()

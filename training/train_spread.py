"""
Spread Model Training — Predicts home team margin of victory.

Why a separate spread model:
- Totals and spreads have different market inefficiencies
- Home court advantage (~3.2 pts in NBA) creates systematic edge
- Rest differential, B2B, travel distance matter more for spreads
- Revenge games, motivation flags shift margins more than totals

Target: actual_home_margin (positive = home won by X)
Features: same as total model + spread-specific (home court, rest diff, win pct diff)

Walk-forward validation prevents data leakage.
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# ── Spread-specific features ───────────────────────────────────────────────
# Home court advantage is the single biggest signal in spread prediction
HOME_COURT_ADVANTAGE = 3.2   # NBA historical average (pts)

SPREAD_FEATURE_COLS = [
    # Core team quality
    "home_pts_for", "home_pts_against", "home_net_rating", "home_win_pct",
    "home_last5_pts_for", "away_pts_for", "away_pts_against",
    "away_net_rating", "away_win_pct", "away_last5_pts_for",
    # Margin-specific matchup features
    "net_rating_diff",          # home_net_rating - away_net_rating (key spread predictor)
    "win_pct_diff",             # derived: home_win_pct - away_win_pct
    "pts_for_diff",             # home offensive advantage
    "pts_against_diff",         # home defensive advantage
    "rest_diff",                # home_rest_days - away_rest_days (freshness edge)
    "home_b2b", "away_b2b",    # fatigue signals
    # Home/away split performance (teams play differently at home vs away)
    "home_home_ortg", "home_away_ortg",   # home team: how they score at home vs road
    "away_home_ortg", "away_away_ortg",   # away team: how they score at home vs road
    "home_home_drtg", "home_away_drtg",
    "away_home_drtg", "away_away_drtg",
    "home_home_win_pct", "away_away_win_pct",  # most predictive: home wins at home, away wins away
    # H2H history (revenge context)
    "h2h_total_avg", "h2h_games",
    # Injury impact (margin-adjusted)
    "home_injury_impact", "away_injury_impact",
    # Market spread (strongest single predictor when available)
    "open_spread",
]

TARGET = "actual_home_margin"


def load_data() -> pd.DataFrame:
    path = DATA_DIR / "training_features.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Engineer spread-specific features ─────────────────────────────────
    df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]
    df["pts_for_diff"] = df["home_pts_for"] - df["away_pts_for"]
    df["pts_against_diff"] = df["home_pts_against"] - df["away_pts_against"]  # lower = better defense
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    # Fill missing home/away splits with estimated values
    for col in ["home_home_ortg", "home_away_ortg", "away_home_ortg", "away_away_ortg",
                "home_home_drtg", "home_away_drtg", "away_home_drtg", "away_away_drtg",
                "home_home_win_pct", "away_away_win_pct"]:
        if col not in df.columns:
            df[col] = df["home_net_rating"] if "ortg" in col or "win" in col else 110.0

    # Fill NaN in open_spread with 0 (home pick'em)
    if "open_spread" in df.columns:
        df["open_spread"] = df["open_spread"].fillna(0)
    else:
        df["open_spread"] = 0.0

    # Drop games without margin data
    df = df[df[TARGET].notna()].copy()

    log.info(f"Loaded {len(df)} games | margin range: {df[TARGET].min():.0f} to {df[TARGET].max():.0f}")
    return df


def walk_forward_eval(df: pd.DataFrame, feature_cols: list, model_type: str = "ridge") -> dict:
    """
    Walk-forward validation on spread model.
    Train on games 1..N, test on games N+1..N+350.
    Expand window iteratively.
    """
    available_cols = [c for c in feature_cols if c in df.columns]
    log.info(f"  [{model_type}] Using {len(available_cols)} features")

    X = df[available_cols].fillna(0).values
    y = df[TARGET].values
    spreads = df["open_spread"].fillna(0).values

    n = len(X)
    min_train = 500
    step = 350

    all_preds = []
    all_actuals = []
    all_spreads = []

    for start in range(min_train, n - step, step):
        X_train, y_train = X[:start], y[:start]
        X_test, y_test = X[start:start + step], y[start:start + step]
        spread_test = spreads[start:start + step]

        if model_type == "ridge":
            mdl = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
        elif model_type == "hist_gb":
            mdl = HistGradientBoostingRegressor(
                max_iter=200, max_depth=4, learning_rate=0.05,
                min_samples_leaf=20, random_state=42
            )
        else:
            mdl = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_samples_leaf=20, random_state=42, subsample=0.8
            )

        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)

        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_spreads.extend(spread_test)

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    lines = np.array(all_spreads)

    mae = mean_absolute_error(actuals, preds)

    # Direction accuracy: does sign(pred) == sign(actual)?
    dir_acc = np.mean(np.sign(preds) == np.sign(actuals))

    # ATS accuracy: when model says home covers, does home cover?
    has_spread = lines != 0
    if has_spread.sum() > 0:
        model_edge = preds[has_spread] - lines[has_spread]
        model_home_covers = model_edge > 0   # model thinks home beats spread
        actual_home_covers = actuals[has_spread] > lines[has_spread]
        ats_acc = np.mean(model_home_covers == actual_home_covers)
    else:
        ats_acc = float("nan")

    # ROI simulation: bet home when pred > spread + 0.5 pt buffer
    correct_bets = 0
    total_bets = 0
    for pred, actual, line in zip(preds, actuals, lines):
        if abs(pred - line) < 1.0:  # skip near-pick'ems
            continue
        if pred > line:  # bet home
            total_bets += 1
            if actual > line:
                correct_bets += 1
        else:  # bet away
            total_bets += 1
            if actual < line:
                correct_bets += 1

    win_rate = correct_bets / total_bets if total_bets > 0 else 0.5
    roi = (win_rate * 100 - (1 - win_rate) * 110) / 110 * 100  # -110 odds

    return {
        "mae": round(mae, 2),
        "dir_acc": round(dir_acc * 100, 1),
        "ats_acc": round(ats_acc * 100, 1) if not np.isnan(ats_acc) else None,
        "roi": round(roi, 2),
        "n_games": len(preds),
        "n_bet_games": total_bets,
        "win_rate": round(win_rate * 100, 1),
    }


def train_final_spread_model(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Train final spread model on all data."""
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].fillna(0).values
    y = df[TARGET].values

    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=5, learning_rate=0.04,
        min_samples_leaf=15, random_state=42,
    )
    model.fit(X, y)

    # Feature importance (permutation-style approximation)
    importances = {}
    baseline_mae = mean_absolute_error(y, model.predict(X))
    for i, col in enumerate(available_cols[:15]):
        X_perm = X.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_mae = mean_absolute_error(y, model.predict(X_perm))
        importances[col] = round(perm_mae - baseline_mae, 4)

    top_features = sorted(importances.items(), key=lambda x: -x[1])[:10]

    return model, available_cols, top_features


def main():
    version = datetime.now().strftime("%Y%m%d_%H%M")
    log.info("=" * 60)
    log.info(f"🏀 NBA SPREAD MODEL — Training v{version}")
    log.info("=" * 60)

    df = load_data()
    log.info(f"Training on {len(df)} games | Home court avg: +{df[TARGET].mean():.1f} pts\n")

    results = {}
    for model_type in ["ridge", "hist_gb"]:
        log.info(f"Evaluating [{model_type}]...")
        res = walk_forward_eval(df, SPREAD_FEATURE_COLS, model_type)
        results[model_type] = res
        log.info(
            f"  MAE: {res['mae']:.1f} pts | Dir: {res['dir_acc']}% | "
            f"ATS: {res.get('ats_acc', 'N/A')}% | ROI: {res['roi']:.1f}% | "
            f"Win rate: {res['win_rate']}% ({res['n_bet_games']} bets)"
        )

    log.info("\nTraining final model (HistGB on all data)...")
    model, used_cols, top_features = train_final_spread_model(df, SPREAD_FEATURE_COLS)

    log.info("\nTop features:")
    for feat, importance in top_features:
        log.info(f"  {feat}: {importance:.4f}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_cols": used_cols,
        "model_type": "hist_gradient_boosting_spread",
        "version": version,
        "target": TARGET,
        "home_court_avg": float(df[TARGET].mean()),
        "spread_std": float(df[TARGET].std()),
        "performance": results,
    }

    versioned_path = MODEL_DIR / f"spread_v{version}.pkl"
    latest_path = MODEL_DIR / "spread_latest.pkl"

    with open(versioned_path, "wb") as f:
        pickle.dump(bundle, f)
    with open(latest_path, "wb") as f:
        pickle.dump(bundle, f)

    log.info(f"\n✅ Spread model saved: spread_v{version}.pkl")

    # Update PERFORMANCE.md
    perf_path = PROJECT_ROOT / "PERFORMANCE.md"
    existing = perf_path.read_text() if perf_path.exists() else ""
    spread_section = f"""
## Spread Model — v{version}

| Model | MAE | Direction | ATS Acc | Win Rate | ROI |
|-------|-----|-----------|---------|----------|-----|
"""
    for name, res in results.items():
        ats = f"{res['ats_acc']}%" if res.get("ats_acc") else "N/A"
        spread_section += f"| {name} | {res['mae']} pts | {res['dir_acc']}% | {ats} | {res['win_rate']}% | {res['roi']:.1f}% |\n"

    spread_section += f"\n**Top features:** {', '.join(f[0] for f in top_features[:5])}\n"
    spread_section += f"**Home court average:** +{df[TARGET].mean():.1f} pts\n"

    if "## Spread Model" not in existing:
        perf_path.write_text(existing + spread_section)
    log.info("📊 PERFORMANCE.md updated")

    return bundle


if __name__ == "__main__":
    main()

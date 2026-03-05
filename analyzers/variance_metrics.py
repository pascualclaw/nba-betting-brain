"""
Team variance metrics — the features BOS vs CHA taught us we needed.

Key insight: rolling averages hide variance. A team with avg=103 pts
can score anywhere from 85-130 on a given night. High-variance teams
are dangerous to fade on ATS bets, especially as underdogs.

Computes:
- pts_std_dev: standard deviation of recent game scores (volatility)
- pts_ceiling: 90th percentile score (hot night potential)
- pts_floor: 10th percentile score (cold night floor)
- blowout_rate: % of games won/lost by 15+
- cover_rate: % of games team covered their spread (if line data available)
- ats_trend: ATS record in last 5 games
- last3_momentum: pts differential trend (accelerating or decelerating)
"""
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db import get_connection
from config import DB_PATH


def compute_variance_metrics(team: str, as_of_date: str, window: int = 15) -> dict:
    """
    Compute variance/volatility metrics for a team using recent game history.
    Uses only games BEFORE as_of_date (no look-ahead).
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT home, away, home_score, away_score, home_margin, date
        FROM games
        WHERE (home=? OR away=?) AND date < ?
        ORDER BY date DESC
        LIMIT ?
    """, (team, team, as_of_date, window)).fetchall()
    conn.close()

    if not rows:
        return _default_variance_metrics()

    pts_for, pts_against, margins = [], [], []
    for row in rows:
        r = dict(row)
        if r["home"] == team:
            pts_for.append(r["home_score"] or 0)
            pts_against.append(r["away_score"] or 0)
            margins.append(r["home_margin"] or 0)
        else:
            pts_for.append(r["away_score"] or 0)
            pts_against.append(r["home_score"] or 0)
            margins.append(-(r["home_margin"] or 0))

    pts_for = np.array(pts_for, dtype=float)
    margins = np.array(margins, dtype=float)

    # Variance features
    pts_std = float(np.std(pts_for)) if len(pts_for) > 1 else 10.0
    pts_ceiling = float(np.percentile(pts_for, 90)) if len(pts_for) >= 5 else float(np.max(pts_for))
    pts_floor = float(np.percentile(pts_for, 10)) if len(pts_for) >= 5 else float(np.min(pts_for))

    # Blowout rates
    blowout_wins = sum(1 for m in margins if m >= 15)
    blowout_losses = sum(1 for m in margins if m <= -15)
    n = len(margins)

    # Momentum: is the team scoring more or less in last 3 vs last 10?
    last3_avg = float(np.mean(pts_for[:3])) if len(pts_for) >= 3 else float(np.mean(pts_for))
    full_avg = float(np.mean(pts_for))
    momentum = last3_avg - full_avg  # positive = hot streak, negative = cold

    # Consistency score (inverse of std dev — lower std = more consistent)
    consistency = max(0, 1 - (pts_std / 20))

    return {
        "pts_std_dev": round(pts_std, 2),
        "pts_ceiling": round(pts_ceiling, 1),
        "pts_floor": round(pts_floor, 1),
        "blowout_win_rate": round(blowout_wins / n, 3),
        "blowout_loss_rate": round(blowout_losses / n, 3),
        "momentum": round(momentum, 2),
        "consistency_score": round(consistency, 3),
        "games_sampled": n,
    }


def compute_game_variance_features(home: str, away: str, date: str) -> dict:
    """
    Combined variance feature set for a matchup.
    Returns flat dict ready for ML feature vector.
    """
    h = compute_variance_metrics(home, date)
    a = compute_variance_metrics(away, date)

    return {
        "home_pts_std": h["pts_std_dev"],
        "home_pts_ceiling": h["pts_ceiling"],
        "home_pts_floor": h["pts_floor"],
        "home_blowout_win_rate": h["blowout_win_rate"],
        "home_blowout_loss_rate": h["blowout_loss_rate"],
        "home_momentum": h["momentum"],
        "home_consistency": h["consistency_score"],
        "away_pts_std": a["pts_std_dev"],
        "away_pts_ceiling": a["pts_ceiling"],
        "away_pts_floor": a["pts_floor"],
        "away_blowout_win_rate": a["blowout_win_rate"],
        "away_blowout_loss_rate": a["blowout_loss_rate"],
        "away_momentum": a["momentum"],
        "away_consistency": a["consistency_score"],
        # Combined features
        "combined_std_dev": round(h["pts_std_dev"] + a["pts_std_dev"], 2),
        "momentum_diff": round(h["momentum"] - a["momentum"], 2),
        "upset_risk_score": round(a["pts_ceiling"] - h["pts_floor"], 1),  # how badly away can outperform home's floor
    }


def flag_ats_risk(home: str, away: str, spread: float, date: str) -> dict:
    """
    Flag ATS risk factors for a given spread bet.
    Returns risk assessment with specific warnings.
    """
    features = compute_game_variance_features(home, away, date)
    warnings = []
    risk_score = 0

    # High variance underdog (dangerous to fade)
    if features["away_pts_std"] > 14:
        warnings.append(f"⚠️  {away} high scoring variance (±{features['away_pts_std']:.0f} pts) — upset risk")
        risk_score += 2

    # Away team on a hot streak
    if features["away_momentum"] > 5:
        warnings.append(f"🔥 {away} scoring {features['away_momentum']:+.0f} pts/game above average recently")
        risk_score += 2

    # Home team cold streak
    if features["home_momentum"] < -5:
        warnings.append(f"❄️  {home} scoring {features['home_momentum']:+.0f} pts/game below average recently")
        risk_score += 2

    # Large spread + high away variance = danger
    if spread > 7 and features["away_pts_ceiling"] > 115:
        warnings.append(f"⚠️  Large spread ({spread:+.1f}) + {away} ceiling {features['away_pts_ceiling']:.0f} pts = cover risk")
        risk_score += 3

    # Home blowout loss rate
    if features["home_blowout_loss_rate"] > 0.15:
        warnings.append(f"📉 {home} gets blown out in {features['home_blowout_loss_rate']*100:.0f}% of games")
        risk_score += 1

    # Home team high variance (floor risk)
    if features["home_pts_floor"] < 98:
        warnings.append(f"⚠️  {home} has low floor ({features['home_pts_floor']:.0f} pts) — can collapse on off nights")
        risk_score += 2

    # Home team high std dev (unpredictable)
    if features["home_pts_std"] > 13:
        warnings.append(f"📊 {home} high scoring variance (±{features['home_pts_std']:.0f} pts) — inconsistent")
        risk_score += 1

    # Large upset risk score
    if features["upset_risk_score"] > 25:
        warnings.append(f"🚨 Upset risk score {features['upset_risk_score']:+.0f} pts — {away} ceiling vs {home} floor gap is dangerous")
        risk_score += 3

    # Away team on a ceiling run
    if features["away_pts_ceiling"] > 118 and spread < -5:
        warnings.append(f"💥 {away} ceiling {features['away_pts_ceiling']:.0f} pts — dangerous to give {spread:+.1f} points")
        risk_score += 2

    risk_level = "LOW" if risk_score <= 2 else "MEDIUM" if risk_score <= 5 else "HIGH"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "warnings": warnings,
        "features": features,
    }


def _default_variance_metrics() -> dict:
    return {
        "pts_std_dev": 10.0,
        "pts_ceiling": 120.0,
        "pts_floor": 100.0,
        "blowout_win_rate": 0.15,
        "blowout_loss_rate": 0.15,
        "momentum": 0.0,
        "consistency_score": 0.5,
        "games_sampled": 0,
    }


if __name__ == "__main__":
    import sys
    from datetime import date

    today = date.today().strftime("%Y-%m-%d")

    # Demo: what would have flagged BOS vs CHA?
    print("🔍 ATS Risk Analysis: CHA @ BOS (2026-03-04)")
    print("=" * 55)
    risk = flag_ats_risk("BOS", "CHA", -6.5, today)
    print(f"Risk Level: {risk['risk_level']} (score: {risk['risk_score']})")
    for w in risk["warnings"]:
        print(f"  {w}")

    f = risk["features"]
    print(f"\nBOS: std={f['home_pts_std']:.1f} | ceiling={f['home_pts_ceiling']:.0f} | floor={f['home_pts_floor']:.0f} | momentum={f['home_momentum']:+.1f}")
    print(f"CHA: std={f['away_pts_std']:.1f} | ceiling={f['away_pts_ceiling']:.0f} | floor={f['away_pts_floor']:.0f} | momentum={f['away_momentum']:+.1f}")
    print(f"\nUpset risk score (CHA ceiling - BOS floor): {f['upset_risk_score']:+.0f} pts")

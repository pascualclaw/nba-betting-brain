"""
pipeline/feature_engineer.py — Convert raw pre-game snapshots into ML features.

Takes the output of SnapshotBuilder and produces a flat, numeric feature vector
suitable for XGBoost/sklearn. All features are computed from pre-game data only.

Usage:
    from pipeline.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()
    features = fe.build_features(snapshot)
    X, y = fe.build_dataset(snapshots, games)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ─── Feature column order (must be stable across versions) ───────────────────

FEATURE_COLUMNS = [
    # Team quality — rolling 20 games
    "home_ortg",
    "home_drtg",
    "home_pace",
    "home_net_rtg",
    "away_ortg",
    "away_drtg",
    "away_pace",
    "away_net_rtg",
    # Matchup derived
    "pace_avg",
    "ortg_drtg_product",
    "net_rtg_diff",        # home_net_rtg - away_net_rtg
    "ortg_matchup",        # home_ortg vs away_drtg
    "drtg_matchup",        # away_ortg vs home_drtg
    # H2H
    "h2h_avg_total",
    "h2h_q1_avg",
    "h2h_games",
    # Rest / schedule
    "home_rest_days",
    "away_rest_days",
    "rest_diff",           # home_rest - away_rest
    "home_b2b",
    "away_b2b",
    # Injuries
    "home_key_absences",
    "away_key_absences",
    "home_absence_impact",
    "away_absence_impact",
    "net_absence_impact",  # home_absence - away_absence (net pts advantage)
    # Location
    "is_neutral",
    # Recent form — rolling 10 games
    "home_last10_ortg",
    "away_last10_drtg",
    "home_form_delta",     # last10_ortg - ortg (momentum indicator)
    "away_form_delta",
    "h2h_vs_season_delta", # h2h_avg_total - season_avg_total
]

# Approximate season avg total for normalisation fallback
SEASON_AVG_TOTAL = 220.0


class FeatureEngineer:
    """
    Converts pre-game snapshots into a flat feature dict.

    The feature set is designed to avoid any look-ahead — all inputs
    are derived from rolling windows computed BEFORE the game.
    """

    def __init__(self, league_avg_ortg: float = config.LEAGUE_AVG_ORTG) -> None:
        self.league_avg_ortg = league_avg_ortg

    def build_features(self, snapshot: dict) -> dict[str, float]:
        """
        Build a feature dict from a pre-game snapshot.

        Args:
            snapshot: dict from SnapshotBuilder.build_pre_game_snapshot()

        Returns:
            Feature dict with stable column order (keys from FEATURE_COLUMNS)
        """
        home_ortg = float(snapshot.get("home_ortg") or 110.0)
        home_drtg = float(snapshot.get("home_drtg") or 110.0)
        home_pace = float(snapshot.get("home_pace") or 98.0)
        home_net_rtg = float(snapshot.get("home_net_rtg") or 0.0)
        home_last10_ortg = float(snapshot.get("home_last10_ortg") or home_ortg)

        away_ortg = float(snapshot.get("away_ortg") or 110.0)
        away_drtg = float(snapshot.get("away_drtg") or 110.0)
        away_pace = float(snapshot.get("away_pace") or 98.0)
        away_net_rtg = float(snapshot.get("away_net_rtg") or 0.0)
        away_last10_drtg = float(snapshot.get("away_last10_drtg") or away_drtg)

        h2h_avg_total = float(snapshot.get("h2h_avg_total") or SEASON_AVG_TOTAL)
        h2h_q1_avg = float(snapshot.get("h2h_q1_avg") or 55.0)
        h2h_games = int(snapshot.get("h2h_games") or 0)

        home_rest = int(snapshot.get("home_rest_days") or 2)
        away_rest = int(snapshot.get("away_rest_days") or 2)

        home_b2b = int(bool(snapshot.get("home_b2b", 0)))
        away_b2b = int(bool(snapshot.get("away_b2b", 0)))

        home_key_absences = int(snapshot.get("home_key_absences") or 0)
        away_key_absences = int(snapshot.get("away_key_absences") or 0)
        home_absence_impact = float(snapshot.get("home_absence_impact") or 0.0)
        away_absence_impact = float(snapshot.get("away_absence_impact") or 0.0)

        is_neutral = int(bool(snapshot.get("is_neutral", 0)))

        # Derived features
        pace_avg = (home_pace + away_pace) / 2.0

        # Projected total component: how well do these offenses match the defenses?
        ortg_drtg_product = (
            (home_ortg * away_drtg / self.league_avg_ortg)
            + (away_ortg * home_drtg / self.league_avg_ortg)
        ) / 2.0

        net_rtg_diff = home_net_rtg - away_net_rtg
        ortg_matchup = home_ortg - away_drtg   # positive = home offense advantage
        drtg_matchup = away_ortg - home_drtg   # positive = away offense advantage

        rest_diff = home_rest - away_rest
        net_absence_impact = home_absence_impact - away_absence_impact

        home_form_delta = home_last10_ortg - home_ortg
        away_form_delta = away_last10_drtg - away_drtg

        h2h_vs_season_delta = h2h_avg_total - SEASON_AVG_TOTAL

        features: dict[str, float] = {
            "home_ortg": home_ortg,
            "home_drtg": home_drtg,
            "home_pace": home_pace,
            "home_net_rtg": home_net_rtg,
            "away_ortg": away_ortg,
            "away_drtg": away_drtg,
            "away_pace": away_pace,
            "away_net_rtg": away_net_rtg,
            "pace_avg": round(pace_avg, 3),
            "ortg_drtg_product": round(ortg_drtg_product, 3),
            "net_rtg_diff": round(net_rtg_diff, 3),
            "ortg_matchup": round(ortg_matchup, 3),
            "drtg_matchup": round(drtg_matchup, 3),
            "h2h_avg_total": h2h_avg_total,
            "h2h_q1_avg": h2h_q1_avg,
            "h2h_games": float(h2h_games),
            "home_rest_days": float(min(home_rest, 10)),  # cap at 10 days
            "away_rest_days": float(min(away_rest, 10)),
            "rest_diff": float(rest_diff),
            "home_b2b": float(home_b2b),
            "away_b2b": float(away_b2b),
            "home_key_absences": float(home_key_absences),
            "away_key_absences": float(away_key_absences),
            "home_absence_impact": home_absence_impact,
            "away_absence_impact": away_absence_impact,
            "net_absence_impact": round(net_absence_impact, 3),
            "is_neutral": float(is_neutral),
            "home_last10_ortg": home_last10_ortg,
            "away_last10_drtg": away_last10_drtg,
            "home_form_delta": round(home_form_delta, 3),
            "away_form_delta": round(away_form_delta, 3),
            "h2h_vs_season_delta": round(h2h_vs_season_delta, 3),
        }

        return features

    def features_to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in canonical column order."""
        return np.array(
            [features.get(col, 0.0) for col in FEATURE_COLUMNS], dtype=np.float32
        )

    def build_dataset(
        self,
        snapshots: list[dict],
        games: list[dict],
        target: str = "total",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build a full training dataset.

        Args:
            snapshots: list of pre-game snapshots (from SnapshotBuilder)
            games: list of completed game records (for labels)
            target: which label to extract — 'total' | 'home_margin' | 'q1_total'

        Returns:
            (X, y): feature DataFrame and label Series
        """
        # Build game lookup
        games_by_id = {g["game_id"]: g for g in games}

        rows: list[dict] = []
        labels: list[float] = []

        for snap in snapshots:
            game_id = snap.get("game_id")
            game = games_by_id.get(game_id)
            if not game:
                continue

            # Extract label
            if target == "total":
                label = game.get("total")
            elif target == "home_margin":
                hs = game.get("home_score")
                as_ = game.get("away_score")
                label = (hs - as_) if (hs is not None and as_ is not None) else None
            elif target == "q1_total":
                label = game.get("q1_total")
            else:
                raise ValueError(f"Unknown target: {target}")

            if label is None:
                continue

            features = self.build_features(snap)
            rows.append(features)
            labels.append(float(label))

        X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        y = pd.Series(labels, name=target)

        logger.info(
            "Built dataset: %d samples, %d features, target='%s'",
            len(X),
            len(FEATURE_COLUMNS),
            target,
        )

        return X, y

    def validate_features(self, features: dict) -> list[str]:
        """
        Validate a feature dict for missing or out-of-range values.

        Returns:
            list of warning strings (empty = all good)
        """
        warnings: list[str] = []

        for col in FEATURE_COLUMNS:
            if col not in features:
                warnings.append(f"Missing feature: {col}")
            elif features[col] is None or (
                isinstance(features[col], float) and np.isnan(features[col])
            ):
                warnings.append(f"NaN feature: {col}")

        # Sanity checks
        if features.get("home_ortg", 0) < 80 or features.get("home_ortg", 0) > 135:
            warnings.append(f"Suspicious home_ortg: {features.get('home_ortg')}")
        if features.get("away_ortg", 0) < 80 or features.get("away_ortg", 0) > 135:
            warnings.append(f"Suspicious away_ortg: {features.get('away_ortg')}")

        return warnings

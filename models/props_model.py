"""
Player props prediction model.

Takes player/game context and predicts whether a player will go OVER/UNDER
their DraftKings prop line. Uses rolling stats, opponent defense, home/away,
and injury context (role expansion) as features.
"""
import sys
import time
from pathlib import Path
from datetime import datetime

_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collectors.player_stats import get_player_rolling_stats, get_opponent_def_stats
from config import NBA_API_DELAY, MIN_EDGE_TO_BET, PROP_ROLLING_WINDOW


# Minimum edge (%) to make a recommendation
MIN_CONFIDENCE = 0.52
STRONG_CONFIDENCE = 0.60

# Stat type normalizations
VALID_PROP_TYPES = {"points", "rebounds", "assists", "pra", "PRA"}


class PropsPredictor:
    """
    Predicts whether a player will go OVER or UNDER a prop line.
    
    Features used:
    - Rolling 10-game average (and last-5 trend)
    - Standard deviation (volatility)
    - Opponent defensive rank for the stat type
    - Home/Away edge (~+1.5% boost for home players)
    - Injury context: role expansion multiplier
    - Recent form momentum (last5 vs last10 trend)
    """

    def predict_prop(
        self,
        player: str,
        team: str,
        opponent: str,
        prop_type: str,
        line: float,
        date: str,
        is_home: bool = None,
        injury_context: dict = None,
    ) -> dict:
        """
        Predict a player prop.
        
        Args:
            player: Player full name (e.g. "Nique Clifford")
            team: Player's team tricode (e.g. "PHX")
            opponent: Opponent tricode (e.g. "SAC")
            prop_type: "points", "rebounds", "assists", or "PRA"
            line: The prop line (e.g. 19.5)
            date: Game date string (e.g. "2026-03-04")
            is_home: True if player's team is home (None = unknown)
            injury_context: {
                "key_bigs_out": int,   # center/PF absences on player's team
                "key_bigs_out_opp": int,
                "usage_bump": float,   # direct usage increase estimate (0-1)
                "notes": str,
            }
        
        Returns:
            {
                predicted: float,
                confidence: float,   # 0.5 = coin flip, 1.0 = certain
                recommendation: str, # "OVER", "UNDER", or "PASS"
                edge: float,         # predicted - line (positive = lean over)
                reasoning: str,
                features: dict,      # raw inputs used
            }
        """
        prop_type_norm = prop_type.lower() if prop_type != "PRA" else "pra"
        if prop_type_norm not in {p.lower() for p in VALID_PROP_TYPES}:
            return self._error_result(f"Unknown prop type: {prop_type}")

        # 1. Get rolling player stats
        rolling = get_player_rolling_stats(player, n_games=PROP_ROLLING_WINDOW)
        if "error" in rolling:
            return self._error_result(f"Could not get stats for {player}: {rolling['error']}")

        # 2. Get opponent defensive stats
        time.sleep(NBA_API_DELAY)
        def_stats = get_opponent_def_stats(opponent, prop_type)
        if "error" in def_stats:
            # Don't fail — just use neutral
            def_stats = {
                "rank": 15, "opp_val_allowed": 0, "is_weak_defense": False,
                "is_strong_defense": False, "def_rating": 110,
            }

        # 3. Pull the right stat averages
        avg_val, last5_val, std_val = self._get_stat_values(rolling, prop_type_norm)

        if avg_val == 0:
            return self._error_result(f"No {prop_type} data for {player}")

        # 4. Build prediction
        predicted = avg_val
        reasoning_parts = []
        adjustments = []

        # --- Feature: Recent trend (last5 vs rolling avg) ---
        trend = last5_val - avg_val
        trend_pct = trend / avg_val if avg_val > 0 else 0
        if abs(trend_pct) >= 0.1:
            # Blend in 30% weight to recent trend
            predicted = predicted + (trend * 0.3)
            direction = "↑ hot" if trend > 0 else "↓ cold"
            adjustments.append(f"trend {direction} ({trend:+.1f} vs 10-game avg)")

        # --- Feature: Opponent defense ---
        def_rank = def_stats.get("rank", 15)
        if def_stats.get("is_weak_defense"):
            # Weak defense: boost prediction by up to 8%
            def_boost = avg_val * 0.08 * ((def_rank - 20) / 10)
            predicted += def_boost
            adjustments.append(f"vs weak {opponent} defense (#{def_rank} worst)")
        elif def_stats.get("is_strong_defense"):
            # Strong defense: reduce prediction by up to 8%
            def_cut = avg_val * 0.08 * ((11 - def_rank) / 10)
            predicted -= def_cut
            adjustments.append(f"vs strong {opponent} defense (#{def_rank} best)")

        # --- Feature: Home/away ---
        home_boost = 0.0
        if is_home is True:
            home_boost = avg_val * 0.025  # ~2.5% home boost
            predicted += home_boost
            adjustments.append("home court (+2.5%)")
        elif is_home is False:
            predicted -= avg_val * 0.015
            adjustments.append("road game (-1.5%)")

        # --- Feature: Injury context / role expansion ---
        injury_boost = 0.0
        if injury_context:
            bigs_out = injury_context.get("key_bigs_out", 0)
            usage_bump = injury_context.get("usage_bump", 0.0)
            notes = injury_context.get("notes", "")

            if usage_bump > 0:
                injury_boost = avg_val * usage_bump
                predicted += injury_boost
                adjustments.append(f"role expansion (+{usage_bump*100:.0f}% usage bump: {notes})")
            elif bigs_out > 0 and prop_type_norm in ("rebounds", "pra"):
                # Rough: each missing big → +1.5 boards/PRA for next center
                board_boost = bigs_out * 1.5
                predicted += board_boost
                injury_boost = board_boost
                adjustments.append(f"{bigs_out} key big(s) out → +{board_boost:.1f} {prop_type} opportunity")

        # 5. Compute edge and confidence
        edge = predicted - line

        # Confidence model: based on edge relative to std dev
        if std_val > 0:
            # Normalized edge: how many std devs above/below the line
            z = edge / std_val
            # Map to 0.5-0.85 confidence range via sigmoid-ish
            import math
            raw_conf = 0.5 + (0.35 * (2 / (1 + math.exp(-z * 0.8)) - 1))
        else:
            raw_conf = 0.5 + min(abs(edge) / (line * 0.4 + 0.01), 0.3) * (1 if edge > 0 else -1)

        # Bump confidence for supporting signals
        conf_bonus = 0.0
        if def_stats.get("is_weak_defense") and edge > 0:
            conf_bonus += 0.03
        if injury_boost > 0 and edge > 0:
            conf_bonus += 0.04
        if rolling.get("hit_rate", 0) >= 0.7:
            conf_bonus += 0.02
            adjustments.append(f"high historical hit rate ({rolling['hit_rate']*100:.0f}%)")

        confidence = min(max(raw_conf + conf_bonus, 0.45), 0.85)

        # 6. Recommendation
        if confidence >= MIN_CONFIDENCE and edge > 0:
            rec = "OVER"
        elif confidence >= MIN_CONFIDENCE and edge < 0:
            rec = "UNDER"
        else:
            rec = "PASS"

        # Build reasoning string
        reasoning_parts.append(f"{player} averages {avg_val:.1f} {prop_type} (rolling {rolling['n_games']}G)")
        reasoning_parts.append(f"last 5 avg: {last5_val:.1f}")
        if adjustments:
            reasoning_parts.append("Adjustments: " + "; ".join(adjustments))
        reasoning_parts.append(f"Predicted: {predicted:.1f} vs line {line} → edge {edge:+.1f}")

        return {
            "player": player,
            "team": team,
            "opponent": opponent,
            "prop_type": prop_type,
            "line": line,
            "predicted": round(predicted, 2),
            "edge": round(edge, 2),
            "confidence": round(confidence, 3),
            "recommendation": rec,
            "reasoning": " | ".join(reasoning_parts),
            "features": {
                "rolling_avg": avg_val,
                "last5_avg": last5_val,
                "std_dev": std_val,
                "trend": round(trend, 2),
                "def_rank": def_rank,
                "is_weak_def": def_stats.get("is_weak_defense", False),
                "injury_boost": round(injury_boost, 2),
                "home_boost": round(home_boost, 2),
                "n_games": rolling.get("n_games", 0),
                "hit_rate": rolling.get("hit_rate", 0),
            },
        }

    def _get_stat_values(self, rolling: dict, prop_type: str) -> tuple[float, float, float]:
        """Extract avg, last5, std for the requested prop type."""
        if prop_type == "pra":
            return (
                rolling.get("pra_avg", 0),
                rolling.get("pra_last5", 0),
                rolling.get("pra_std", 0),
            )
        elif prop_type == "points":
            return (
                rolling.get("pts_avg", 0),
                rolling.get("pts_last5", 0),
                rolling.get("pts_std", 0),
            )
        elif prop_type == "rebounds":
            return (
                rolling.get("reb_avg", 0),
                rolling.get("reb_last5", 0),
                rolling.get("reb_std", 0),
            )
        elif prop_type == "assists":
            return (
                rolling.get("ast_avg", 0),
                rolling.get("ast_last5", 0),
                rolling.get("ast_std", 0),
            )
        return (0, 0, 1)

    def _error_result(self, msg: str) -> dict:
        return {
            "predicted": 0,
            "confidence": 0,
            "recommendation": "PASS",
            "edge": 0,
            "reasoning": f"ERROR: {msg}",
            "error": msg,
            "features": {},
        }


if __name__ == "__main__":
    predictor = PropsPredictor()
    
    # Test: Clifford PRA
    print("\n🔮 Testing PropsPredictor...")
    print("\n--- Nique Clifford PRA 19.5 ---")
    result = predictor.predict_prop(
        player="Nique Clifford",
        team="PHX",
        opponent="SAC",
        prop_type="PRA",
        line=19.5,
        date="2026-03-04",
        is_home=False,
    )
    print(f"  Predicted: {result['predicted']}")
    print(f"  Edge: {result['edge']:+.2f}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Reasoning: {result['reasoning']}")
    
    print("\n--- Maxime Raynaud rebounds 9.5 (Sabonis out) ---")
    result2 = predictor.predict_prop(
        player="Maxime Raynaud",
        team="SAC",
        opponent="PHX",
        prop_type="rebounds",
        line=9.5,
        date="2026-03-04",
        is_home=True,
        injury_context={
            "key_bigs_out": 2,
            "notes": "Sabonis + Cardwell both out",
        },
    )
    print(f"  Predicted: {result2['predicted']}")
    print(f"  Edge: {result2['edge']:+.2f}")
    print(f"  Confidence: {result2['confidence']*100:.1f}%")
    print(f"  Recommendation: {result2['recommendation']}")
    print(f"  Reasoning: {result2['reasoning']}")

"""
Dean Oliver's Four Factors — NBA's most validated predictive framework.

Four Factors explain ~75% of NBA wins and are our biggest model gap.
Adding these as features should meaningfully improve MAE on totals.

The Four Factors:
1. eFG% (Effective Field Goal %) — shooting efficiency accounting for 3-pointers
   eFG% = (FGM + 0.5 × 3PM) / FGA
   Weight in wins: ~40%

2. TOV% (Turnover Rate) — how often possessions end in turnovers
   TOV% = TOV / (FGA + 0.44×FTA + TOV)
   Weight in wins: ~25%

3. ORB% (Offensive Rebounding Rate) — extra possession rate
   ORB% = ORB / (ORB + opp_DRB)
   Weight in wins: ~20%

4. FT Rate (Free Throw Rate) — free throw generation
   FT Rate = FTA / FGA
   Weight in wins: ~15%

Plus: Pace (possessions per game), Net Rating (ORTG - DRTG)
These combine with Four Factors for possessions-based total projection.

Usage:
    from analyzers.four_factors import FourFactors, project_total_possessions

    ff = FourFactors()
    home_ff = ff.get_team_factors("LAC", window=15)
    away_ff = ff.get_team_factors("IND", window=15)
    projected_total = ff.project_game_total("LAC", "IND")
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "nba_betting.db"
DATA_DIR = Path(__file__).parent.parent / "data"


def compute_efg(fgm: float, fg3m: float, fga: float) -> Optional[float]:
    """Effective Field Goal % = (FGM + 0.5 × 3PM) / FGA"""
    if fga <= 0:
        return None
    return (fgm + 0.5 * fg3m) / fga


def compute_tov_rate(tov: float, fga: float, fta: float) -> Optional[float]:
    """Turnover Rate = TOV / (FGA + 0.44×FTA + TOV)"""
    denom = fga + 0.44 * fta + tov
    if denom <= 0:
        return None
    return tov / denom


def compute_orb_rate(orb: float, opp_drb: float) -> Optional[float]:
    """Offensive Rebound Rate = ORB / (ORB + opponent DRB)"""
    denom = orb + opp_drb
    if denom <= 0:
        return None
    return orb / denom


def compute_ft_rate(fta: float, fga: float) -> Optional[float]:
    """Free Throw Rate = FTA / FGA"""
    if fga <= 0:
        return None
    return fta / fga


def estimate_possessions(fga: float, orb: float, tov: float, fta: float) -> float:
    """
    Estimate possessions using the standard formula:
    Poss ≈ FGA + 0.44×FTA − ORB + TOV
    """
    return fga + 0.44 * fta - orb + tov


def compute_game_pace(
    home_fga: float, home_orb: float, home_tov: float, home_fta: float,
    away_fga: float, away_orb: float, away_tov: float, away_fta: float,
) -> float:
    """
    Game pace = average possessions per team.
    Uses symmetric average of both teams' possession estimates.
    """
    home_poss = estimate_possessions(home_fga, home_orb, home_tov, home_fta)
    away_poss = estimate_possessions(away_fga, away_orb, away_tov, away_fta)
    return 0.5 * (home_poss + away_poss)


class FourFactors:
    """
    Compute and retrieve Four Factors stats for NBA teams.
    Uses the SQLite games database as source.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

    def _get_conn(self) -> Optional[sqlite3.Connection]:
        if not self.db_path.exists():
            log.warning(f"DB not found at {self.db_path}")
            return None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_team_factors(
        self,
        team: str,
        as_of_date: Optional[str] = None,
        window: int = 15,
    ) -> Dict[str, Any]:
        """
        Get Four Factors + pace + efficiency for a team over the last `window` games.

        Returns dict with:
        - efg_pct: effective field goal %
        - tov_rate: turnover rate
        - orb_rate: offensive rebound rate
        - ft_rate: free throw rate
        - pace: estimated possessions per game
        - ortg: offensive rating (pts per 100 possessions)
        - drtg: defensive rating (pts allowed per 100 possessions)
        - net_rtg: ortg - drtg
        - pts_per_game: average points
        - opp_pts_per_game: average points allowed
        """
        conn = self._get_conn()
        if not conn:
            return self._default_factors(team)

        try:
            date_filter = f"AND date < '{as_of_date}'" if as_of_date else ""

            # Pull recent games where this team played (home or away)
            query = f"""
                SELECT
                    CASE WHEN home = ? THEN home_score ELSE away_score END as pts_for,
                    CASE WHEN home = ? THEN away_score ELSE home_score END as pts_against,
                    home, away, home_score, away_score, date
                FROM games
                WHERE (home = ? OR away = ?)
                {date_filter}
                ORDER BY date DESC
                LIMIT {window}
            """
            rows = conn.execute(query, [team, team, team, team]).fetchall()

            if not rows:
                return self._default_factors(team)

            # Compute averages
            pts_for = [r["pts_for"] for r in rows if r["pts_for"]]
            pts_against = [r["pts_against"] for r in rows if r["pts_against"]]

            avg_pts = sum(pts_for) / len(pts_for) if pts_for else 105.0
            avg_opp_pts = sum(pts_against) / len(pts_against) if pts_against else 108.0

            # Estimate pace from scoring (proxy: assume ~200 possessions per game → poss ≈ total_pts / 2 × scaling)
            # Better: use advanced stats if available. For now, proxy from pts.
            avg_total = avg_pts + avg_opp_pts
            # NBA average: ~200 possessions, ~220 pts total → ~1.1 pts/poss
            # pace proxy = total_pts / 1.1
            pace_proxy = avg_total / 1.1

            # ORTG = pts / possessions × 100
            ortg = (avg_pts / (pace_proxy / 2)) * 100 if pace_proxy > 0 else 108.0
            drtg = (avg_opp_pts / (pace_proxy / 2)) * 100 if pace_proxy > 0 else 108.0
            net_rtg = ortg - drtg

            # Without raw box scores for FGM/FGA/etc., we estimate Four Factors from aggregates
            # These are approximations — they'll be calibrated better with 15yr training data
            # Source: league avg eFG% ~0.53, TOV% ~0.13, ORB% ~0.28, FT Rate ~0.24
            league_avg_efg = 0.530
            league_avg_tov = 0.130
            league_avg_orb = 0.280
            league_avg_ftr = 0.240

            # Estimate deviations from scoring patterns
            # High-scoring teams tend to have higher eFG% and lower TOV%
            pts_rel = (avg_pts - 108.0) / 15.0  # relative to league avg

            efg_pct = min(0.60, max(0.48, league_avg_efg + pts_rel * 0.015))
            tov_rate = min(0.17, max(0.09, league_avg_tov - pts_rel * 0.005))
            orb_rate = league_avg_orb  # hard to estimate without box scores
            ft_rate = min(0.32, max(0.15, league_avg_ftr + pts_rel * 0.01))

            return {
                "team": team,
                "games_sampled": len(rows),
                "window": window,
                "pts_per_game": round(avg_pts, 1),
                "opp_pts_per_game": round(avg_opp_pts, 1),
                "pace": round(pace_proxy, 1),
                "ortg": round(ortg, 1),
                "drtg": round(drtg, 1),
                "net_rtg": round(net_rtg, 1),
                "efg_pct": round(efg_pct, 3),
                "tov_rate": round(tov_rate, 3),
                "orb_rate": round(orb_rate, 3),
                "ft_rate": round(ft_rate, 3),
                "data_quality": "estimated",  # upgrade when we have full box scores
            }

        except Exception as e:
            log.error(f"Four Factors query failed for {team}: {e}")
            return self._default_factors(team)
        finally:
            conn.close()

    def _default_factors(self, team: str) -> Dict[str, Any]:
        """League average defaults when data is unavailable."""
        return {
            "team": team,
            "games_sampled": 0,
            "window": 0,
            "pts_per_game": 108.0,
            "opp_pts_per_game": 108.0,
            "pace": 98.0,
            "ortg": 110.0,
            "drtg": 110.0,
            "net_rtg": 0.0,
            "efg_pct": 0.530,
            "tov_rate": 0.130,
            "orb_rate": 0.280,
            "ft_rate": 0.240,
            "data_quality": "default",
        }

    def project_game_total(
        self,
        home: str,
        away: str,
        as_of_date: Optional[str] = None,
        home_injury_adj: float = 0.0,  # injury adjustment in pts
        away_injury_adj: float = 0.0,
        rest_adj_home: float = 0.0,    # rest adjustment in possessions
        rest_adj_away: float = 0.0,
        home_court_advantage: float = 1.8,  # home court adds ~1.8 pts avg
    ) -> Dict[str, Any]:
        """
        Project game total using possessions × efficiency.

        Core formula (Dean Oliver / UnderdogChance):
        Poss_game = 0.5 × (pace_home + pace_away) + rest_adj
        Pts_home = Poss_game × (ortg_home / 100) + home_court_advantage + injury_adj
        Pts_away = Poss_game × (ortg_away / 100) + injury_adj
        Total = Pts_home + Pts_away
        """
        home_ff = self.get_team_factors(home, as_of_date=as_of_date)
        away_ff = self.get_team_factors(away, as_of_date=as_of_date)

        # Expected possessions (blend paces)
        # pace stored as TOTAL game pace (both teams combined), so /2 = per-team pace
        pace_home = home_ff["pace"]
        pace_away = away_ff["pace"]
        per_team_pace_home = pace_home / 2
        per_team_pace_away = pace_away / 2
        # Average per-team possessions for this matchup
        poss_expected = 0.5 * (per_team_pace_home + per_team_pace_away) + rest_adj_home + rest_adj_away

        # Matchup-adjusted ORTG: home offense vs away defense, away offense vs home defense
        # Adj = (team_ortg × opp_drtg / league_avg_drtg)
        league_avg_rating = 112.0
        home_ortg_adj = (home_ff["ortg"] * away_ff["drtg"]) / league_avg_rating
        away_ortg_adj = (away_ff["ortg"] * home_ff["drtg"]) / league_avg_rating

        # Project points
        pts_home = (poss_expected / 100) * home_ortg_adj + home_court_advantage + home_injury_adj
        pts_away = (poss_expected / 100) * away_ortg_adj + away_injury_adj
        projected_total = pts_home + pts_away
        projected_spread = pts_home - pts_away

        return {
            "home": home,
            "away": away,
            "projected_total": round(projected_total, 1),
            "projected_spread": round(projected_spread, 1),
            "pts_home": round(pts_home, 1),
            "pts_away": round(pts_away, 1),
            "expected_possessions": round(poss_expected, 1),
            "home_ortg_adj": round(home_ortg_adj, 1),
            "away_ortg_adj": round(away_ortg_adj, 1),
            "home_pace": pace_home,
            "away_pace": pace_away,
            "home_net_rtg": home_ff["net_rtg"],
            "away_net_rtg": away_ff["net_rtg"],
            "home_efg": home_ff["efg_pct"],
            "away_efg": away_ff["efg_pct"],
            "method": "possessions_x_efficiency",
        }


def compute_matchup_four_factors(
    home: str,
    away: str,
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute the four-factor matchup differential for a game.
    Returns features ready for ML model input.
    """
    ff = FourFactors()
    home_ff = ff.get_team_factors(home, as_of_date=as_of_date)
    away_ff = ff.get_team_factors(away, as_of_date=as_of_date)

    return {
        # Individual factors
        "home_efg": home_ff["efg_pct"],
        "away_efg": away_ff["efg_pct"],
        "efg_diff": round(home_ff["efg_pct"] - away_ff["efg_pct"], 3),
        "home_tov_rate": home_ff["tov_rate"],
        "away_tov_rate": away_ff["tov_rate"],
        "tov_diff": round(home_ff["tov_rate"] - away_ff["tov_rate"], 3),
        "home_orb_rate": home_ff["orb_rate"],
        "away_orb_rate": away_ff["orb_rate"],
        "orb_diff": round(home_ff["orb_rate"] - away_ff["orb_rate"], 3),
        "home_ft_rate": home_ff["ft_rate"],
        "away_ft_rate": away_ff["ft_rate"],
        "ft_rate_diff": round(home_ff["ft_rate"] - away_ff["ft_rate"], 3),
        # Pace and efficiency
        "home_pace": home_ff["pace"],
        "away_pace": away_ff["pace"],
        "pace_avg": round((home_ff["pace"] + away_ff["pace"]) / 2, 1),
        "home_ortg": home_ff["ortg"],
        "away_ortg": away_ff["ortg"],
        "home_drtg": home_ff["drtg"],
        "away_drtg": away_ff["drtg"],
        "home_net_rtg": home_ff["net_rtg"],
        "away_net_rtg": away_ff["net_rtg"],
        "net_rtg_diff": round(home_ff["net_rtg"] - away_ff["net_rtg"], 1),
    }


if __name__ == "__main__":
    ff = FourFactors()

    # Test on tonight's game
    print("\n=== IND @ LAC — Four Factors Analysis ===")
    proj = ff.project_game_total("LAC", "IND")
    print(f"Projected total: {proj['projected_total']} pts")
    print(f"Projected spread: LAC {proj['projected_spread']:+.1f}")
    print(f"Expected possessions: {proj['expected_possessions']}")
    print(f"LAC adj ORTG: {proj['home_ortg_adj']}, IND adj ORTG: {proj['away_ortg_adj']}")
    print(f"\nLAC factors:")
    lac_ff = ff.get_team_factors("LAC")
    for k, v in lac_ff.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
    print(f"\nIND factors:")
    ind_ff = ff.get_team_factors("IND")
    for k, v in ind_ff.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")

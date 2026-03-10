"""
simulators/ncaab_monte_carlo.py — NCAAB Tournament Monte Carlo Engine
======================================================================
Equal priority to NBA. Specifically tuned for conference tournament play.

Key differences from NBA engine:
1. Team-level distributions (player logs harder to get for NCAAB)
2. Neutral site removes home court advantage (+3.5 pts for home team)
3. Tournament variance boost: +15% std (higher variance, elimination pressure)
4. B2B/B3B tournament penalty: -4% scoring efficiency per extra game
5. 3PT-reliance flag: teams with high 3PA% face higher variance
6. Quality filter: only output where our P > market implied by >10%

Data sources:
- ncaab_betting.db: ncaab_games table for historical team scoring
- Odds API: basketball_ncaab for DK lines
- ESPN scoreboard: game context, neutral site flag, tournament round
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import sys

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulators.monte_carlo import american_to_implied_prob, remove_vig

logger = logging.getLogger(__name__)

NCAAB_DB_PATH = ROOT / "data" / "ncaab_betting.db"

# Tournament-specific adjustments
NEUTRAL_SITE_HOME_REMOVAL = 3.5   # remove this from "home" team when neutral site
TOURNAMENT_VAR_BOOST = 1.15       # 15% higher std in tournament games
B2B_PENALTY = 0.96                # -4% efficiency for B2B in tournament
B3B_PENALTY = 0.93                # -7% for 3rd game in 3 days
MIN_EDGE_PCT = 10.0               # only show picks with >10% edge
MIN_GAMES_FOR_DIST = 5            # minimum game history for reliable dist
LEAGUE_AVG_NCAAB_PTS = 72.0       # league avg scoring per team per game
LEAGUE_AVG_NCAAB_STD = 10.5       # empirical std for NCAAB team scores

# High 3PT reliance → higher variance (these are estimated values for known teams)
HIGH_3PT_TEAMS = {
    "Gonzaga", "BYU", "Houston", "Kansas State", "Providence", "Marquette",
    "Vermont", "Wright State", "Detroit Mercy",
}

# ESPN team ID → abbreviation (for key tonight teams)
ESPN_ID_TO_ABBR = {
    "2250": "GONZ", "2608": "SCU", "239": "BAY", "9": "AZST",
    "275": "PITT", "24": "STAN", "254": "UTAH", "2132": "CIN",
    "150": "MD", "2483": "ORE", "221": "BYU", "2306": "KSU",
    "164": "NW", "213": "PSU", "2305": "KANS", "66": "VT",
    "97": "WF", "52": "MOND", "71": "HOFS",
}


def get_team_scoring_dist(team_id: str, conn: sqlite3.Connection,
                           n_games: int = 10) -> Optional[Dict]:
    """
    Build scoring distribution for a team from ncaab_games history.
    Returns {mean_pts, std_pts, mean_pts_against, n_games, three_pt_reliant}
    """
    rows = conn.execute("""
        SELECT
            CASE WHEN home_team_id=? THEN home_score ELSE away_score END as pts_for,
            CASE WHEN home_team_id=? THEN away_score ELSE home_score END as pts_against,
            date,
            neutral_site
        FROM ncaab_games
        WHERE (home_team_id=? OR away_team_id=?)
          AND home_score IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
    """, (team_id, team_id, team_id, team_id, n_games)).fetchall()

    if len(rows) < MIN_GAMES_FOR_DIST:
        return None

    pts_for = [r[0] for r in rows if r[0] is not None]
    pts_against = [r[1] for r in rows if r[1] is not None]

    if not pts_for:
        return None

    mean_pts = float(np.mean(pts_for))
    std_pts = float(np.std(pts_for)) if len(pts_for) > 1 else LEAGUE_AVG_NCAAB_STD
    std_pts = max(std_pts, 7.0)  # floor

    mean_against = float(np.mean(pts_against))

    return {
        "mean_pts": mean_pts,
        "std_pts": std_pts,
        "mean_against": mean_against,
        "n_games": len(pts_for),
        "team_id": team_id,
    }


class NCAABGameSimulator:
    """
    Monte Carlo simulator for a single NCAAB tournament game.

    Usage:
        sim = NCAABGameSimulator(
            home_team="Gonzaga", away_team="Santa Clara",
            home_id="2250", away_id="2608",
            market_total=163.5, market_spread=-6.5,
            neutral_site=True, tournament_round="Final"
        )
        sim.load_distributions()
        sim.set_b2b(home_b2b=False, away_b2b=True)
        results = sim.run(n_sims=1_000_000)
        picks = sim.arbitrage_scan(market_ml_home=-115*6, market_ml_away=105*6)
    """

    def __init__(self, home_team: str, away_team: str,
                 home_id: str = None, away_id: str = None,
                 market_total: float = None, market_spread: float = None,
                 market_ml_home: int = None, market_ml_away: int = None,
                 neutral_site: bool = True,
                 tournament_round: str = "",
                 conference: str = "",
                 n_sims: int = 1_000_000):
        self.home_team = home_team
        self.away_team = away_team
        self.home_id = home_id
        self.away_id = away_id
        self.market_total = market_total
        self.market_spread = market_spread
        self.market_ml_home = market_ml_home
        self.market_ml_away = market_ml_away
        self.neutral_site = neutral_site
        self.tournament_round = tournament_round
        self.conference = conference
        self.n_sims = n_sims

        # Distributions (loaded from DB)
        self.home_dist: Optional[Dict] = None
        self.away_dist: Optional[Dict] = None

        # Context
        self.home_b2b = False
        self.away_b2b = False
        self.home_b3b = False
        self.away_b3b = False
        self.home_out: List[str] = []
        self.away_out: List[str] = []

        # Results
        self.home_scores: Optional[np.ndarray] = None
        self.away_scores: Optional[np.ndarray] = None
        self.totals: Optional[np.ndarray] = None
        self.margins: Optional[np.ndarray] = None

    def load_distributions(self, conn: sqlite3.Connection = None,
                           db_path: Path = None):
        """Load team scoring distributions from NCAAB DB."""
        if conn is None:
            conn = sqlite3.connect(db_path or NCAAB_DB_PATH)
            close_it = True
        else:
            close_it = False

        if self.home_id:
            self.home_dist = get_team_scoring_dist(self.home_id, conn)
        if self.away_id:
            self.away_dist = get_team_scoring_dist(self.away_id, conn)

        if close_it:
            conn.close()

        if not self.home_dist:
            logger.warning(f"No DB data for {self.home_team} — using league avg")
        if not self.away_dist:
            logger.warning(f"No DB data for {self.away_team} — using league avg")

    def set_rest(self, home_b2b=False, away_b2b=False,
                 home_b3b=False, away_b3b=False):
        self.home_b2b = home_b2b
        self.away_b2b = away_b2b
        self.home_b3b = home_b3b
        self.away_b3b = away_b3b

    def set_injuries(self, home_out=None, away_out=None):
        self.home_out = home_out or []
        self.away_out = away_out or []

    def _build_team_params(self, team_name: str, dist: Optional[Dict],
                            opp_dist: Optional[Dict],
                            is_home: bool, is_b2b: bool, is_b3b: bool,
                            out_players: List[str]) -> Tuple[float, float]:
        """Build (mean, std) for a team's scoring in this matchup."""
        if dist:
            base_pts = dist["mean_pts"]
            base_std = dist["std_pts"]
            opp_def = dist.get("mean_against", LEAGUE_AVG_NCAAB_PTS)
        else:
            base_pts = LEAGUE_AVG_NCAAB_PTS
            base_std = LEAGUE_AVG_NCAAB_STD
            opp_def = LEAGUE_AVG_NCAAB_PTS

        # Opponent defensive adjustment
        # Use opponent's mean_against vs league avg
        if opp_dist:
            opp_def_quality = opp_dist["mean_against"]  # how much opp gives up
            league_avg_def = LEAGUE_AVG_NCAAB_PTS
            def_adj = (opp_def_quality - league_avg_def) * 0.40
        else:
            def_adj = 0.0

        adj_pts = base_pts + def_adj

        # Neutral site: remove home court advantage if applicable
        if is_home and self.neutral_site:
            adj_pts -= NEUTRAL_SITE_HOME_REMOVAL
        elif is_home and not self.neutral_site:
            adj_pts += 3.0  # non-neutral home court

        # B2B/B3B rest penalty
        if is_b3b:
            adj_pts *= B3B_PENALTY
        elif is_b2b:
            adj_pts *= B2B_PENALTY

        # Out player penalty (estimate ~5-8 pts for typical starter)
        if out_players:
            adj_pts -= len(out_players) * 6.0  # ~6 pts per OUT starter

        # Tournament variance boost
        std = base_std * TOURNAMENT_VAR_BOOST

        # High-3PT teams: additional variance
        if team_name in HIGH_3PT_TEAMS:
            std *= 1.08

        std = max(std, 7.0)

        return float(adj_pts), float(std)

    def run(self, n_sims: int = None) -> Dict[str, Any]:
        """Run 1M vectorized simulations."""
        if n_sims is None:
            n_sims = self.n_sims

        if self.home_dist is None and self.away_dist is None:
            self.load_distributions()

        home_mean, home_std = self._build_team_params(
            self.home_team, self.home_dist, self.away_dist,
            is_home=True, is_b2b=self.home_b2b, is_b3b=self.home_b3b,
            out_players=self.home_out
        )
        away_mean, away_std = self._build_team_params(
            self.away_team, self.away_dist, self.home_dist,
            is_home=False, is_b2b=self.away_b2b, is_b3b=self.away_b3b,
            out_players=self.away_out
        )

        logger.info(
            f"  NCAAB: {self.away_team} @ {self.home_team} "
            f"({'neutral' if self.neutral_site else 'home'}): "
            f"Home μ={home_mean:.1f}±{home_std:.1f} | "
            f"Away μ={away_mean:.1f}±{away_std:.1f} | "
            f"Proj={home_mean+away_mean:.1f}"
        )

        rng = np.random.default_rng()
        home_scores = rng.normal(home_mean, home_std, n_sims)
        away_scores = rng.normal(away_mean, away_std, n_sims)

        home_scores = np.clip(home_scores, 40, 130)
        away_scores = np.clip(away_scores, 40, 130)

        self.home_scores = home_scores
        self.away_scores = away_scores
        self.totals = home_scores + away_scores
        self.margins = home_scores - away_scores

        return {
            "home_mean": home_mean,
            "away_mean": away_mean,
            "proj_total": home_mean + away_mean,
            "proj_margin": home_mean - away_mean,
            "n_sims": n_sims,
        }

    def probability_over(self, line: float) -> float:
        return float(np.mean(self.totals > line)) if self.totals is not None else 0.5

    def probability_under(self, line: float) -> float:
        return float(np.mean(self.totals < line)) if self.totals is not None else 0.5

    def probability_home_covers(self, spread: float) -> float:
        if self.margins is None:
            return 0.5
        return float(np.mean(self.margins > -spread))

    def probability_home_wins(self) -> float:
        if self.margins is None:
            return 0.5
        return float(np.mean(self.margins > 0))

    def ev_bet(self, our_prob: float, market_odds: int) -> Dict[str, float]:
        market_implied = american_to_implied_prob(market_odds)
        edge = our_prob - market_implied
        if market_odds > 0:
            decimal_odds = market_odds / 100
        else:
            decimal_odds = 100 / abs(market_odds)
        ev_pct = (our_prob * decimal_odds - (1 - our_prob)) * 100
        kelly = edge / decimal_odds if decimal_odds > 0 else 0.0
        kelly_quarter = max(0.0, kelly * 0.25)
        return {
            "ev_pct": round(ev_pct, 2),
            "kelly_pct": round(kelly_quarter * 100, 2),
            "edge_pct": round(edge * 100, 2),
            "our_prob": round(our_prob * 100, 1),
            "market_implied": round(market_implied * 100, 1),
        }

    def confidence_tier(self, our_prob: float) -> str:
        if our_prob >= 0.75:
            return "HIGH"
        elif our_prob >= 0.65:
            return "MEDIUM"
        elif our_prob >= 0.55:
            return "LOW"
        else:
            return "NO EDGE"

    def tournament_flags(self) -> List[str]:
        """Generate tournament-specific risk flags."""
        flags = []
        if self.neutral_site:
            flags.append("NEUTRAL SITE")
        if self.home_b2b or self.away_b2b:
            flags.append(f"B2B: {'HOME' if self.home_b2b else ''} {'AWAY' if self.away_b2b else ''}".strip())
        if self.home_b3b or self.away_b3b:
            flags.append(f"B3B: {'HOME' if self.home_b3b else ''} {'AWAY' if self.away_b3b else ''}".strip())
        if self.home_team in HIGH_3PT_TEAMS:
            flags.append(f"{self.home_team} HIGH 3PT VARIANCE")
        if self.away_team in HIGH_3PT_TEAMS:
            flags.append(f"{self.away_team} HIGH 3PT VARIANCE")
        if self.home_out:
            flags.append(f"HOME OUT: {', '.join(self.home_out)}")
        if self.away_out:
            flags.append(f"AWAY OUT: {', '.join(self.away_out)}")
        return flags

    def arbitrage_scan(self,
                        market_total: float = None,
                        market_spread: float = None,
                        market_ml_home: int = None,
                        market_ml_away: int = None,
                        min_edge_pct: float = MIN_EDGE_PCT) -> List[Dict]:
        """
        Scan all markets for edges >min_edge_pct%.
        Same logic as NBA but with tournament flags attached.
        """
        if self.totals is None:
            raise RuntimeError("Call run() first")

        bets = []
        mt = market_total or self.market_total
        ms = market_spread or self.market_spread
        ml_home = market_ml_home or self.market_ml_home
        ml_away = market_ml_away or self.market_ml_away
        flags = self.tournament_flags()
        game_label = f"{self.away_team} @ {self.home_team}"
        if self.neutral_site:
            game_label += " (neutral)"

        def make_bet(market_name, line_str, our_p, mkt_impl, odds):
            edge = (our_p - mkt_impl) * 100
            if edge >= min_edge_pct:
                ev = self.ev_bet(our_p, odds)
                bets.append({
                    "market": market_name,
                    "line": line_str,
                    "our_prob": our_p,
                    "market_implied": mkt_impl,
                    "edge_pct": edge,
                    "ev_pct": ev["ev_pct"],
                    "kelly_pct": ev["kelly_pct"],
                    "tier": self.confidence_tier(our_p),
                    "market_odds": odds,
                    "flags": flags,
                    "game": game_label,
                    "conference": self.conference,
                    "round": self.tournament_round,
                })

        # Total
        if mt is not None:
            make_bet("TOTAL OVER", f"O {mt}", self.probability_over(mt),
                     american_to_implied_prob(-110), -110)
            make_bet("TOTAL UNDER", f"U {mt}", self.probability_under(mt),
                     american_to_implied_prob(-110), -110)

        # Spread
        if ms is not None:
            home_sp_odds = -115
            away_sp_odds = -105
            make_bet(f"{self.home_team} ATS", f"{ms:+g}",
                     self.probability_home_covers(ms),
                     american_to_implied_prob(home_sp_odds), home_sp_odds)
            make_bet(f"{self.away_team} ATS", f"{-ms:+g}",
                     1 - self.probability_home_covers(ms),
                     american_to_implied_prob(away_sp_odds), away_sp_odds)

        # ML
        if ml_home is not None and ml_away is not None:
            home_win_p = self.probability_home_wins()
            away_win_p = 1 - home_win_p
            h_impl = american_to_implied_prob(ml_home)
            a_impl = american_to_implied_prob(ml_away)
            h_clean, a_clean = remove_vig(h_impl, a_impl)
            make_bet(f"{self.home_team} ML", f"{ml_home:+d}",
                     home_win_p, h_clean, ml_home)
            make_bet(f"{self.away_team} ML", f"{ml_away:+d}",
                     away_win_p, a_clean, ml_away)

        bets.sort(key=lambda x: x["ev_pct"], reverse=True)
        return bets

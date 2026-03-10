"""
simulators/monte_carlo.py — NBA Monte Carlo Simulation Engine v2
================================================================
1,000,000 vectorized numpy simulations per game.

Architecture based on:
  - Demsyn-Jones (2019) "Misadventures in Monte Carlo" — JSA
  - arXiv:2304.09918 "Smart Sports Predictions via Hybrid Simulation"

Core formula (Paper 1):
    team_score_sim = Σ_players(sample_min(p) × sample_ppm(p))
                     × team_persistent_multiplier
                     + home_court_adj

Key design principles:
  1. Minutes as distribution N(mean_min, std_min) — #1 noise source
  2. PPM as separate distribution N(mean_ppm, std_ppm) — propagates correctly
  3. Team persistent pace effect — beyond individual player output
  4. Adaptive 20-game window with roster change detection
  5. Game-state branches for live updates (pre-game: always 'normal')
  6. Pre-game precision: ALL inputs locked before running
     (confirmed starters/injuries, B2B, referee crew, H2H pace)
"""

import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DB_PATH = ROOT / "data" / "nba_betting.db"

# ── Leverage weighting (arXiv:1604.03186) ─────────────────────────────────
# 25 pts in a blowout ≠ 25 pts in a close game.
# Weight each game by how much it reflects true player performance.
def calculate_game_leverage(plus_minus: float) -> float:
    """
    Returns leverage weight based on a player's plus_minus in a game.
    High leverage = close game; low leverage = blowout (garbage time stats inflated).

    Using |plus_minus| as proxy for game margin:
      0-5  pts: 1.30 (very close game, high leverage)
      6-12 pts: 1.10
      13-20 pts: 0.85
      21+  pts: 0.60 (blowout, garbage time)
    """
    abs_pm = abs(float(plus_minus))
    if   abs_pm <= 5:  return 1.30
    elif abs_pm <= 12: return 1.10
    elif abs_pm <= 20: return 0.85
    else:              return 0.60


def leverage_weighted_stats(pts_arr: np.ndarray, mins_arr: np.ndarray,
                              pm_arr: np.ndarray) -> Dict[str, float]:
    """
    Compute leverage-weighted mean and std for pts and PPM.
    Upweights close-game performance, downweights blowout stats.

    Returns dict with mean_pts_lw, std_pts_lw, mean_ppm_lw, std_ppm_lw,
    and a flag if raw_avg >> leverage_avg (potential blowout padding).
    """
    weights = np.array([calculate_game_leverage(pm) for pm in pm_arr])
    weights = weights / weights.sum()  # normalize to sum=1

    # Weighted mean for pts
    mean_pts_lw = float(np.sum(weights * pts_arr))
    std_pts_lw  = float(np.sqrt(np.sum(weights * (pts_arr - mean_pts_lw)**2)))
    std_pts_lw  = max(std_pts_lw, 2.0)

    # PPM: use weighted ratio-of-means
    ppm_arr = pts_arr / np.maximum(mins_arr, 1.0)
    mean_ppm_lw = float(np.sum(weights * ppm_arr))
    std_ppm_lw  = float(np.sqrt(np.sum(weights * (ppm_arr - mean_ppm_lw)**2)))
    std_ppm_lw  = max(std_ppm_lw, mean_ppm_lw * 0.15)

    # Raw unweighted mean for comparison
    mean_pts_raw = float(np.mean(pts_arr))

    # Flag if player is padding in blowouts
    blowout_padding = (mean_pts_raw - mean_pts_lw) > 3.0

    return {
        "mean_pts_lw":   mean_pts_lw,
        "std_pts_lw":    std_pts_lw,
        "mean_ppm_lw":   mean_ppm_lw,
        "std_ppm_lw":    std_ppm_lw,
        "mean_pts_raw":  mean_pts_raw,
        "blowout_padding": blowout_padding,
        "padding_delta": round(mean_pts_raw - mean_pts_lw, 1),
    }

# ── Constants ──────────────────────────────────────────────────────────────
LEAGUE_AVG_PACE       = 100.5   # possessions/48 min, 2025-26
LEAGUE_AVG_TOTAL      = 226.0   # avg game total, 2025-26
HOME_COURT_ADJ        = 3.0     # pts added to home team mean
B2B_PACE_FACTOR       = 0.95    # B2B scoring efficiency
DTD_MISS_PROB         = 0.30    # P(DTD player misses game)

# Minutes modeling
MIN_FLOOR             = 0.0
MIN_CAP               = 48.0
PPM_FLOOR             = 0.0
PPM_CAP               = 3.5     # ~84 pts in 24 min = crazy ceiling

# Team persistent pace
PACE_WEIGHT           = 0.35    # how much team OU rate adjusts output
LEAGUE_OU_RATE        = 0.50    # baseline over rate

# Adaptive window
ADAPTIVE_WINDOW       = 20      # games — optimal per arxiv paper (20-40 range)
NEW_PLAYER_WEIGHT     = 3.0     # 3× weight for recently-added players
MIN_ROSTER_CHANGE_PCT = 0.50    # >50% drop in avg minutes = roster change flag

# Game-state branches (for live mode)
CLOSE_MARGIN_THRESH   = 8.0
LARGE_DEFICIT_THRESH  = 15.0

# Replacement-level scoring (pts/min for a G-League call-up)
REPLACEMENT_PPM       = 0.35    # ~17 pts/48 min replacement level
REPLACEMENT_MIN_FRAC  = 0.85    # backup gets 85% of missing player's minutes

# Minimum team score floor (no NBA team scores <75)
TEAM_SCORE_FLOOR      = 75.0


def american_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(p1: float, p2: float) -> Tuple[float, float]:
    total = p1 + p2
    return p1 / total, p2 / total


# ── Player Distribution Builder ────────────────────────────────────────────

def load_player_distributions(team: str, conn: sqlite3.Connection,
                               window: int = ADAPTIVE_WINDOW) -> List[Dict]:
    """
    Load per-player (mean_min, std_min, mean_ppm, std_ppm) from player_game_logs.

    Adaptive window logic:
    - Use last `window` games by default
    - Roster change detection: if player's last-5 avg_min < 50% of prior-15 → EXCLUDE
    - New player detection: if player appears only in last 10 games → weight recent 3×
    - Returns list of player dicts sorted by avg_min descending
    """
    try:
        rows = conn.execute("""
            SELECT player_name, player_id, date, pts, min, plus_minus
            FROM player_game_logs
            WHERE team = ?
            ORDER BY player_name, date DESC
        """, (team.upper(),)).fetchall()
    except Exception as e:
        logger.warning(f"No player_game_logs for {team}: {e}")
        return []

    # Group by player
    from collections import defaultdict
    player_games: Dict[str, List[Dict]] = defaultdict(list)
    player_ids: Dict[str, str] = {}
    for name, pid, dt, pts, mins, *rest in rows:
        plus_minus = rest[0] if rest else 0.0
        if len(player_games[name]) < window:
            player_games[name].append({
                "date":        dt,
                "pts":         float(pts or 0),
                "min":         float(mins or 0),
                "plus_minus":  float(plus_minus or 0),
            })
        player_ids[name] = pid

    players_out = []
    for name, games in player_games.items():
        if len(games) < 3:
            continue

        # ── Adaptive window: roster change detection ──────────────────
        if len(games) >= 8:
            last5_min  = np.mean([g["min"] for g in games[:5]])
            prior_min  = np.mean([g["min"] for g in games[5:]])
            is_new_player = (len(games) <= 10 and prior_min < 5.0)
            is_losing_role = (prior_min > 10 and last5_min < prior_min * MIN_ROSTER_CHANGE_PCT)

            if is_losing_role:
                logger.debug(f"  {name}: role change detected (last5 {last5_min:.1f} vs prior {prior_min:.1f} min) — downweighted")
                # Still include but weight toward recent (already sorted recent-first)
                games = games[:5]  # only use last 5 games for changed-role players

            if is_new_player:
                logger.debug(f"  {name}: new player — weighting recent 3×")
                # Repeat recent games 3× to upweight
                games = games[:5] * 3 + games[5:]
        else:
            is_losing_role = False
            is_new_player = False

        mins_arr = np.array([g["min"]        for g in games], dtype=float)
        pts_arr  = np.array([g["pts"]        for g in games], dtype=float)
        pm_arr   = np.array([g["plus_minus"] for g in games], dtype=float)

        # Filter out DNPs (min = 0)
        active_mask = mins_arr > 5.0
        if active_mask.sum() < 3:
            continue
        mins_arr = mins_arr[active_mask]
        pts_arr  = pts_arr[active_mask]
        pm_arr   = pm_arr[active_mask]

        # ── Minutes distribution ──────────────────────────────────────
        mean_min = float(np.mean(mins_arr))
        std_min  = float(np.std(mins_arr))
        std_min  = max(std_min, mean_min * 0.15)  # floor: at least 15% relative std
        # Paper: std_min up to ±16 min is observed — allow it, don't cap artificially

        # ── PPM: leverage-weighted (arXiv:1604.03186) ─────────────────
        # Use leverage-weighted stats instead of raw averages.
        # This downweights blowout garbage-time stats that inflate player distributions.
        lw = leverage_weighted_stats(pts_arr, mins_arr, pm_arr)
        mean_ppm = lw["mean_ppm_lw"]
        std_ppm  = lw["std_ppm_lw"]

        # Cross-check: unweighted ratio-of-means for reference
        mean_ppm_raw = float(np.sum(pts_arr) / np.sum(mins_arr))

        if lw["blowout_padding"]:
            logger.debug(
                f"  ⚠️ {name}: blowout padding detected "
                f"(raw {lw['mean_pts_raw']:.1f} vs lw {lw['mean_pts_lw']:.1f} pts, "
                f"Δ={lw['padding_delta']:+.1f})"
            )

        exp_pts = mean_min * mean_ppm

        players_out.append({
            "name":             name,
            "player_id":        player_ids.get(name, ""),
            "mean_min":         mean_min,
            "std_min":          std_min,
            "mean_ppm":         mean_ppm,         # leverage-weighted
            "std_ppm":          std_ppm,           # leverage-weighted
            "mean_ppm_raw":     mean_ppm_raw,      # unweighted (for comparison)
            "exp_pts":          exp_pts,
            "n_games":          int(active_mask.sum()),
            "is_new_player":    is_new_player,
            "role_changed":     is_losing_role,
            "blowout_padding":  lw["blowout_padding"],
            "padding_delta":    lw["padding_delta"],
        })

    # Sort by avg minutes descending (rotation order)
    players_out.sort(key=lambda p: p["mean_min"], reverse=True)
    logger.debug(f"  {team}: {len(players_out)} player distributions built")
    return players_out


def get_team_snapshot(team: str, conn: sqlite3.Connection) -> Optional[Dict]:
    row = conn.execute("""
        SELECT pts_for_avg, pts_against_avg, total_avg, pace_proxy,
               net_rating, last5_pts_for, last5_pts_against, games, win_pct
        FROM team_snapshots WHERE team = ? ORDER BY date DESC LIMIT 1
    """, (team.upper(),)).fetchone()
    if not row:
        return None
    return {
        "pts_for":      row[0],
        "pts_against":  row[1],
        "total_avg":    row[2],
        "pace":         row[3],
        "net_rating":   row[4],
        "last5_for":    row[5],
        "last5_against":row[6],
        "games":        row[7],
        "win_pct":      row[8],
    }


def get_team_pace_signature(team: str, conn: sqlite3.Connection, n_games: int = 40) -> float:
    """
    Team's persistent scoring tendency vs league average.
    Returns a modest multiplier (≈ 0.97–1.03) reflecting whether this team
    consistently scores above/below league average BEYOND what player stats predict.

    Formula: 1.0 + (team_pts_for - LEAGUE_AVG_PTS) / LEAGUE_AVG_PTS * PACE_WEIGHT
    where PACE_WEIGHT = 0.25 (dampened — individual player stats explain most variance)

    Loaded from team_snapshots for stability.
    """
    LEAGUE_AVG_PTS = 114.0
    try:
        row = conn.execute("""
            SELECT pts_for_avg FROM team_snapshots
            WHERE team = ? ORDER BY date DESC LIMIT 1
        """, (team,)).fetchone()
        if not row:
            return 1.0
        pts_for = float(row[0])
        # Modest adjustment: ±3% max for typical team variation
        adj = (pts_for - LEAGUE_AVG_PTS) / LEAGUE_AVG_PTS * PACE_WEIGHT
        return float(np.clip(1.0 + adj, 0.93, 1.07))
    except Exception as e:
        logger.debug(f"Pace signature load failed for {team}: {e}")
        return 1.0


def build_replacement_player(out_player: Dict, available_mins_pool: float) -> Dict:
    """
    Build a synthetic replacement player distribution for a missing player.
    Replacement gets ~85% of the OUT player's minutes at league-minimum PPM.
    """
    rep_min = out_player["mean_min"] * REPLACEMENT_MIN_FRAC
    return {
        "name":      f"[Replacement for {out_player['name']}]",
        "player_id": "replacement",
        "mean_min":  rep_min,
        "std_min":   rep_min * 0.20,        # slightly more uncertain
        "mean_ppm":  REPLACEMENT_PPM,        # ~17 pts/48 min
        "std_ppm":   REPLACEMENT_PPM * 0.25,
        "exp_pts":   rep_min * REPLACEMENT_PPM,
        "n_games":   5,
        "is_new_player": False,
        "role_changed":  False,
    }


# ── Game Simulator ─────────────────────────────────────────────────────────

class GameSimulator:
    """
    Monte Carlo game simulator — player-level decomposition.

    Core formula (Demsyn-Jones 2019):
        team_score = Σ_p(sample_min(p) × sample_ppm(p))
                     × team_persistent_multiplier
                     + home_court_adj

    Usage:
        sim = GameSimulator("BOS", "SAS", market_total=222.5, market_spread=-3.5)
        sim.load_player_distributions(db_conn)
        sim.set_injuries(home_out=["Nikola Vucevic"], away_out=["Harrison Barnes"])
        sim.set_b2b(home_b2b=False, away_b2b=False)
        sim.set_game_state(margin=0, time_remaining=48.0)  # pre-game
        results = sim.run(n_sims=1_000_000)
        picks = sim.arbitrage_scan(...)
    """

    def __init__(self, home_team: str, away_team: str,
                 market_total: float = None, market_spread: float = None,
                 market_ml_home: int = None, market_ml_away: int = None,
                 n_sims: int = 1_000_000):
        self.home = home_team.upper()
        self.away = away_team.upper()
        self.market_total   = market_total
        self.market_spread  = market_spread
        self.market_ml_home = market_ml_home
        self.market_ml_away = market_ml_away
        self.n_sims         = n_sims

        # Player distributions (populated by load_player_distributions)
        self.home_players: List[Dict] = []
        self.away_players: List[Dict] = []
        self.home_snap:    Optional[Dict] = None
        self.away_snap:    Optional[Dict] = None
        self.home_pace_mult: float = 1.0
        self.away_pace_mult: float = 1.0

        # Injury state
        self.home_out:  List[str] = []
        self.home_dtd:  List[str] = []
        self.away_out:  List[str] = []
        self.away_dtd:  List[str] = []

        # Context
        self.home_b2b: bool  = False
        self.away_b2b: bool  = False
        self.referee_total_adj: float = 0.0
        self.referee_home_bias: float = 0.0

        # Game state (pre-game default; updated for live mode)
        self.current_margin:     float = 0.0   # positive = home leading
        self.time_remaining_min: float = 48.0  # full game = 48 min
        self.is_live:            bool  = False

        # Results
        self.home_scores: Optional[np.ndarray] = None
        self.away_scores: Optional[np.ndarray] = None
        self.totals:      Optional[np.ndarray] = None
        self.margins:     Optional[np.ndarray] = None
        self._run_metadata: Dict = {}

    # ── Loaders ────────────────────────────────────────────────────────

    def load_player_distributions(self, conn: sqlite3.Connection = None,
                                   db_path: Path = None):
        """Load all player distributions + team snapshots from DB."""
        close_conn = conn is None
        if conn is None:
            conn = sqlite3.connect(db_path or DB_PATH)

        self.home_snap    = get_team_snapshot(self.home, conn)
        self.away_snap    = get_team_snapshot(self.away, conn)
        self.home_players = load_player_distributions(self.home, conn)
        self.away_players = load_player_distributions(self.away, conn)
        self.home_pace_mult = get_team_pace_signature(self.home, conn)
        self.away_pace_mult = get_team_pace_signature(self.away, conn)

        if close_conn:
            conn.close()

        logger.info(
            f"  Loaded: {self.home}({len(self.home_players)} players, "
            f"pace×{self.home_pace_mult:.3f}) | "
            f"{self.away}({len(self.away_players)} players, "
            f"pace×{self.away_pace_mult:.3f})"
        )

    def set_injuries(self, home_out: List[str] = None, home_dtd: List[str] = None,
                     away_out:  List[str] = None, away_dtd:  List[str] = None):
        self.home_out  = home_out  or []
        self.home_dtd  = home_dtd  or []
        self.away_out  = away_out  or []
        self.away_dtd  = away_dtd  or []

    def set_b2b(self, home_b2b: bool = False, away_b2b: bool = False):
        self.home_b2b = home_b2b
        self.away_b2b = away_b2b

    def set_referee_adj(self, crew_total_adj: float = 0.0, home_bias: float = 0.0):
        self.referee_total_adj = crew_total_adj
        self.referee_home_bias = home_bias

    def set_game_state(self, margin: float = 0.0, time_remaining_min: float = 48.0):
        """
        Set current game state for live simulation.
        margin: positive = home team leading.
        time_remaining_min: minutes of game clock remaining.
        """
        self.current_margin     = margin
        self.time_remaining_min = time_remaining_min
        self.is_live = (time_remaining_min < 47.0)

    # ── Distribution Resolution ────────────────────────────────────────

    def _resolve_active_lineup(self, players: List[Dict], out_names: List[str],
                                dtd_names: List[str]) -> List[Dict]:
        """
        Apply injury list to player roster:
        - OUT players → replaced by synthetic replacement distribution
        - DTD players → included with P(miss) = DTD_MISS_PROB
        Returns list of active + replacement player dicts.
        """
        out_lower = {n.lower() for n in out_names}
        dtd_lower = {n.lower() for n in dtd_names}

        active = []
        replaced_count = 0

        for p in players:
            name_lower = p["name"].lower()
            last_name  = name_lower.split()[-1]

            # Check if OUT
            is_out = (
                name_lower in out_lower or
                last_name in {n.split()[-1].lower() for n in out_names}
            )
            if is_out:
                # Replace with backup-level player
                rep = build_replacement_player(p, 0)
                rep["_replaced_player"] = p["name"]
                active.append(rep)
                replaced_count += 1
                logger.debug(f"    OUT: {p['name']} → replacement ({rep['exp_pts']:.1f} exp pts)")
                continue

            # Check if DTD
            is_dtd = (
                name_lower in dtd_lower or
                last_name in {n.split()[-1].lower() for n in dtd_names}
            )
            if is_dtd:
                # Scale down expected output by DTD_MISS_PROB
                p = p.copy()
                p["mean_min"]  *= (1.0 - DTD_MISS_PROB)
                p["mean_ppm"]  *= (1.0 - DTD_MISS_PROB * 0.15)  # slight productivity hit
                p["_is_dtd"]   = True
                logger.debug(f"    DTD: {p['name']} → {p['mean_min']:.1f} min (reduced)")

            active.append(p)

        return active

    # ── Game-State Branching ───────────────────────────────────────────

    def _game_state_adjustments(self, is_home: bool) -> Tuple[float, float]:
        """
        Apply game-state branching per arxiv paper.
        Returns (std_multiplier, pace_multiplier) for remaining time.

        Pre-game: margin=0, time=48 → (1.0, 1.0)
        Live trailing: increase variance and pace
        Live leading: decrease variance and pace
        """
        if not self.is_live:
            return 1.0, 1.0  # pre-game: normal

        # From perspective of team (home or away)
        if is_home:
            perspective_margin = self.current_margin
        else:
            perspective_margin = -self.current_margin

        # Scale adjustments by time remaining (less impact late in game)
        time_scale = self.time_remaining_min / 48.0

        if abs(perspective_margin) < CLOSE_MARGIN_THRESH:
            # Close game: normal distributions
            return 1.0, 1.0

        elif -LARGE_DEFICIT_THRESH < perspective_margin < -CLOSE_MARGIN_THRESH:
            # Trailing 8-15: high-variance strategy
            std_mult  = 1.0 + 0.20 * time_scale
            pace_mult = 1.0 + 0.10 * time_scale
            return std_mult, pace_mult

        elif perspective_margin < -LARGE_DEFICIT_THRESH:
            # Garbage time if trailing 15+: bench pts rate 30% lower
            std_mult  = 1.0 + 0.10 * time_scale   # still slightly higher variance
            pace_mult = 0.70                         # running clock with bench
            return std_mult, pace_mult

        elif perspective_margin > LARGE_DEFICIT_THRESH:
            # Leading 15+: conservative, slow the game
            std_mult  = 1.0 - 0.15 * time_scale
            pace_mult = 1.0 - 0.08 * time_scale
            return max(0.70, std_mult), max(0.85, pace_mult)

        else:
            # Leading 8-15: slight conservative
            std_mult  = 1.0 - 0.05 * time_scale
            return std_mult, 1.0

    # ── Core Simulation ────────────────────────────────────────────────

    def _simulate_team(self, players: List[Dict], is_home: bool,
                        is_b2b: bool, opp_snap: Optional[Dict],
                        team_pace_mult: float, rng, N: int) -> np.ndarray:
        """
        Simulate N game scores for one team using player-level decomposition.

        Formula: team_score = Σ_p(min_p × ppm_p) × persistent_mult + home_adj

        All operations vectorized over N simulations.
        """
        if not players:
            # No player data → fall back to team snapshot if available
            return self._fallback_team_sim(is_home, is_b2b, opp_snap,
                                            team_pace_mult, rng, N)

        # Game-state adjustments (pre-game: all 1.0)
        gs_std_mult, gs_pace_mult = self._game_state_adjustments(is_home)

        # ── Minutes normalization ─────────────────────────────────────
        # NBA: total team minutes per game = 5 players × 48 min = 240 min.
        # If sum(mean_min) across all rotation players ≠ 240, scale to enforce budget.
        # This prevents inflated totals when roster has 9-10 players all averaging 25+ min.
        TOTAL_MINUTES_BUDGET = 240.0
        raw_min_sum = sum(p["mean_min"] for p in players)
        min_scale = TOTAL_MINUTES_BUDGET / raw_min_sum if raw_min_sum > 0 else 1.0
        # Only apply scaling if sum is meaningfully off (>5% deviation)
        if abs(min_scale - 1.0) < 0.05:
            min_scale = 1.0

        # ── Vectorized player simulation ──────────────────────────────
        # Each player contributes N samples of pts = min × ppm
        # Formula: pts_i = sample_min(i) × sample_ppm(i)  [Paper 1 core decomp]
        team_raw = np.zeros(N)

        for player in players:
            mean_min = player["mean_min"] * min_scale   # normalize to 240-min budget
            std_min  = player["std_min"]  * gs_std_mult
            mean_ppm = player["mean_ppm"]
            std_ppm  = player["std_ppm"]  * gs_std_mult

            # Sample minutes and PPM independently for each of N sims
            sampled_min = rng.normal(mean_min, std_min, N)
            sampled_ppm = rng.normal(mean_ppm, std_ppm, N)

            # Apply floors and caps
            sampled_min = np.clip(sampled_min * gs_pace_mult, MIN_FLOOR, MIN_CAP)
            sampled_ppm = np.clip(sampled_ppm, PPM_FLOOR, PPM_CAP)

            player_pts = sampled_min * sampled_ppm
            team_raw += player_pts

        # ── Opponent defensive adjustment ─────────────────────────────
        # Scales the raw player-sum output to account for tonight's specific opponent.
        # Player logs were built against all opponents; tonight's defense may be better/worse.
        # This is the matchup-specific adjustment BEYOND what historical stats capture.
        if opp_snap:
            opp_pts_allowed = opp_snap.get("pts_against", 114.0)
            opp_l5_allowed  = opp_snap.get("last5_against", opp_pts_allowed)
            opp_def_blended = 0.60 * opp_pts_allowed + 0.40 * opp_l5_allowed
            # Dampened adjustment: 25% pass-through (player logs already partially capture opp quality)
            opp_def_factor = 1.0 + (opp_def_blended - 114.0) / 114.0 * 0.25
        else:
            opp_def_factor = 1.0

        # Matchup pace normalization: harmonic mean of both teams' pace proxies
        team_pace = (self.home_snap or {}).get("pace", LEAGUE_AVG_PACE) if is_home \
                    else (self.away_snap or {}).get("pace", LEAGUE_AVG_PACE)
        opp_pace  = opp_snap.get("pace", LEAGUE_AVG_PACE) if opp_snap else LEAGUE_AVG_PACE
        matchup_pace = 2 * team_pace * opp_pace / (team_pace + opp_pace)
        pace_factor  = matchup_pace / LEAGUE_AVG_PACE  # typically 0.97–1.04

        # Team persistent pace signature (modest: ±3% for typical teams)
        # This captures system effects beyond individual player stats
        persistent_mult = team_pace_mult * opp_def_factor * pace_factor

        team_scores = team_raw * persistent_mult

        # B2B penalty
        if is_b2b:
            team_scores *= B2B_PACE_FACTOR

        # Home court
        if is_home:
            team_scores += HOME_COURT_ADJ

        # Referee adjustments
        team_scores += self.referee_total_adj / 2.0
        if is_home:
            team_scores += self.referee_home_bias

        # Floor
        team_scores = np.maximum(team_scores, TEAM_SCORE_FLOOR)

        return team_scores

    def _fallback_team_sim(self, is_home: bool, is_b2b: bool,
                            opp_snap: Optional[Dict], team_pace_mult: float,
                            rng, N: int) -> np.ndarray:
        """
        Fallback when no player logs available — use team snapshot distribution.
        Less precise but still incorporates pace and opponent adjustments.
        """
        snap = self.home_snap if is_home else self.away_snap
        if not snap:
            mean = 112.0 + (HOME_COURT_ADJ if is_home else 0)
            std  = 13.0
            return np.clip(rng.normal(mean, std, N), TEAM_SCORE_FLOOR, 170)

        season_pts = snap["pts_for"]
        l5_pts = snap.get("last5_for") or season_pts
        base = 0.60 * season_pts + 0.40 * l5_pts

        opp_def = opp_snap.get("pts_against", 114.0) if opp_snap else 114.0
        def_adj = (opp_def - 114.0) * 0.35

        adj = base + def_adj
        adj *= team_pace_mult
        if is_b2b:
            adj *= B2B_PACE_FACTOR
        if is_home:
            adj += HOME_COURT_ADJ
        adj += self.referee_total_adj / 2.0

        std = max(13.0, adj * 0.10)
        return np.clip(rng.normal(adj, std, N), TEAM_SCORE_FLOOR, 170)

    # ── Run ────────────────────────────────────────────────────────────

    def run(self, n_sims: int = None) -> Dict[str, Any]:
        """
        Run N vectorized simulations using player-level decomposition.

        Steps:
        1. Resolve injury-adjusted lineups (OUT → replacement, DTD → scaled)
        2. For each team: sample (min, ppm) per player × N sims
        3. Sum player contributions → team raw score
        4. Apply persistent pace multiplier × matchup pace factor
        5. Apply B2B, home court, referee adjustments
        6. Compute full joint score distribution
        """
        if n_sims is None:
            n_sims = self.n_sims
        N = n_sims

        # Load if not already loaded
        if not self.home_players and not self.away_players:
            self.load_player_distributions()

        # Resolve lineups (remove OUT, scale DTD)
        home_lineup = self._resolve_active_lineup(
            self.home_players, self.home_out, self.home_dtd)
        away_lineup = self._resolve_active_lineup(
            self.away_players, self.away_out, self.away_dtd)

        rng = np.random.default_rng()

        home_scores = self._simulate_team(
            home_lineup, is_home=True, is_b2b=self.home_b2b,
            opp_snap=self.away_snap, team_pace_mult=self.home_pace_mult,
            rng=rng, N=N)

        away_scores = self._simulate_team(
            away_lineup, is_home=False, is_b2b=self.away_b2b,
            opp_snap=self.home_snap, team_pace_mult=self.away_pace_mult,
            rng=rng, N=N)

        self.home_scores = home_scores
        self.away_scores = away_scores
        self.totals      = home_scores + away_scores
        self.margins     = home_scores - away_scores  # positive = home wins

        home_mean = float(np.mean(home_scores))
        away_mean = float(np.mean(away_scores))

        self._run_metadata = {
            "home_mean":     home_mean,
            "away_mean":     away_mean,
            "home_std":      float(np.std(home_scores)),
            "away_std":      float(np.std(away_scores)),
            "proj_total":    home_mean + away_mean,
            "proj_margin":   home_mean - away_mean,
            "home_players":  len(home_lineup),
            "away_players":  len(away_lineup),
            "n_sims":        N,
        }

        logger.info(
            f"  [{self.away} @ {self.home}] "
            f"Home: {home_mean:.1f}±{self._run_metadata['home_std']:.1f} | "
            f"Away: {away_mean:.1f}±{self._run_metadata['away_std']:.1f} | "
            f"Total: {home_mean+away_mean:.1f} | "
            f"Margin: {home_mean-away_mean:+.1f}"
        )

        return self._run_metadata

    # ── Probability Methods ────────────────────────────────────────────

    def probability_over(self, line: float) -> float:
        if self.totals is None: raise RuntimeError("Call run() first")
        return float(np.mean(self.totals > line))

    def probability_under(self, line: float) -> float:
        if self.totals is None: raise RuntimeError("Call run() first")
        return float(np.mean(self.totals < line))

    def probability_home_covers(self, spread: float) -> float:
        """spread < 0 = home favored (e.g., -3.5). Covers if margin > -spread."""
        if self.margins is None: raise RuntimeError("Call run() first")
        return float(np.mean(self.margins > -spread))

    def probability_away_covers(self, spread: float) -> float:
        return 1.0 - self.probability_home_covers(spread)

    def probability_home_wins(self) -> float:
        if self.margins is None: raise RuntimeError("Call run() first")
        return float(np.mean(self.margins > 0))

    def probability_away_wins(self) -> float:
        return 1.0 - self.probability_home_wins()

    def total_distribution(self) -> Dict[str, float]:
        if self.totals is None: return {}
        return {
            "mean": float(np.mean(self.totals)),
            "std":  float(np.std(self.totals)),
            "p10":  float(np.percentile(self.totals, 10)),
            "p25":  float(np.percentile(self.totals, 25)),
            "p50":  float(np.percentile(self.totals, 50)),
            "p75":  float(np.percentile(self.totals, 75)),
            "p90":  float(np.percentile(self.totals, 90)),
        }

    # ── EV / Kelly ─────────────────────────────────────────────────────

    def ev_bet(self, our_prob: float, market_odds: int) -> Dict[str, float]:
        market_implied = american_to_implied_prob(market_odds)
        edge = our_prob - market_implied
        decimal = (market_odds / 100) if market_odds > 0 else (100 / abs(market_odds))
        ev_pct = (our_prob * decimal - (1 - our_prob)) * 100
        kelly  = edge / decimal if decimal > 0 else 0.0
        return {
            "ev_pct":       round(ev_pct, 2),
            "kelly_pct":    round(max(0.0, kelly * 0.25) * 100, 2),  # quarter-Kelly
            "edge_pct":     round(edge * 100, 2),
            "our_prob":     round(our_prob * 100, 1),
            "market_implied": round(market_implied * 100, 1),
        }

    def confidence_tier(self, our_prob: float) -> str:
        if our_prob >= 0.75: return "HIGH"
        if our_prob >= 0.65: return "MEDIUM"
        if our_prob >= 0.55: return "LOW"
        return "NO EDGE"

    # ── Arbitrage Scanner ──────────────────────────────────────────────

    def arbitrage_scan(self,
                        market_total:    float = None,
                        market_spread:   float = None,
                        market_ml_home:  int   = None,
                        market_ml_away:  int   = None,
                        home_spread_odds:int   = -115,
                        away_spread_odds:int   = -105,
                        over_odds:       int   = -110,
                        under_odds:      int   = -110,
                        min_edge_pct:    float = 10.0,
                        line_movements:  Dict  = None,
                        use_calibration: bool  = True) -> List[Dict]:
        """
        Scan all markets for edges > min_edge_pct%.

        Pipeline per market:
          1. Raw MC probability from 1M simulations
          2. Calibrate via isotonic regression (arXiv:2303.06021)
          3. Compare calibrated prob vs vig-removed market implied
          4. Attach line movement signal (SHARP CONFIRM / FADE)
          5. Output if calibrated edge >= min_edge_pct

        Output format per pick:
          raw_prob, calibrated_prob, market_implied, edge_pct,
          ev_pct, kelly_pct, tier, line_movement
        """
        if self.totals is None:
            raise RuntimeError("Call run() before arbitrage_scan()")

        # Lazy-load calibrators (fits on historical data, cached after first load)
        if use_calibration:
            try:
                from analyzers.calibration import calibrate_probability
                _cal = calibrate_probability
            except ImportError:
                logger.warning("Calibration module unavailable — using raw probs")
                _cal = lambda p, mt, **kw: (p, "raw", 0.0)
        else:
            _cal = lambda p, mt, **kw: (p, "raw", 0.0)

        bets = []
        mt  = market_total  or self.market_total
        ms  = market_spread or self.market_spread
        ml_h = market_ml_home or self.market_ml_home
        ml_a = market_ml_away or self.market_ml_away
        lm   = line_movements or {}

        proj_total  = float(np.mean(self.totals))
        proj_margin = float(np.mean(self.margins))

        def add_bet(market_name, line_str, raw_p, odds,
                    cal_market: str, our_side: str = "home",
                    mkt_type: str = "spread"):
            # ── Calibrate ───────────────────────────────────────────
            cal_p, cal_method, correction_pp = _cal(raw_p, cal_market)

            mkt_impl = american_to_implied_prob(odds)
            edge = (cal_p - mkt_impl) * 100

            if edge < min_edge_pct:
                return

            ev  = self.ev_bet(cal_p, odds)
            tier = self.confidence_tier(cal_p)

            # ── Line movement ──────────────────────────────────────
            movement_info = lm.get(market_name, {})
            movement_pts  = movement_info.get("movement")
            movement_label = "⚪ No movement data"
            movement_class = "neutral"
            if movement_pts is not None:
                from collectors.opening_line_logger import classify_movement
                mv = classify_movement(movement_pts, our_side, mkt_type)
                movement_label = mv.get("label", "⚪ Neutral")
                movement_class = mv.get("classification", "neutral")

            logger.info(
                f"    [{market_name}] raw {raw_p*100:.1f}% → "
                f"cal {cal_p*100:.1f}% ({correction_pp:+.1f}pp) | "
                f"mkt {mkt_impl*100:.1f}% | edge {edge:.1f}% | {tier}"
            )

            bets.append({
                "market":          market_name,
                "line":            line_str,
                "raw_prob":        raw_p,
                "our_prob":        cal_p,           # calibrated — this is what matters
                "calibrated_prob": cal_p,
                "market_implied":  mkt_impl,
                "edge_pct":        edge,
                "ev_pct":          ev["ev_pct"],
                "kelly_pct":       ev["kelly_pct"],
                "tier":            tier,
                "market_odds":     odds,
                "cal_method":      cal_method,
                "cal_correction":  correction_pp,
                "proj_total":      proj_total,
                "proj_margin":     proj_margin,
                "line_movement":   movement_pts,
                "movement_label":  movement_label,
                "movement_class":  movement_class,
            })

        # ── Total ──────────────────────────────────────────────────
        if mt is not None:
            add_bet("TOTAL OVER",  f"O {mt}", self.probability_over(mt),
                    over_odds,  "total_over",  "over",  "total_over")
            add_bet("TOTAL UNDER", f"U {mt}", self.probability_under(mt),
                    under_odds, "total_under", "under", "total_under")

        # ── Spread ─────────────────────────────────────────────────
        if ms is not None:
            add_bet(f"{self.home} ATS", f"{ms:+g}",
                    self.probability_home_covers(ms),
                    home_spread_odds, "spread", "home", "spread")
            add_bet(f"{self.away} ATS", f"{-ms:+g}",
                    self.probability_away_covers(ms),
                    away_spread_odds, "spread", "away", "spread")

        # ── Moneyline ──────────────────────────────────────────────
        if ml_h is not None and ml_a is not None:
            home_win = self.probability_home_wins()
            away_win = 1 - home_win
            h_impl   = american_to_implied_prob(ml_h)
            a_impl   = american_to_implied_prob(ml_a)
            h_clean, a_clean = remove_vig(h_impl, a_impl)

            # Calibrate ML probabilities
            h_cal, h_method, h_corr = _cal(home_win, "home_ml")
            a_cal, a_method, a_corr = _cal(away_win, "away_ml")

            if (h_cal - h_clean) * 100 >= min_edge_pct:
                ev = self.ev_bet(h_cal, ml_h)
                movement_info = lm.get(f"{self.home} ML", {})
                bets.append({
                    "market": f"{self.home} ML", "line": f"{ml_h:+d}",
                    "raw_prob": home_win, "our_prob": h_cal,
                    "calibrated_prob": h_cal, "market_implied": h_clean,
                    "edge_pct": (h_cal - h_clean) * 100,
                    "ev_pct": ev["ev_pct"], "kelly_pct": ev["kelly_pct"],
                    "tier": self.confidence_tier(h_cal), "market_odds": ml_h,
                    "cal_method": h_method, "cal_correction": h_corr,
                    "proj_total": proj_total, "proj_margin": proj_margin,
                    "line_movement": None, "movement_label": "⚪ No movement data",
                    "movement_class": "neutral",
                })
                logger.info(
                    f"    [{self.home} ML] raw {home_win*100:.1f}% → "
                    f"cal {h_cal*100:.1f}% ({h_corr:+.1f}pp) | "
                    f"mkt {h_clean*100:.1f}% | edge {(h_cal-h_clean)*100:.1f}%"
                )

            if (a_cal - a_clean) * 100 >= min_edge_pct:
                ev = self.ev_bet(a_cal, ml_a)
                bets.append({
                    "market": f"{self.away} ML", "line": f"{ml_a:+d}",
                    "raw_prob": away_win, "our_prob": a_cal,
                    "calibrated_prob": a_cal, "market_implied": a_clean,
                    "edge_pct": (a_cal - a_clean) * 100,
                    "ev_pct": ev["ev_pct"], "kelly_pct": ev["kelly_pct"],
                    "tier": self.confidence_tier(a_cal), "market_odds": ml_a,
                    "cal_method": a_method, "cal_correction": a_corr,
                    "proj_total": proj_total, "proj_margin": proj_margin,
                    "line_movement": None, "movement_label": "⚪ No movement data",
                    "movement_class": "neutral",
                })
                logger.info(
                    f"    [{self.away} ML] raw {away_win*100:.1f}% → "
                    f"cal {a_cal*100:.1f}% ({a_corr:+.1f}pp) | "
                    f"mkt {a_clean*100:.1f}% | edge {(a_cal-a_clean)*100:.1f}%"
                )

        bets.sort(key=lambda x: x["ev_pct"], reverse=True)
        return bets

    def summary(self) -> Dict[str, Any]:
        if self.totals is None:
            raise RuntimeError("Call run() first")
        return {
            "game":           f"{self.away} @ {self.home}",
            "proj_home":      round(float(np.mean(self.home_scores)), 1),
            "proj_away":      round(float(np.mean(self.away_scores)), 1),
            "proj_total":     round(float(np.mean(self.totals)), 1),
            "proj_margin":    round(float(np.mean(self.margins)), 1),
            "p_over":         round(self.probability_over(self.market_total), 4)  if self.market_total  else None,
            "p_under":        round(self.probability_under(self.market_total), 4) if self.market_total  else None,
            "p_home_cover":   round(self.probability_home_covers(self.market_spread), 4) if self.market_spread is not None else None,
            "p_home_win":     round(self.probability_home_wins(), 4),
            "total_dist":     self.total_distribution(),
            "home_out":       self.home_out,
            "away_out":       self.away_out,
            "home_b2b":       self.home_b2b,
            "away_b2b":       self.away_b2b,
            "ref_total_adj":  self.referee_total_adj,
            "n_sims":         self.n_sims,
            **self._run_metadata,
        }

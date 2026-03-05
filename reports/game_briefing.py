"""
Pre-game Briefing Generator

Given a game (home team, away team, optional date), outputs a complete structured
briefing covering model projection, injuries, recent form, H2H, top props, and
final recommendations.

Output:
  - Formatted terminal output
  - Returns dict with all sections
  - Saves to data/briefings/{YYYY-MM-DD}_{HOME}_{AWAY}_briefing.json

Usage:
    python reports/game_briefing.py PHX SAC
    python reports/game_briefing.py PHX SAC --date 2026-03-04
    python reports/game_briefing.py PHX SAC --odds-file data/odds/today.json
"""

import sys
import json
import argparse
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent.parent / "data"
BRIEFINGS_DIR = DATA_DIR / "briefings"


# ── Optional module imports with graceful fallback ────────────────────────────

def _try_import_matchup():
    try:
        from analyzers.matchup_analyzer import full_game_analysis
        return full_game_analysis
    except Exception as e:
        log.debug(f"matchup_analyzer unavailable: {e}")
        return None


def _try_import_injury():
    try:
        from analyzers.injury_impact import compute_injury_impact, get_team_injuries
        return compute_injury_impact, get_team_injuries
    except Exception as e:
        log.debug(f"injury_impact unavailable: {e}")
        return None, None


def _try_import_props():
    try:
        # prop_analyzer may be built by another subagent
        from analyzers.prop_analyzer import get_top_props_for_game
        return get_top_props_for_game
    except Exception:
        try:
            # Try collectors path
            from collectors.prop_analyzer import get_top_props_for_game
            return get_top_props_for_game
        except Exception as e:
            log.debug(f"prop_analyzer unavailable: {e}")
            return None


def _try_import_model():
    try:
        from training.train import load_model, predict_game
        return load_model, predict_game
    except Exception as e:
        log.debug(f"train module unavailable: {e}")
        return None, None


def _try_import_h2h():
    try:
        from collectors.h2h_collector import load_h2h, compute_h2h_stats
        return load_h2h, compute_h2h_stats
    except Exception as e:
        log.debug(f"h2h_collector unavailable: {e}")
        return None, None


def _try_import_nba_api():
    try:
        from collectors.nba_api import get_team_recent_games
        return get_team_recent_games
    except Exception as e:
        log.debug(f"nba_api get_team_recent_games unavailable: {e}")
        return None


def _try_import_home_away_splits():
    try:
        from reports.home_away_splits import load_splits, get_split_features_for_game
        return load_splits, get_split_features_for_game
    except Exception as e:
        log.debug(f"home_away_splits unavailable: {e}")
        return None, None


# ── Model projection ───────────────────────────────────────────────────────────

def get_model_projection(home: str, away: str) -> Dict[str, Any]:
    """
    Get model projection from the trained ML model.
    Falls back to a math-based estimate if model is unavailable.
    """
    load_model_fn, predict_fn = _try_import_model()
    if load_model_fn and predict_fn:
        try:
            model = load_model_fn()
            result = predict_fn(model, home, away)
            return {
                "projected_total": result.get("total", 218.0),
                "projected_spread": result.get("spread", 0.0),
                "confidence": result.get("confidence", "MEDIUM"),
                "source": "ml_model",
            }
        except Exception as e:
            log.debug(f"Model prediction failed: {e}")

    # Math fallback using home/away splits or defaults
    _, get_split_features = _try_import_home_away_splits()
    if get_split_features:
        try:
            feats = get_split_features(home, away)
            home_pts = feats.get("home_home_ortg", 113.0)
            away_pts = feats.get("away_away_ortg", 111.0)
            away_allowed = feats.get("home_home_drtg", 113.0)
            home_allowed = feats.get("away_away_drtg", 115.0)
            # Simple Pythagorean-style estimate
            league_avg = 113.0
            h_proj = home_pts * away_allowed / league_avg
            a_proj = away_pts * home_allowed / league_avg
            projected_total = round(h_proj + a_proj, 1)
            projected_spread = round(h_proj - a_proj, 1)
            return {
                "projected_total": projected_total,
                "projected_spread": projected_spread,
                "confidence": "LOW",
                "source": "math_fallback",
            }
        except Exception as e:
            log.debug(f"Split-based projection failed: {e}")

    return {
        "projected_total": 218.0,
        "projected_spread": 0.0,
        "confidence": "LOW",
        "source": "default",
    }


# ── Injury impact ─────────────────────────────────────────────────────────────

def get_injury_summary(home: str, away: str) -> Dict[str, Any]:
    """
    Compute injury impact for both teams.
    Returns structured impact dict with per-team breakdowns.
    """
    compute_fn, get_injuries_fn = _try_import_injury()

    if compute_fn:
        try:
            impact = compute_fn(home, away)
            return impact
        except Exception as e:
            log.debug(f"compute_injury_impact failed: {e}")

    if get_injuries_fn:
        try:
            home_inj = get_injuries_fn(home)
            away_inj = get_injuries_fn(away)
            return {
                "home_team": home,
                "away_team": away,
                "home_injuries": home_inj,
                "away_injuries": away_inj,
                "home_impact_pts": 0.0,
                "away_impact_pts": 0.0,
                "net_total_impact": 0.0,
                "source": "injuries_only",
            }
        except Exception as e:
            log.debug(f"get_team_injuries failed: {e}")

    return {
        "home_team": home,
        "away_team": away,
        "home_injuries": [],
        "away_injuries": [],
        "home_impact_pts": 0.0,
        "away_impact_pts": 0.0,
        "net_total_impact": 0.0,
        "source": "unavailable",
    }


# ── Recent form ───────────────────────────────────────────────────────────────

def get_recent_form(home: str, away: str, n: int = 5) -> Dict[str, Any]:
    """
    Get last N game scores for each team.
    Tries NBA API, then falls back to H2H data / game files.
    """
    get_recent = _try_import_nba_api()

    form: Dict[str, Any] = {}
    for team in [home, away]:
        if get_recent:
            try:
                games = get_recent(team, n=n)
                scores = [g.get("pts", g.get("home_score", g.get("score", 110))) for g in games[-n:]]
                form[team] = {"scores": scores, "avg": round(sum(scores)/len(scores), 1) if scores else 110.0}
                continue
            except Exception as e:
                log.debug(f"get_team_recent_games failed for {team}: {e}")

        # Fallback: scan data/games/ directory
        try:
            game_files = sorted((DATA_DIR / "games").glob(f"*{team}*.json"), reverse=True)[:n]
            scores = []
            for gf in game_files:
                with open(gf) as f:
                    g = json.load(f)
                if g.get("home") == team:
                    scores.append(g.get("home_score", 110))
                elif g.get("away") == team:
                    scores.append(g.get("away_score", 110))
            if scores:
                form[team] = {"scores": scores, "avg": round(sum(scores)/len(scores), 1)}
            else:
                form[team] = {"scores": [], "avg": None}
        except Exception as e:
            log.debug(f"Fallback form lookup failed for {team}: {e}")
            form[team] = {"scores": [], "avg": None}

    return form


# ── H2H ───────────────────────────────────────────────────────────────────────

def get_h2h_summary(home: str, away: str) -> Dict[str, Any]:
    """Get head-to-head stats for this matchup this season."""
    load_h2h_fn, compute_h2h_fn = _try_import_h2h()

    if load_h2h_fn:
        try:
            h2h_data = load_h2h_fn(home, away)
            if h2h_data:
                stats = compute_h2h_fn(home, away, h2h_data) if compute_h2h_fn else {}
                return {
                    "games": h2h_data,
                    "stats": stats,
                    "source": "h2h_collector",
                }
        except Exception as e:
            log.debug(f"H2H load failed: {e}")

    # Fallback: load from data/h2h/
    try:
        key1 = f"{home}_{away}"
        key2 = f"{away}_{home}"
        for key in [key1, key2]:
            h2h_path = DATA_DIR / "h2h" / f"{key}.json"
            if h2h_path.exists():
                with open(h2h_path) as f:
                    data = json.load(f)
                return {
                    "games": data.get("games", []),
                    "stats": data.get("stats", {}),
                    "source": "json_file",
                }
    except Exception as e:
        log.debug(f"H2H file fallback failed: {e}")

    return {"games": [], "stats": {}, "source": "unavailable"}


# ── Prop opportunities ────────────────────────────────────────────────────────

def get_prop_opportunities(home: str, away: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Get top prop bet opportunities for this game.
    Falls back to loading from data/props/ if analyzer unavailable.
    """
    get_props_fn = _try_import_props()

    if get_props_fn:
        try:
            props = get_props_fn(home, away, limit=limit)
            return props if props else []
        except Exception as e:
            log.debug(f"prop_analyzer failed: {e}")

    # Fallback: load saved prop files
    try:
        prop_files = sorted(
            (DATA_DIR / "props").glob(f"{home}_{away}*.json")
        )
        props = []
        for pf in prop_files[:limit]:
            with open(pf) as f:
                data = json.load(f)
            props.append(data)
        return props
    except Exception as e:
        log.debug(f"Props file fallback failed: {e}")

    return []


# ── Odds loading ──────────────────────────────────────────────────────────────

def load_today_odds(home: str, away: str, game_date: str) -> Optional[Dict[str, Any]]:
    """Try to load odds from data/odds/ for today's game."""
    try:
        odds_dir = DATA_DIR / "odds"
        if not odds_dir.exists():
            return None
        # Look for any JSON file that might contain this matchup
        for odds_file in sorted(odds_dir.glob("*.json"), reverse=True):
            try:
                with open(odds_file) as f:
                    data = json.load(f)
                # Handle list or dict
                games_list = data if isinstance(data, list) else data.get("games", [data])
                for g in games_list:
                    h = g.get("home_team", g.get("home", "")).upper()
                    a = g.get("away_team", g.get("away", "")).upper()
                    if (home in h or h in home) and (away in a or a in away):
                        return g
            except Exception:
                continue
    except Exception as e:
        log.debug(f"Odds load failed: {e}")
    return None


# ── Recommendations ───────────────────────────────────────────────────────────

def build_recommendations(
    projection: Dict[str, Any],
    injury_summary: Dict[str, Any],
    h2h: Dict[str, Any],
    odds: Optional[Dict[str, Any]],
    home: str,
    away: str,
) -> Dict[str, Any]:
    """
    Build bet recommendations from all available data.
    """
    projected_total = projection["projected_total"]
    net_injury_impact = injury_summary.get("net_total_impact", 0.0)
    injury_adjusted_total = round(projected_total + net_injury_impact, 1)

    recs: Dict[str, Any] = {
        "projected_total": projected_total,
        "injury_adjusted_total": injury_adjusted_total,
        "total_rec": None,
        "spread_rec": None,
        "notes": [],
    }

    # Get the betting line if available
    total_line = None
    spread_line = None
    book = "DraftKings"
    if odds:
        total_line = odds.get("total", odds.get("total_line"))
        spread_line = odds.get("spread", odds.get("home_spread"))
        book = odds.get("book", odds.get("sportsbook", "DraftKings"))

    # Total recommendation
    if total_line is not None:
        edge = injury_adjusted_total - float(total_line)
        direction = "UNDER" if edge < 0 else "OVER"
        abs_edge = abs(edge)
        if abs_edge >= 4:
            conf = "HIGH"
        elif abs_edge >= 2:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        recs["total_rec"] = {
            "direction": direction,
            "line": float(total_line),
            "proj": injury_adjusted_total,
            "edge": round(edge, 1),
            "confidence": conf,
            "book": book,
        }
        recs["total_summary"] = (
            f"{direction} {total_line} (model projects {projected_total}, "
            f"injury-adjusted {injury_adjusted_total})"
        )
    else:
        # No line available — just give the projection
        recs["total_summary"] = (
            f"Projected total: {projected_total} pts "
            f"(injury-adjusted: {injury_adjusted_total})"
        )

    # Spread recommendation
    spread = projection.get("projected_spread", 0.0)
    if spread_line is not None and abs(spread) >= 2:
        fav = home if spread < 0 else away
        recs["spread_rec"] = {
            "favorite": fav,
            "spread": spread_line,
            "proj_margin": spread,
            "book": book,
        }
        recs["spread_summary"] = f"{fav} {spread_line:+.1f} or better"
    elif abs(spread) >= 2:
        fav = home if spread < 0 else away
        recs["spread_summary"] = f"{fav} favored by {abs(spread):.1f} (projected)"
    else:
        recs["spread_summary"] = "Projected as even matchup"

    # H2H note
    h2h_stats = h2h.get("stats", {})
    h2h_avg = h2h_stats.get("avg_total", h2h_stats.get("h2h_total_avg"))
    if h2h_avg and abs(h2h_avg - projected_total) > 5:
        recs["notes"].append(
            f"H2H avg total ({h2h_avg:.1f}) differs from projection ({projected_total:.1f}) by "
            f"{abs(h2h_avg - projected_total):.1f} pts"
        )

    return recs


# ── Briefing formatter ────────────────────────────────────────────────────────

def format_briefing(
    home: str,
    away: str,
    game_date: str,
    projection: Dict[str, Any],
    injury_summary: Dict[str, Any],
    form: Dict[str, Any],
    h2h: Dict[str, Any],
    props: List[Dict[str, Any]],
    odds: Optional[Dict[str, Any]],
    recs: Dict[str, Any],
) -> str:
    """Format all briefing data into a readable string."""
    book = odds.get("book", odds.get("sportsbook", "DraftKings")) if odds else "DraftKings"

    lines = [
        "━" * 45,
        f"🏀 GAME BRIEFING: {away} @ {home}",
        f"   {game_date} | {book}",
        "━" * 45,
        "",
    ]

    # ── MODEL PROJECTION ────────────────────────────
    conf = projection.get("confidence", "MEDIUM")
    proj_total = projection.get("projected_total", 218.0)
    proj_spread = projection.get("projected_spread", 0.0)
    fav = home if proj_spread < 0 else (away if proj_spread > 0 else "EVEN")
    spread_str = f"{fav} {abs(proj_spread):.1f}" if proj_spread != 0 else "EVEN"

    lines += [
        "📊 MODEL PROJECTION",
        f"   Projected total: {proj_total} pts",
        f"   Projected spread: {spread_str}",
        f"   Confidence: {conf}",
        f"   Source: {projection.get('source', 'unknown')}",
        "",
    ]

    # ── INJURY IMPACT ────────────────────────────────
    home_inj = injury_summary.get("home_injuries", [])
    away_inj = injury_summary.get("away_injuries", [])
    home_impact = injury_summary.get("home_impact_pts", 0.0)
    away_impact = injury_summary.get("away_impact_pts", 0.0)
    net_impact = injury_summary.get("net_total_impact", 0.0)

    lines.append("⚠️  INJURY IMPACT")
    if away_inj:
        away_names = ", ".join(
            f"{p.get('player', p.get('name', 'Unknown'))} ({p.get('status', 'OUT')})"
            for p in away_inj[:3]
        )
        lines.append(f"   {away} missing: {away_names} → {abs(away_impact):.1f} pts impact")
    else:
        lines.append(f"   {away} missing: none reported")

    if home_inj:
        home_names = ", ".join(
            f"{p.get('player', p.get('name', 'Unknown'))} ({p.get('status', 'OUT')})"
            for p in home_inj[:3]
        )
        lines.append(f"   {home} missing: {home_names} → {abs(home_impact):.1f} pts impact")
    else:
        lines.append(f"   {home} missing: none reported")

    if net_impact != 0:
        direction = "lean UNDER" if net_impact < 0 else "lean OVER"
        lines.append(f"   Net impact on total: {net_impact:.1f} pts ({direction})")
    lines.append("")

    # ── RECENT FORM ──────────────────────────────────
    lines.append("📈 RECENT FORM (last 5)")
    for team in [away, home]:
        team_form = form.get(team, {})
        scores = team_form.get("scores", [])
        avg = team_form.get("avg")
        if scores:
            scores_str = ", ".join(str(s) for s in scores)
            avg_str = f"avg={avg:.1f}" if avg else ""
            lines.append(f"   {team}: {scores_str}  {avg_str}")
        else:
            lines.append(f"   {team}: no recent data")
    lines.append("")

    # ── H2H ─────────────────────────────────────────
    lines.append("🤝 H2H THIS SEASON")
    h2h_games = h2h.get("games", [])
    h2h_stats = h2h.get("stats", {})

    if h2h_games:
        n = len(h2h_games)
        # Count wins
        home_wins = sum(1 for g in h2h_games
                        if g.get("winner") == home or
                        (g.get("home") == home and g.get("home_score", 0) > g.get("away_score", 0)) or
                        (g.get("away") == home and g.get("away_score", 0) > g.get("home_score", 0)))

        avg_total = h2h_stats.get("avg_total", h2h_stats.get("h2h_total_avg"))
        if avg_total:
            lines.append(f"   {n} games: {home} {home_wins}-{n-home_wins} | Avg total: {avg_total:.1f}")
        else:
            lines.append(f"   {n} games: {home} {home_wins}-{n-home_wins}")

        # Show most recent game
        recent_g = h2h_games[-1] if h2h_games else None
        if recent_g:
            hs = recent_g.get("home_score", 0)
            as_ = recent_g.get("away_score", 0)
            total = recent_g.get("total", hs + as_)
            total_line_h2h = h2h_stats.get("typical_line")
            under_str = f"← UNDER {total_line_h2h}" if total_line_h2h and total < total_line_h2h else ""
            lines.append(f"   Last: {recent_g.get('home', home)} {hs}-{as_} (total {total}) {under_str}")
    else:
        lines.append("   No H2H data available this season")
    lines.append("")

    # ── TOP PROPS ────────────────────────────────────
    lines.append("💡 TOP PROP OPPORTUNITIES")
    if props:
        for i, prop in enumerate(props[:3], 1):
            player = prop.get("player", prop.get("name", "Unknown"))
            market = prop.get("market", prop.get("prop_type", "PTS"))
            line = prop.get("line", prop.get("threshold", 20))
            pred = prop.get("prediction", prop.get("predicted", line))
            direction = "OVER" if float(pred or line) > float(line) else "UNDER"
            emoji = "✅" if direction == "OVER" else "📉"
            conf_p = prop.get("confidence", "MEDIUM")
            lines.append(
                f"   {i}. {player} {line}+ {market.upper()}: "
                f"predicted {pred} | {direction} {emoji} ({conf_p} confidence)"
            )
    else:
        lines.append("   No prop data available")
    lines.append("")

    # ── ATS RISK FLAGS ───────────────────────────────
    try:
        from analyzers.variance_metrics import flag_ats_risk, compute_variance_metrics
        today_str = game_date
        spread_val = odds.get("spread") if odds else None
        if spread_val is not None:
            risk = flag_ats_risk(home, away, float(spread_val), today_str)
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk["risk_level"], "⚪")
            lines.append(f"📊 ATS RISK ASSESSMENT")
            lines.append(f"   {risk_emoji} {risk['risk_level']} (score: {risk['risk_score']}/10)")
            for w in risk["warnings"][:4]:
                lines.append(f"   {w}")
            # Line movement if available
            try:
                from collectors.line_tracker import get_line_movement, get_sharp_flags
                mv = get_line_movement(home, away, today_str)
                if mv:
                    direction_str = "🔥 SHARP ON UNDERDOG" if mv["direction"] == "toward_underdog" else "sharp on favorite"
                    lines.append(f"   📈 Line moved {mv['spread_change']:+.1f} pts → {direction_str}")
                    if mv.get("significant"):
                        lines.append(f"   ⚠️  SIGNIFICANT line movement — fade public?")
            except Exception:
                pass
            lines.append("")
    except Exception:
        pass

    # ── POSSESSIONS-BASED PROJECTION ────────────────
    try:
        from analyzers.four_factors import FourFactors
        ff = FourFactors()
        poss_proj = ff.project_game_total(home, away)
        lines.append("⚡ POSSESSIONS MODEL")
        lines.append(f"   Per-team poss: {poss_proj['expected_possessions']:.0f} | Projected total: {poss_proj['projected_total']} pts")
        lines.append(f"   {home} pts: {poss_proj['pts_home']:.0f} | {away} pts: {poss_proj['pts_away']:.0f}")
        lines.append(f"   Projected spread: {home} {poss_proj['projected_spread']:+.1f}")
        lines.append(f"   Pace — {home}: {poss_proj['home_pace']/2:.0f} poss/gm | {away}: {poss_proj['away_pace']/2:.0f} poss/gm")
        lines.append("")
    except Exception:
        poss_proj = None

    # ── EV ANALYSIS ─────────────────────────────────
    try:
        from analyzers.ev_calculator import EVCalculator
        ev_calc = EVCalculator(bankroll=500.0)

        if odds:
            total_line = odds.get("total") or odds.get("total_line")
            spread_line = odds.get("spread") or odds.get("home_spread")
            model_total = (poss_proj["projected_total"] if poss_proj
                           else projection.get("projected_total", 220))
            model_spread = (poss_proj["projected_spread"] if poss_proj
                            else projection.get("projected_spread", 0))

            lines.append("💰 EXPECTED VALUE ANALYSIS")
            if total_line:
                ev_result = ev_calc.evaluate_total(float(total_line), model_total)
                if ev_result["recommendation"] == "BET":
                    lines.append(f"   ✅ TOTAL BET: {ev_result['direction']} {total_line} | EV: {ev_result['ev_pct']} | Kelly: ${ev_result['suggested_bet']:.0f}")
                else:
                    lines.append(f"   ⛔ TOTAL: {ev_result.get('reason', 'No edge')}")
            if spread_line:
                ev_spread = ev_calc.evaluate_spread(home, away, float(spread_line), model_spread)
                if ev_spread["recommendation"] == "BET":
                    lines.append(f"   ✅ SPREAD BET: {ev_spread['bet_team']} | EV: {ev_spread['ev_pct']} | Kelly: ${ev_spread['suggested_bet']:.0f}")
                else:
                    lines.append(f"   ⛔ SPREAD: {ev_spread.get('reason', 'No edge')}")
            lines.append("   ⚠️  Only bet when EV ≥ 3%. Kelly sizing = optimal stake.")
            lines.append("")
    except Exception:
        pass

    # ── RECOMMENDATIONS ──────────────────────────────
    lines.append("🎯 RECOMMENDATIONS")
    total_line_val = recs.get("total_rec", {})
    lines.append(f"   Total: {recs.get('total_summary', 'N/A')}")
    lines.append(f"   Spread: {recs.get('spread_summary', 'N/A')}")
    for note in recs.get("notes", []):
        lines.append(f"   ⚠️  {note}")
    lines.append("")
    lines.append("━" * 45)

    return "\n".join(lines)


# ── Main briefing function ────────────────────────────────────────────────────

def generate_briefing(
    home: str,
    away: str,
    game_date: Optional[str] = None,
    odds_file: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate a complete pre-game briefing.

    Args:
        home: Home team tricode (e.g., "SAC")
        away: Away team tricode (e.g., "PHX")
        game_date: Date string YYYY-MM-DD (defaults to today)
        odds_file: Optional path to odds JSON file
        save: Save briefing to data/briefings/ (default True)
        verbose: Print to terminal (default True)

    Returns:
        dict with all briefing sections
    """
    home = home.upper()
    away = away.upper()

    if not game_date:
        game_date = date.today().isoformat()

    if verbose:
        print(f"\nGenerating briefing: {away} @ {home} on {game_date}...")

    # ── Gather all data ───────────────────────────────
    projection = get_model_projection(home, away)
    injury_summary = get_injury_summary(home, away)
    form = get_recent_form(home, away)
    h2h = get_h2h_summary(home, away)
    props = get_prop_opportunities(home, away)

    # Load odds
    odds: Optional[Dict[str, Any]] = None
    if odds_file:
        try:
            with open(odds_file) as f:
                raw = json.load(f)
            games_list = raw if isinstance(raw, list) else raw.get("games", [raw])
            for g in games_list:
                h = g.get("home_team", g.get("home", "")).upper()
                a = g.get("away_team", g.get("away", "")).upper()
                if (home in h or h in home) and (away in a or a in away):
                    odds = g
                    break
        except Exception as e:
            log.warning(f"Could not load odds file {odds_file}: {e}")
    if odds is None:
        odds = load_today_odds(home, away, game_date)

    # Build recommendations
    recs = build_recommendations(projection, injury_summary, h2h, odds, home, away)

    # Format the briefing
    formatted = format_briefing(
        home, away, game_date, projection, injury_summary,
        form, h2h, props, odds, recs
    )

    if verbose:
        print(formatted)

    # Assemble the output dict
    briefing = {
        "generated_at": datetime.now().isoformat(),
        "game_date": game_date,
        "home": home,
        "away": away,
        "odds": odds,
        "projection": projection,
        "injury_summary": injury_summary,
        "form": form,
        "h2h": h2h,
        "props": props,
        "recommendations": recs,
        "formatted_text": formatted,
    }

    # Save briefing
    if save:
        save_briefing(briefing, home, away, game_date)

    return briefing


def save_briefing(briefing: Dict[str, Any], home: str, away: str, game_date: str) -> Path:
    """Save briefing to data/briefings/{date}_{home}_{away}_briefing.json."""
    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{game_date}_{home}_{away}_briefing.json"
    path = BRIEFINGS_DIR / filename

    # Don't serialize unparseable bits
    serializable = {k: v for k, v in briefing.items() if k != "formatted_text"}
    serializable["formatted_text"] = briefing.get("formatted_text", "")

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"💾 Briefing saved: {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-game briefing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reports/game_briefing.py PHX SAC
  python reports/game_briefing.py PHX SAC --date 2026-03-04
  python reports/game_briefing.py PHX SAC --odds-file data/odds/today.json
  make briefing HOME=SAC AWAY=PHX
""",
    )
    parser.add_argument("home", help="Home team tricode (e.g., SAC)")
    parser.add_argument("away", help="Away team tricode (e.g., PHX)")
    parser.add_argument("--date", default=None, help="Game date YYYY-MM-DD (default: today)")
    parser.add_argument("--odds-file", default=None, help="Path to odds JSON file")
    parser.add_argument("--no-save", action="store_true", help="Don't save briefing to disk")
    parser.add_argument("--quiet", action="store_true", help="Don't print to terminal")
    args = parser.parse_args()

    briefing = generate_briefing(
        home=args.home,
        away=args.away,
        game_date=args.date,
        odds_file=args.odds_file,
        save=not args.no_save,
        verbose=not args.quiet,
    )

    # Exit 0 on success
    sys.exit(0)


if __name__ == "__main__":
    main()

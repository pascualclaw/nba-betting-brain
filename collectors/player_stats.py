"""
Player stats collector — pulls rolling game logs and opponent defensive stats from NBA API.
Caches results to data/players/ to avoid hammering the API.
"""
import json
import time
import sys
from pathlib import Path
from datetime import datetime

_REPO_ROOT = str(Path(__file__).parent.parent)
_COLLECTORS_DIR = str(Path(__file__).parent)
# Remove collectors/ from sys.path to avoid shadowing the installed nba_api package
if _COLLECTORS_DIR in sys.path:
    sys.path.remove(_COLLECTORS_DIR)
# Ensure repo root is first for local modules
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

from config import DATA_DIR, CURRENT_SEASON, NBA_API_DELAY

PLAYERS_DIR = DATA_DIR / "players"
PLAYERS_DIR.mkdir(parents=True, exist_ok=True)

# Position-to-stat mapping for opponent defensive analysis
STAT_COLUMN_MAP = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "PRA": None,  # computed from PTS + REB + AST
    "pra": None,
}

# Team tricode → full name mappings for NBA API
TEAM_NAME_MAP = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def find_player_id(player_name: str) -> int | None:
    """Map player name to NBA player ID using fuzzy search."""
    # Try exact match first
    results = nba_players.find_players_by_full_name(player_name)
    if results:
        # Prefer active players
        active = [p for p in results if p["is_active"]]
        return (active[0] if active else results[0])["id"]
    
    # Try last name only
    parts = player_name.strip().split()
    if len(parts) >= 2:
        last = parts[-1]
        results = nba_players.find_players_by_full_name(last)
        if results:
            active = [p for p in results if p["is_active"]]
            return (active[0] if active else results[0])["id"]
    
    return None


def _cache_path(player_name: str, season: str) -> Path:
    safe_name = player_name.replace(" ", "_").replace("'", "")
    return PLAYERS_DIR / f"{safe_name}_{season}.json"


def _load_cache(player_name: str, season: str) -> dict | None:
    path = _cache_path(player_name, season)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # Cache valid for 4 hours
            cached_at = data.get("_cached_at", 0)
            if (time.time() - cached_at) < 4 * 3600:
                return data
        except Exception:
            pass
    return None


def _save_cache(player_name: str, season: str, data: dict):
    data["_cached_at"] = time.time()
    path = _cache_path(player_name, season)
    path.write_text(json.dumps(data, indent=2))


def get_player_rolling_stats(player_name: str, n_games: int = 10, 
                              season: str = CURRENT_SEASON,
                              use_cache: bool = True) -> dict:
    """
    Get rolling player stats from the last n_games games.
    
    Returns:
        {
            player_name, player_id, n_games,
            pts_avg, reb_avg, ast_avg, pra_avg, min_avg,
            pts_last5, reb_last5, ast_last5, pra_last5,
            pts_std, reb_std, ast_std, pra_std,
            hit_rate,  # % of last n games hitting "typical" DK prop line (rough)
            games: [list of individual game stats]
        }
    """
    if use_cache:
        cached = _load_cache(player_name, season)
        if cached and cached.get("n_games") == n_games:
            return cached

    player_id = find_player_id(player_name)
    if not player_id:
        return {"error": f"Player not found: {player_name}", "player_name": player_name}

    try:
        time.sleep(NBA_API_DELAY)
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = log.get_data_frames()[0]
    except Exception as e:
        return {"error": f"NBA API error: {e}", "player_name": player_name}

    if df.empty:
        return {"error": f"No game log data for {player_name}", "player_name": player_name}

    # Most recent n_games
    df = df.head(n_games)

    games = []
    for _, row in df.iterrows():
        pts = int(row.get("PTS", 0))
        reb = int(row.get("REB", 0))
        ast = int(row.get("AST", 0))
        pra = pts + reb + ast
        
        # Parse minutes
        min_str = str(row.get("MIN", "0"))
        try:
            mins = float(min_str.split(":")[0]) + float(min_str.split(":")[1]) / 60 if ":" in min_str else float(min_str)
        except Exception:
            mins = 0.0

        games.append({
            "date": str(row.get("GAME_DATE", "")),
            "matchup": str(row.get("MATCHUP", "")),
            "wl": str(row.get("WL", "")),
            "minutes": round(mins, 1),
            "pts": pts,
            "reb": reb,
            "ast": ast,
            "pra": pra,
        })

    def avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0.0
    def std(lst):
        if len(lst) < 2:
            return 0.0
        m = avg(lst)
        return round((sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5, 2)

    pts_list = [g["pts"] for g in games]
    reb_list = [g["reb"] for g in games]
    ast_list = [g["ast"] for g in games]
    pra_list = [g["pra"] for g in games]
    min_list = [g["minutes"] for g in games]

    last5 = games[:5]
    pts_avg = avg(pts_list)
    reb_avg = avg(reb_list)
    ast_avg = avg(ast_list)
    pra_avg = avg(pra_list)

    # Estimate a "typical" DK prop line as ~90% of rolling avg, then calc hit rate
    def hit_rate_for(values: list, avg_val: float) -> float:
        if not values or avg_val == 0:
            return 0.0
        line = round(avg_val * 0.9 - 0.5)  # rough DK-style line
        hits = sum(1 for v in values if v > line)
        return round(hits / len(values), 3)

    result = {
        "player_name": player_name,
        "player_id": player_id,
        "season": season,
        "n_games": len(games),
        "pts_avg": pts_avg,
        "reb_avg": reb_avg,
        "ast_avg": ast_avg,
        "pra_avg": pra_avg,
        "min_avg": avg(min_list),
        "pts_std": std(pts_list),
        "reb_std": std(reb_list),
        "ast_std": std(ast_list),
        "pra_std": std(pra_list),
        "pts_last5": avg([g["pts"] for g in last5]),
        "reb_last5": avg([g["reb"] for g in last5]),
        "ast_last5": avg([g["ast"] for g in last5]),
        "pra_last5": avg([g["pra"] for g in last5]),
        "hit_rate": hit_rate_for(pra_list, pra_avg),
        "games": games,
    }

    _save_cache(player_name, season, result)
    return result


def get_opponent_def_stats(opponent: str, stat_type: str,
                            season: str = CURRENT_SEASON) -> dict:
    """
    Get how many pts/reb/ast the opponent allows per game.
    Uses LeagueDashTeamStats (opponent stats aren't per-position, but team opp stats
    give us a good signal for how easy/hard they are to score against).
    
    Returns:
        {
            team, stat_type,
            opp_pts_allowed, opp_reb_allowed, opp_ast_allowed,
            rank_pts (1=best defense, 30=worst), rank_reb, rank_ast,
            league_avg_pts, league_avg_reb, league_avg_ast,
            defensive_rating  (pts allowed per 100 possessions)
        }
    """
    cache_path = DATA_DIR / "players" / f"def_stats_{season}.json"
    def_cache = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
            cached_at = raw.get("_cached_at", 0)
            if (time.time() - cached_at) < 6 * 3600:
                def_cache = raw
        except Exception:
            pass

    if not def_cache:
        try:
            time.sleep(NBA_API_DELAY)
            # Get opponent stats
            opp_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Opponent",
                per_mode_detailed="PerGame",
            )
            opp_df = opp_stats.get_data_frames()[0]

            time.sleep(NBA_API_DELAY)
            # Get defensive rating
            adv_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            )
            adv_df = adv_stats.get_data_frames()[0]

            # Build lookup by team abbreviation
            team_lookup = {}
            for _, row in opp_df.iterrows():
                abb = str(row.get("TEAM_ABBREVIATION", ""))
                team_lookup[abb] = {
                    "opp_pts": float(row.get("OPP_PTS", 0)),
                    "opp_reb": float(row.get("OPP_REB", row.get("OPP_OREB", 0))),
                    "opp_ast": float(row.get("OPP_AST", 0)),
                }

            # Merge defensive rating
            for _, row in adv_df.iterrows():
                abb = str(row.get("TEAM_ABBREVIATION", ""))
                if abb in team_lookup:
                    team_lookup[abb]["def_rating"] = float(row.get("DEF_RATING", 0))

            # Compute ranks (lower opp pts = better defense = rank 1)
            pts_sorted = sorted(team_lookup.items(), key=lambda x: x[1].get("opp_pts", 0))
            reb_sorted = sorted(team_lookup.items(), key=lambda x: x[1].get("opp_reb", 0))
            ast_sorted = sorted(team_lookup.items(), key=lambda x: x[1].get("opp_ast", 0))
            
            for rank, (abb, _) in enumerate(pts_sorted, 1):
                team_lookup[abb]["rank_pts_def"] = rank
            for rank, (abb, _) in enumerate(reb_sorted, 1):
                team_lookup[abb]["rank_reb_def"] = rank
            for rank, (abb, _) in enumerate(ast_sorted, 1):
                team_lookup[abb]["rank_ast_def"] = rank

            # League averages
            all_pts = [v["opp_pts"] for v in team_lookup.values() if v.get("opp_pts")]
            all_reb = [v["opp_reb"] for v in team_lookup.values() if v.get("opp_reb")]
            all_ast = [v["opp_ast"] for v in team_lookup.values() if v.get("opp_ast")]

            def_cache = {
                "teams": team_lookup,
                "league_avg_pts": round(sum(all_pts) / len(all_pts), 2) if all_pts else 0,
                "league_avg_reb": round(sum(all_reb) / len(all_reb), 2) if all_reb else 0,
                "league_avg_ast": round(sum(all_ast) / len(all_ast), 2) if all_ast else 0,
                "_cached_at": time.time(),
            }
            cache_path.write_text(json.dumps(def_cache, indent=2))

        except Exception as e:
            return {"error": f"Failed to get defensive stats: {e}", "team": opponent}

    teams = def_cache.get("teams", {})
    team_data = teams.get(opponent.upper(), {})

    stat_map = {
        "points": "opp_pts",
        "pts": "opp_pts",
        "rebounds": "opp_reb",
        "reb": "opp_reb",
        "assists": "opp_ast",
        "ast": "opp_ast",
        "PRA": None,
        "pra": None,
    }
    
    key = stat_map.get(stat_type.lower() if stat_type != "PRA" else "PRA", "opp_pts")
    
    opp_pts = team_data.get("opp_pts", def_cache.get("league_avg_pts", 0))
    opp_reb = team_data.get("opp_reb", def_cache.get("league_avg_reb", 0))
    opp_ast = team_data.get("opp_ast", def_cache.get("league_avg_ast", 0))

    # For PRA, combine all three opp stats — but we scale it to per-player avg
    if stat_type.upper() == "PRA":
        # Rough: opponents allow (opp_pts + opp_reb + opp_ast) combined per game
        # We return raw opp_allowed for the stat requested
        opp_val = opp_pts + opp_reb + opp_ast
        rank = team_data.get("rank_pts_def", 15)  # use pts rank as proxy
    else:
        opp_val = team_data.get(key, 0) if key else 0
        rank_key = {"opp_pts": "rank_pts_def", "opp_reb": "rank_reb_def", "opp_ast": "rank_ast_def"}.get(key, "rank_pts_def")
        rank = team_data.get(rank_key, 15)

    league_avg = {
        "opp_pts": def_cache.get("league_avg_pts", 0),
        "opp_reb": def_cache.get("league_avg_reb", 0),
        "opp_ast": def_cache.get("league_avg_ast", 0),
    }.get(key, 0) if key else 0

    return {
        "team": opponent.upper(),
        "stat_type": stat_type,
        "opp_val_allowed": round(opp_val, 2),
        "opp_pts_allowed": round(opp_pts, 2),
        "opp_reb_allowed": round(opp_reb, 2),
        "opp_ast_allowed": round(opp_ast, 2),
        "def_rating": round(team_data.get("def_rating", 0), 2),
        "rank": rank,  # 1 = best defense (allows least), 30 = worst
        "league_avg": round(league_avg, 2),
        "is_weak_defense": rank >= 20,   # bottom 11 teams = weak defense
        "is_strong_defense": rank <= 10, # top 10 teams = strong defense
    }


if __name__ == "__main__":
    import sys
    player = sys.argv[1] if len(sys.argv) > 1 else "Nique Clifford"
    
    print(f"\n🏀 Rolling stats for {player}...")
    stats = get_player_rolling_stats(player, n_games=10)
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
    else:
        print(f"  Player ID: {stats['player_id']}")
        print(f"  Games analyzed: {stats['n_games']}")
        print(f"  PTS: {stats['pts_avg']} avg (last 5: {stats['pts_last5']})")
        print(f"  REB: {stats['reb_avg']} avg (last 5: {stats['reb_last5']})")
        print(f"  AST: {stats['ast_avg']} avg (last 5: {stats['ast_last5']})")
        print(f"  PRA: {stats['pra_avg']} avg (last 5: {stats['pra_last5']})")
        print(f"  MIN: {stats['min_avg']} avg")
        print(f"  Hit rate (rolling): {stats['hit_rate']*100:.0f}%")
        print(f"\n  Last {stats['n_games']} games:")
        for g in stats["games"]:
            print(f"    {g['date']:12s} {g['matchup']:20s} | PTS:{g['pts']:3d} REB:{g['reb']:3d} AST:{g['ast']:3d} PRA:{g['pra']:3d} MIN:{g['minutes']:5.1f}")
    
    print(f"\n🛡️  PHX defensive stats vs rebounds:")
    def_stats = get_opponent_def_stats("PHX", "rebounds")
    if "error" in def_stats:
        print(f"❌ Error: {def_stats['error']}")
    else:
        print(f"  OPP REB allowed: {def_stats['opp_reb_allowed']}/game")
        print(f"  Rank: #{def_stats['rank']} (1=best defense, 30=worst)")
        print(f"  Weak defense: {def_stats['is_weak_defense']}")

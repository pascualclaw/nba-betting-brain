"""
training/historical_loader.py — Loads historical NBA game data from NBA CDN.

Covers up to 2 full NBA seasons. Resume-friendly: skips dates already in DB.
Computes rolling team stats snapshots (ORTG/DRTG) as of each game date.

Usage:
    python -m training.historical_loader
    python -m training.historical_loader --season 2024-25
    python -m training.historical_loader --start 2025-10-28 --end 2026-03-04
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

import config
from database.db import DB

logger = logging.getLogger(__name__)
console = Console()

# ─── NBA CDN API helpers ───────────────────────────────────────────────────────

NBA_SCHEDULE_TEMPLATE = (
    "https://data.nba.com/data/10s/v2015/json/mobile_teams"
    "/nba/{year}/league/00_full_schedule.json"
)

NBA_BOXSCORE_TEMPLATE = (
    "https://cdn.nba.com/static/json/staticData/boxscore/BS_{game_id}.json"
)

NBA_SCOREBOARD_DATE_TEMPLATE = (
    "https://stats.nba.com/stats/scoreboardV2?GameDate={date}&LeagueID=00&DayOffset=0"
)


def _get(url: str, retries: int = 3, delay: float = 1.5) -> Optional[dict]:
    """HTTP GET with retries and rate-limit courtesy delay."""
    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                headers=config.NBA_STATS_HEADERS,
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = delay * (attempt + 1) * 2
                logger.warning("Rate limited. Waiting %.1fs …", wait)
                time.sleep(wait)
            else:
                logger.debug("HTTP %s for %s", resp.status_code, url)
        except requests.RequestException as exc:
            logger.warning("Request failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(delay * (attempt + 1))
    return None


# ─── Fetch scoreboards by date ────────────────────────────────────────────────

def fetch_scoreboard_for_date(game_date: str) -> list[dict]:
    """
    Fetch all games played on `game_date` (YYYY-MM-DD) from stats.nba.com.
    Returns a list of raw game dicts.
    """
    url = NBA_SCOREBOARD_DATE_TEMPLATE.format(date=game_date)
    data = _get(url)
    if not data:
        return []

    # Parse the NBA stats API response format
    games: list[dict] = []
    try:
        results = data.get("resultSets", [])
        game_header = next(
            (r for r in results if r["name"] == "GameHeader"), None
        )
        line_score = next(
            (r for r in results if r["name"] == "LineScore"), None
        )

        if not game_header or not line_score:
            return []

        gh_headers = game_header["headers"]
        gh_rows = game_header["rowSet"]
        ls_headers = line_score["headers"]
        ls_rows = line_score["rowSet"]

        def to_dict(headers: list[str], row: list) -> dict:
            return dict(zip(headers, row))

        # Build line score lookup: game_id -> list of team rows
        ls_by_game: dict[str, list[dict]] = defaultdict(list)
        for row in ls_rows:
            d = to_dict(ls_headers, row)
            ls_by_game[d["GAME_ID"]].append(d)

        for row in gh_rows:
            gh = to_dict(gh_headers, row)
            game_id = gh["GAME_ID"]
            status_text = gh.get("GAME_STATUS_TEXT", "")

            # Only process completed games
            if "Final" not in status_text and gh.get("GAME_STATUS_ID") != 3:
                continue

            ls_teams = ls_by_game.get(game_id, [])
            if len(ls_teams) < 2:
                continue

            # Identify home vs away by HOME_TEAM_ID
            home_id = str(gh.get("HOME_TEAM_ID", ""))
            away_id = str(gh.get("VISITOR_TEAM_ID", ""))

            home_ls = next(
                (t for t in ls_teams if str(t.get("TEAM_ID", "")) == home_id),
                ls_teams[0],
            )
            away_ls = next(
                (t for t in ls_teams if str(t.get("TEAM_ID", "")) == away_id),
                ls_teams[1],
            )

            def safe_int(v: object) -> Optional[int]:
                try:
                    return int(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            home_score = safe_int(home_ls.get("PTS"))
            away_score = safe_int(away_ls.get("PTS"))
            total = (
                (home_score or 0) + (away_score or 0)
                if home_score is not None and away_score is not None
                else None
            )

            # Quarter scores
            q1_home = safe_int(home_ls.get("PTS_QTR1"))
            q1_away = safe_int(away_ls.get("PTS_QTR1"))
            q1_total = (
                (q1_home or 0) + (q1_away or 0)
                if q1_home is not None and q1_away is not None
                else None
            )

            # Determine season
            season = _date_to_season(game_date)

            games.append(
                {
                    "game_id": game_id,
                    "date": game_date,
                    "home": home_ls.get("TEAM_ABBREVIATION", ""),
                    "away": away_ls.get("TEAM_ABBREVIATION", ""),
                    "home_score": home_score,
                    "away_score": away_score,
                    "total": total,
                    "q1_home": q1_home,
                    "q1_away": q1_away,
                    "q2_home": safe_int(home_ls.get("PTS_QTR2")),
                    "q2_away": safe_int(away_ls.get("PTS_QTR2")),
                    "q3_home": safe_int(home_ls.get("PTS_QTR3")),
                    "q3_away": safe_int(away_ls.get("PTS_QTR3")),
                    "q4_home": safe_int(home_ls.get("PTS_QTR4")),
                    "q4_away": safe_int(away_ls.get("PTS_QTR4")),
                    "ot_home": safe_int(home_ls.get("PTS_OT1", 0)) or 0,
                    "ot_away": safe_int(away_ls.get("PTS_OT1", 0)) or 0,
                    "q1_total": q1_total,
                    "pace": None,
                    "season": season,
                    "status": "FINAL",
                    "arena": gh.get("ARENA_NAME"),
                    "is_neutral": 0,
                }
            )
    except Exception as exc:
        logger.error("Error parsing scoreboard for %s: %s", game_date, exc)

    return games


def _date_to_season(date_str: str) -> str:
    """Convert a date string to an NBA season label (e.g. '2025-10-29' → '2025-26')."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    # NBA season crosses year boundary: Oct-Jun
    if d.month >= 10:
        return f"{d.year}-{str(d.year + 1)[-2:]}"
    else:
        return f"{d.year - 1}-{str(d.year)[-2:]}"


# ─── Rolling stats builder ────────────────────────────────────────────────────

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den != 0 else default


def _compute_ortg(pts: float, possessions: float) -> float:
    """Offensive rating = (PTS / possessions) * 100."""
    return _safe_div(pts * 100.0, possessions, default=110.0)


def _compute_drtg(opp_pts: float, opp_possessions: float) -> float:
    """Defensive rating = (opp_PTS / opp_possessions) * 100."""
    return _safe_div(opp_pts * 100.0, opp_possessions, default=110.0)


def _estimate_possessions(pts: Optional[int], opp_pts: Optional[int]) -> float:
    """
    Rough possession estimate when detailed box score isn't available.
    Uses: possessions ≈ (pts + opp_pts) / 2 / (pts_per_100_poss / 100)
    Simpler: assumes league avg ~113 pts per 100 possessions.
    """
    if pts is None or opp_pts is None:
        return 100.0
    avg_pts = (pts + opp_pts) / 2.0
    # NBA avg ~113 pts/100 poss → ~1.13 pts/poss
    return avg_pts / 1.13


def build_rolling_team_snapshots(
    games: list[dict],
    window: int = 20,
    short_window: int = 10,
) -> dict[str, dict]:
    """
    Given a chronologically sorted list of completed games, compute
    rolling ORTG/DRTG snapshots for each team as of each game date.

    Returns: {team: {date: snapshot_dict}}
    """
    # team → list of (date, pts_for, pts_against, possessions)
    team_games: dict[str, list[tuple]] = defaultdict(list)

    for g in sorted(games, key=lambda x: x["date"]):
        home, away = g["home"], g["away"]
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        poss = _estimate_possessions(hs, as_)
        team_games[home].append((g["date"], hs, as_, poss, g["game_id"]))
        team_games[away].append((g["date"], as_, hs, poss, g["game_id"]))

    snapshots: dict[str, dict[str, dict]] = {}

    for team, history in team_games.items():
        snapshots[team] = {}
        for idx, (gdate, pts_for, pts_against, poss, game_id) in enumerate(history):
            # Use games BEFORE this one (no look-ahead)
            prior = history[:idx]
            window_games = prior[-window:]
            short_games = prior[-short_window:]

            if not window_games:
                snap = {
                    "team": team, "date": gdate, "game_id": game_id,
                    "ortg": 110.0, "drtg": 110.0, "pace": 98.0,
                    "net_rtg": 0.0, "last_10_ortg": 110.0,
                    "last_10_drtg": 110.0, "games_played": 0,
                }
            else:
                total_pts_for = sum(w[1] for w in window_games)
                total_pts_against = sum(w[2] for w in window_games)
                total_poss = sum(w[3] for w in window_games)
                n = len(window_games)

                ortg = _compute_ortg(total_pts_for, total_poss)
                drtg = _compute_drtg(total_pts_against, total_poss)
                pace = _safe_div(total_poss * 48.0, n, default=98.0)

                if short_games:
                    s_pts_for = sum(w[1] for w in short_games)
                    s_pts_against = sum(w[2] for w in short_games)
                    s_poss = sum(w[3] for w in short_games)
                    last_10_ortg = _compute_ortg(s_pts_for, s_poss)
                    last_10_drtg = _compute_drtg(s_pts_against, s_poss)
                else:
                    last_10_ortg = ortg
                    last_10_drtg = drtg

                snap = {
                    "team": team,
                    "date": gdate,
                    "game_id": game_id,
                    "ortg": round(ortg, 2),
                    "drtg": round(drtg, 2),
                    "pace": round(pace, 2),
                    "net_rtg": round(ortg - drtg, 2),
                    "last_10_ortg": round(last_10_ortg, 2),
                    "last_10_drtg": round(last_10_drtg, 2),
                    "games_played": n,
                }

            snapshots[team][gdate] = snap

    return snapshots


# ─── Main loader ───────────────────────────────────────────────────────────────

def load_season(
    season: str,
    db: DB,
    start_override: Optional[str] = None,
    end_override: Optional[str] = None,
    delay_between_dates: float = 0.5,
) -> int:
    """
    Load all games for a given season from NBA CDN.

    Args:
        season: e.g. "2025-26"
        db: Open DB connection
        start_override: Override start date (YYYY-MM-DD)
        end_override: Override end date (YYYY-MM-DD)
        delay_between_dates: Seconds to wait between date requests

    Returns:
        Number of games loaded.
    """
    start_str = start_override or config.SEASON_START_DATES.get(season, "2025-10-28")
    end_str = end_override or date.today().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()

    total_days = (end_dt - start_dt).days + 1
    all_dates = [
        (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(total_days)
    ]

    console.print(
        f"[bold cyan]Loading {season}[/bold cyan]: {start_str} → {end_str} "
        f"({total_days} days)"
    )

    games_loaded = 0
    games_buffer: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Fetching {season} games…", total=len(all_dates))

        for game_date in all_dates:
            progress.update(task, description=f"Fetching {game_date}…")

            # Skip if already loaded
            if db.date_loaded(game_date, season):
                progress.advance(task)
                continue

            games = fetch_scoreboard_for_date(game_date)

            for game in games:
                if not db.game_exists(game["game_id"]):
                    db.upsert_game(game)
                    games_buffer.append(game)
                    games_loaded += 1

            time.sleep(delay_between_dates)
            progress.advance(task)

    console.print(
        f"  ✓ {games_loaded} new games loaded for [bold]{season}[/bold]"
    )

    # Build and store rolling snapshots
    if games_buffer or True:  # always rebuild for loaded season
        console.print("  ↻ Building rolling team stat snapshots…")
        all_season_games = db.get_games_for_training(season)
        snapshots = build_rolling_team_snapshots(
            all_season_games,
            window=config.TEAM_ROLLING_WINDOW,
            short_window=10,
        )

        snap_count = 0
        for team, date_snaps in snapshots.items():
            for snap in date_snaps.values():
                db.upsert_team_snapshot(snap)
                snap_count += 1

        console.print(
            f"  ✓ {snap_count} team snapshots stored for [bold]{season}[/bold]"
        )

    return games_loaded


def main() -> None:
    """Entry point for CLI invocation."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    parser = argparse.ArgumentParser(
        description="Load historical NBA game data into SQLite"
    )
    parser.add_argument(
        "--season",
        help="Season to load (e.g. 2025-26). Loads all configured seasons if omitted.",
    )
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between date requests (default 0.5)",
    )
    args = parser.parse_args()

    seasons = [args.season] if args.season else config.SEASONS

    with DB() as db:
        total = 0
        for season in seasons:
            n = load_season(
                season=season,
                db=db,
                start_override=args.start,
                end_override=args.end,
                delay_between_dates=args.delay,
            )
            total += n

    console.print(f"\n[bold green]Done![/bold green] {total} total games loaded.")


if __name__ == "__main__":
    main()

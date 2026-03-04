"""
Action Network scraper — public betting %, sharp money signals, line movement.

Key signals:
- Public % on a side: >70% public on one side = fade the public opportunity
- Sharp money: line moves AGAINST the public = sharp action on other side
- Steam moves: sudden line jump across multiple books = sharp alert
"""
import requests
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "action_network"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ACTION_BASE = "https://api.actionnetwork.com/web/v1"


def get_nba_games_with_percentages() -> list:
    """
    Pull public betting percentages for today's NBA games.
    """
    url = f"{ACTION_BASE}/games?sport=nba&period=game"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json",
        "Referer": "https://www.actionnetwork.com/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("games", [])
    except Exception as e:
        return [{"error": str(e)}]


def parse_betting_splits(game: dict) -> dict:
    """Parse public % and line data from an Action Network game object."""
    teams = game.get("teams", [])
    if len(teams) < 2:
        return {}

    home = teams[0] if teams[0].get("is_home") else teams[1]
    away = teams[1] if teams[0].get("is_home") else teams[0]

    return {
        "home_team": home.get("full_name", ""),
        "away_team": away.get("full_name", ""),
        "spread_home": game.get("spread", {}).get("spread_home"),
        "total": game.get("total", {}).get("total"),
        "public_pct": {
            "home_spread": game.get("spread", {}).get("home_spread_pct"),
            "away_spread": game.get("spread", {}).get("away_spread_pct"),
            "over": game.get("total", {}).get("over_pct"),
            "under": game.get("total", {}).get("under_pct"),
            "home_ml": game.get("moneyline", {}).get("home_ml_pct"),
            "away_ml": game.get("moneyline", {}).get("away_ml_pct"),
        },
        "money_pct": {
            "home_spread": game.get("spread", {}).get("home_spread_money_pct"),
            "away_spread": game.get("spread", {}).get("away_spread_money_pct"),
            "over": game.get("total", {}).get("over_money_pct"),
            "under": game.get("total", {}).get("under_money_pct"),
        },
    }


def get_sharp_signals(splits: dict) -> list:
    """
    Identify sharp betting signals from public vs money discrepancies.
    
    Sharp signal = lots of public tickets on one side, but money moving other way.
    Example: 75% of bettors on Over, but 60% of money on Under = sharp Under.
    """
    signals = []
    pct = splits.get("public_pct", {})
    money = splits.get("money_pct", {})

    checks = [
        ("over", "under", "Total Over"),
        ("under", "over", "Total Under"),
        ("home_spread", "away_spread", "Home Spread"),
        ("away_spread", "home_spread", "Away Spread"),
    ]

    for public_key, money_key, label in checks:
        pub = pct.get(public_key)
        mon = money.get(public_key)
        if pub and mon:
            try:
                pub_f, mon_f = float(pub), float(mon)
                # Sharp if >60% public but money is weighted against
                if pub_f > 60 and mon_f < 45:
                    signals.append({
                        "signal": f"🔥 SHARP FADE: {label}",
                        "detail": f"{pub_f:.0f}% of bettors but only {mon_f:.0f}% of money",
                        "bet": f"Fade the public — take the other side",
                    })
                # Sharp if money flowing heavily against small public %
                elif mon_f > 65 and pub_f < 40:
                    signals.append({
                        "signal": f"💰 SHARP MONEY: {label}",
                        "detail": f"Only {pub_f:.0f}% public but {mon_f:.0f}% of money",
                        "bet": f"Follow the sharp money",
                    })
            except (ValueError, TypeError):
                continue

    return signals


def format_splits_report(team1: str, team2: str) -> str:
    """Generate a betting splits report for a matchup."""
    games = get_nba_games_with_percentages()

    if not games or (len(games) == 1 and "error" in games[0]):
        err = games[0].get("error", "Unknown") if games else "No data"
        return f"⚠️ Action Network data unavailable: {err}\nTry fetching directly at actionnetwork.com"

    # Find this game
    target = None
    for g in games:
        teams_str = json.dumps(g.get("teams", [])).upper()
        if team1.upper() in teams_str and team2.upper() in teams_str:
            target = g
            break

    if not target:
        return f"Game {team1} vs {team2} not found in Action Network data (may not be listed yet)"

    splits = parse_betting_splits(target)
    signals = get_sharp_signals(splits)

    lines = [
        f"## 📊 Betting Splits: {team1} vs {team2}",
        f"",
        f"**Spread ({splits.get('spread_home', 'N/A')}):**",
        f"  Home: {splits['public_pct'].get('home_spread', '?')}% tickets | {splits['money_pct'].get('home_spread', '?')}% money",
        f"  Away: {splits['public_pct'].get('away_spread', '?')}% tickets | {splits['money_pct'].get('away_spread', '?')}% money",
        f"",
        f"**Total ({splits.get('total', 'N/A')}):**",
        f"  Over: {splits['public_pct'].get('over', '?')}% tickets | {splits['money_pct'].get('over', '?')}% money",
        f"  Under: {splits['public_pct'].get('under', '?')}% tickets | {splits['money_pct'].get('under', '?')}% money",
    ]

    if signals:
        lines.append(f"\n**🔥 Sharp Signals:**")
        for s in signals:
            lines.append(f"  {s['signal']}: {s['detail']}")
            lines.append(f"  → {s['bet']}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Testing Action Network...\n")
    print(format_splits_report("PHX", "SAC"))

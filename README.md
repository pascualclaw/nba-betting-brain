# 🏀 NBA Betting Brain

A self-learning NBA betting intelligence system that gets sharper every day.

## What It Does

- **Pulls** daily NBA game data, box scores, injury reports, H2H history
- **Analyzes** matchup-specific pace, prop hit rates, line movement patterns
- **Tracks** every recommended bet vs actual outcome
- **Learns** from mistakes — builds a lessons database that improves future analysis
- **Reports** pre-game intelligence for any NBA matchup on demand

## Core Loop

```
Daily Cron (morning)
  → Pull yesterday's results
  → Score any tracked props (hit/miss)
  → Update H2H + player stat databases
  → Log lessons learned

On-Demand (per game)
  → Pull live injuries
  → Generate H2H pace analysis
  → Check player prop hit rates in this matchup
  → Surface best bets with confidence %
```

## Lessons Learned So Far

See `data/lessons/lessons.json` — entries added after each tracked game.

Key rules already encoded:
- Always pull H2H game totals, not just team-wide trends
- Check quarter-by-quarter pace in H2H matchups (some teams run hot together)
- Pull actual player boxscores mid-game instead of assuming
- Flag early when Q1 total pace is >4.5 pts/min for under bets
- Close games (≤5 pts margin) = full starters = more scoring in Q4

## Setup

```bash
pip install -r requirements.txt
python daily_run.py          # Run daily update
python game_analysis.py PHX SAC  # Analyze a specific matchup
```

## Data Sources

- NBA CDN API (free, no key needed) — scores, box scores, player stats
- ESPN Game Feed — live play-by-play
- Rotowire/ESPN — injury reports

## Directory Structure

```
data/
  games/      → Historical game results + totals by season
  players/    → Per-player stat lines + prop hit rates
  h2h/        → Head-to-head matchup history (pace, totals, key players)
  props/      → Tracked prop recommendations + actual outcomes
  lessons/    → Lessons learned log
collectors/   → Data pipeline scripts
analyzers/    → Analysis modules
trackers/     → Bet outcome tracking
reports/      → Pre-game report generator
```

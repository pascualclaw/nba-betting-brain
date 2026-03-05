# NBA Betting Brain — Research & Architecture Notes

*Last updated: 2026-03-05 | Based on quant literature review*

---

## What Professional Betting Models Do That We Don't

### 1. Expected Value Gate (CRITICAL GAP)
Professional models only bet when the model shows ≥3% edge over market price.
We've been recommending bets based on "looks good" analysis with no EV gate.

**Formula:**
```
EV = (p × b) - (1 - p)
where p = model win probability, b = decimal odds - 1

Only bet if EV >= 0.03 (3% edge)
```

**Impact:** An 18-month quant study showed that applying a 3%+ edge threshold
significantly improved profitability. "Quality over quantity."

### 2. Kelly Criterion Bet Sizing (HIGH IMPACT)
We bet flat $100-$200. Professional approach: bet proportional to your edge.

**Formula:**
```
f = (bp - q) / b   [Full Kelly]
Use fractional Kelly: f_bet = 0.25 × f  [Quarter Kelly]

b = decimal odds - 1
p = model win probability
q = 1 - p
Max single bet: 5% of bankroll
```

**Impact:** In 18-month study, Kelly sizing (+42.5%) vs flat sizing (+28.3%) =
14% additional return on same model.

### 3. Possessions-Based Projection (MODEL GAP)
We use raw PPG averages. Better approach: project possessions × efficiency.

**Formula:**
```
Poss_expected = 0.5 × (Pace_home + Pace_away) + contextual_adj
Pts_home = Poss_expected × (ORtg_home / 100)
Pts_away = Poss_expected × (ORtg_away / 100)
Total = Pts_home + Pts_away
```

Why better: Separates PACE from EFFICIENCY. A slow-paced, efficient team
looks identical to a fast-paced, inefficient team in raw PPG, but they
have totally different betting profiles.

### 4. Dean Oliver Four Factors (FEATURE GAP)
These four metrics explain ~75% of wins and are NOT currently in our features:
- **eFG%** = (FGM + 0.5 × 3PM) / FGA — shooting efficiency
- **TOV%** = TOV / (FGA + 0.44×FTA + TOV) — ball security
- **ORB%** = ORB / (ORB + opp_DRB) — extra possessions
- **FT Rate** = FTA / FGA — free throw opportunities

Opponent-adjusted versions of these are the gold standard features for NBA models.

### 5. Closing Line Value (CLV) Tracking (VALIDATION GAP)
We never know if our picks are actually beating the market.

CLV = (closing_line - our_line) / our_line
- Positive CLV = we got a better number than the market closed at
- Consistently positive CLV = we have a real edge
- Zero/negative CLV = no edge, variance driving any wins

**The professional rule:** If you can't consistently beat the closing line,
you don't have edge. Your model needs to be fixed, not your bet selection.

### 6. Referee Tendency (NICHE BUT VALIDATED)
Referee assignments are public (NBA.com). Refs vary dramatically:
- High-foul refs = more free throws = higher totals (avg ±5-8 pts)
- Tight game refs = fewer fouls, faster games = lower totals
Top-10 feature in multiple studies.

### 7. Model Decay / Quarterly Retraining
"Performance degraded over time as markets adapted. Required quarterly retraining."
Our model was trained once on 5 seasons. Needs quarterly refresh.

### 8. Live Betting EV (NEW CAPABILITY)
Current market line + our model projection → live EV calculation.
This is the framework for evaluating live bets systematically, not by gut.

---

## Current Model Performance
- Linear Ridge: MAE 14.84 pts, 66.7% direction, ROI 30.85%
- GradientBoosting: MAE 15.27 pts, 65.6% direction, ROI 36.05%
- Top features: math_total_projection, pace_proxy_avg, h2h_total_avg

## Target After Improvements
- MAE target: < 12 pts (from better features)
- Direction target: > 68% (from Four Factors + opponent-adjusted metrics)
- ROI target: > 45% (from EV gate + Kelly sizing)
- CLV target: consistently positive (validates real edge)

---

## Implementation Roadmap

### Phase 1 (Immediate) — EV + Kelly
- `analyzers/ev_calculator.py` — EV gate + Kelly sizing
- `analyzers/clv_tracker.py` — CLV tracking for all bets

### Phase 2 (This week) — Feature Engineering
- `analyzers/four_factors.py` — eFG%, TOV%, ORB%, FT Rate from NBA API
- Update `training/historical_loader.py` — add Four Factors to training data
- Update `training/train.py` FEATURE_COLS — include Four Factors

### Phase 3 (Next week) — Advanced Features
- Possessions-based projection model
- Referee tendency database
- 3PA rate + opponent 3P defense rate
- Motivation factors (tank games, clinched playoff spots, rivalry games)

### Phase 4 (Ongoing) — System Maintenance
- Quarterly model retraining
- CLV review after each 50-bet sample
- Model decay monitoring

---

## Live Betting Framework (From Tonight's Lessons)
1. NEVER bet live without current score from ESPN API
2. Minimum edge threshold: 3% EV before any live bet
3. Kelly sizing for live bets (but cap at 2% of bankroll for high-variance live spots)
4. Flagrant foul on trailing team = immediate cash-out signal
5. Bottom-5 teams: no underdog spread unless within 4 pts
6. Live ML thresholds: Q2=8+, Q3=12+, Q4=15+ point lead

---

## Key Literature
- Klaassen & Magnus (2001): betting efficiency varies by sport
- Dean Oliver (Basketball on Paper, 2004): Four Factors framework
- Kelly (1956): optimal bet sizing via Kelly Criterion
- 18-month quant study (r/quant 2025): +42% Kelly vs +28% flat sizing
- Underdogchance.com (2025): Pace × Efficiency = total projection

---

## The Core Thesis
**NBA totals = possessions × efficiency.**
Any model that doesn't separate these two variables is leaving performance on the table.
Possessions are predictable (pace is sticky). Efficiency varies more (injuries, matchups, refs).
Get possessions right first, then model efficiency variance.

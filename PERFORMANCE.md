# NBA Betting Brain — Model Performance
**Version:** v20260305_0109 | **Generated:** 2026-03-05 01:11 ET

## Backtesting Results (Walk-Forward Validation)

| Model | MAE (pts) | RMSE | Direction Acc | Edge Acc (vs line) | N Games |
|-------|-----------|------|---------------|--------------------|---------|
| baseline_avg224 | 20.46 | 24.77 | 39.3% | N/A | 18408 |
| combined_pts_avg | 15.02 | 18.99 | 70.9% | N/A | 18408 |
| linear_ridge | 14.71 | 18.7 | 71.9% | N/A | 18408 |
| gradient_boosting | 14.85 | 18.77 | 71.7% | N/A | 18408 |

## Simulated Betting ROI

**baseline_avg224:** 65/100 (65.0%) | ROI: 24.09%
**combined_pts_avg:** 59/89 (66.3%) | ROI: 26.56%
**linear_ridge:** 62/93 (66.7%) | ROI: 27.27%
**gradient_boosting:** 64/94 (68.1%) | ROI: 29.98%

## Top Predictive Features (Gradient Boosting)

1. `math_total_projection`: 0.600 ████████████████████████████████████████████████████████████
2. `pace_proxy_avg`: 0.143 ██████████████
3. `h2h_total_avg`: 0.031 ███
4. `home_last5_pts_for`: 0.027 ██
5. `away_last5_pts_for`: 0.025 ██
6. `combined_pts_avg`: 0.023 ██
7. `home_pts_against`: 0.019 █
8. `home_net_rating`: 0.019 █
9. `away_net_rating`: 0.016 █
10. `away_pts_against`: 0.016 █

## Key Takeaways
- MAE < 8 pts = model is useful for identifying value bets
- Direction accuracy > 55% = edge over coin flip
- Edge accuracy > 55% vs posted line = real betting value
- ROI > 0% = model has real betting value

## Next Steps
- Accumulate more games with odds data to improve edge accuracy signal
- Add referee tendencies
- Calibrate confidence intervals
- Track line movement as a feature (sharp vs. square money signal)
## Spread Model — v20260305_1157

| Model | MAE | Direction | ATS Acc | Win Rate | ROI |
|-------|-----|-----------|---------|----------|-----|
| ridge | 10.32 pts | 65.4% | N/A | 67.2% | 28.3% |
| hist_gb | 10.36 pts | 64.6% | N/A | 66.6% | 27.2% |

**Top features:** net_rating_diff, home_net_rating, win_pct_diff, away_last5_pts_for, away_net_rating
**Home court average:** +2.3 pts

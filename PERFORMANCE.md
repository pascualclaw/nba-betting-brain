# NBA Betting Brain — Model Performance
**Version:** v20260304_0844 | **Generated:** 2026-03-04 08:44 ET

## Backtesting Results (Walk-Forward Validation)

| Model | MAE (pts) | RMSE | Direction Accuracy | N Games |
|-------|-----------|------|-------------------|---------|
| baseline_avg224 | 16.07 | 20.31 | 64.2% | 5452 |
| combined_pts_avg | 15.23 | 19.22 | 64.5% | 5452 |
| linear_ridge | 14.84 | 18.74 | 66.7% | 5452 |
| gradient_boosting | 15.27 | 19.25 | 65.6% | 5452 |

## Simulated Betting ROI

**baseline_avg224:** 65/100 (65.0%) | ROI: 24.09%
**combined_pts_avg:** 59/89 (66.3%) | ROI: 26.56%
**linear_ridge:** 61/89 (68.5%) | ROI: 30.85%
**gradient_boosting:** 62/87 (71.3%) | ROI: 36.05%

## Top Predictive Features (XGBoost)

1. `math_total_projection`: 0.185 ██████████████████
2. `pace_proxy_avg`: 0.099 █████████
3. `h2h_total_avg`: 0.076 ███████
4. `away_last5_pts_for`: 0.073 ███████
5. `home_last5_pts_for`: 0.071 ███████
6. `home_pts_against`: 0.066 ██████
7. `away_net_rating`: 0.059 █████
8. `away_pts_against`: 0.056 █████
9. `home_pts_for`: 0.054 █████
10. `net_rating_diff`: 0.054 █████

## Key Takeaways
- MAE < 8 pts = model is useful for identifying value bets
- Direction accuracy > 55% = edge over coin flip
- ROI > 0% = model has real betting value

## Next Steps
- Add injury impact scoring to features
- Add betting line (open/close) as feature when available
- Add referee tendencies
- Calibrate confidence intervals
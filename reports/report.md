# GARCH(1,1) Global Index Volatility Report

Data source: Yahoo Finance adjusted close data

## Key Finding

The most persistent volatility process is **S&P 500**, with alpha + beta = **0.960**.

## Model

```text
r_t = mu + epsilon_t
sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2
```

## Summary

| index | omega | alpha_arch | beta_garch | persistence_alpha_plus_beta | half_life_days | aic | bic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S&P 500 | 0.0521 | 0.1611 | 0.7986 | 0.9597 | 16.8497 | 6012.8285 | 6035.5695 |
| EURO STOXX 50 | 0.0868 | 0.1700 | 0.7667 | 0.9367 | 10.6064 | 6269.1966 | 6291.9321 |
| FTSE 100 | 0.0651 | 0.1758 | 0.7480 | 0.9238 | 8.7438 | 5330.7557 | 5353.5150 |
| Nikkei 225 | 0.1470 | 0.1562 | 0.7638 | 0.9200 | 8.3120 | 6907.9239 | 6930.5455 |

## Generated Charts

- `conditional_volatility.png`
- `persistence.png`


## Forecast Evaluation

One-day-ahead forecasts use an expanding training window. GARCH parameters are
re-estimated every 20 holdout observations; no future conditional volatility is used.
Lower values are better.

| index | method | RMSE | MAE | QLIKE |
| --- | --- | --- | --- | --- |
| EURO STOXX 50 | Historical | 2.5291 | 1.2052 | 0.9870 |
| EURO STOXX 50 | EWMA | 2.4466 | 1.1958 | 0.9456 |
| EURO STOXX 50 | GARCH(1,1) | 2.3215 | 1.1722 | 0.8933 |
| FTSE 100 | Historical | 1.8744 | 0.7101 | 0.3028 |
| FTSE 100 | EWMA | 1.7962 | 0.6934 | 0.3120 |
| FTSE 100 | GARCH(1,1) | 1.6765 | 0.6576 | 0.2200 |
| Nikkei 225 | Historical | 6.7759 | 3.2887 | 1.9412 |
| Nikkei 225 | EWMA | 6.5840 | 3.2278 | 1.9586 |
| Nikkei 225 | GARCH(1,1) | 6.3497 | 2.9109 | 1.8996 |
| S&P 500 | Historical | 4.7137 | 1.3638 | 1.0220 |
| S&P 500 | EWMA | 4.5959 | 1.3153 | 0.9149 |
| S&P 500 | GARCH(1,1) | 4.5280 | 1.3013 | 0.8256 |

### Best Method by Metric

- **EURO STOXX 50**: RMSE — GARCH(1,1), MAE — GARCH(1,1), QLIKE — GARCH(1,1)
- **FTSE 100**: RMSE — GARCH(1,1), MAE — GARCH(1,1), QLIKE — GARCH(1,1)
- **Nikkei 225**: RMSE — GARCH(1,1), MAE — GARCH(1,1), QLIKE — GARCH(1,1)
- **S&P 500**: RMSE — GARCH(1,1), MAE — GARCH(1,1), QLIKE — GARCH(1,1)

# GARCH(1,1) Global Index Volatility Report

Data source: deterministic demo data

## Key Finding

The most persistent volatility process in this run is **FTSE 100** with alpha + beta = **0.983**.

## Model

Daily log returns are modeled with:

```text
r_t = mu + epsilon_t
sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2
```

The persistence metric is `alpha + beta`. Values close to 1 indicate volatility shocks fade slowly.

## Summary

| index | omega | alpha_arch | beta_garch | persistence_alpha_plus_beta | half_life_days | aic | bic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FTSE 100 | 0.0348 | 0.0563 | 0.9268 | 0.9831 | 40.5706 | 7880.8647 | 7903.7589 |
| S&P 500 | 0.0271 | 0.0525 | 0.9293 | 0.9819 | 37.8772 | 7186.4524 | 7209.3466 |
| Nikkei 225 | 0.0495 | 0.0733 | 0.9067 | 0.9799 | 34.1969 | 8223.1791 | 8246.0733 |
| EURO STOXX 50 | 0.0466 | 0.0473 | 0.9146 | 0.9618 | 17.8194 | 6821.8797 | 6844.7740 |

## Generated Charts


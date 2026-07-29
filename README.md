# GARCH Model for Global Indices

Estimate **GARCH** volatility models for major global equity indices and compare how volatility **persists** across markets.

Markets covered (Yahoo Finance):

| Index | Ticker |
|-------|--------|
| S&P 500 | `^GSPC` |
| EURO STOXX 50 | `^STOXX50E` |
| Nikkei 225 | `^N225` |
| FTSE 100 | `^FTSE` |

## Why this project

- Introduces students to **volatility modeling** beyond linear regression  
- Links to risk applications: **VaR**, derivatives pricing, portfolio allocation  
- Strong portfolio piece for **quant finance / risk management** roles  

## Project layout

```
garch_global_indices/
├── data/                     # cached Yahoo Finance prices
├── src/
│   ├── data_loader.py        # download prices, log / simple returns
│   ├── analysis.py           # ADF/KPSS, GARCH MLE, IC, forecast, VaR
│   ├── visualization.py      # EDA + conditional-vol / forecast plots
│   └── utils.py
├── notebooks/
│   └── explore.ipynb         # optional interactive walkthrough
├── results/
│   ├── figures/              # prices, returns, cond. vol, persistence, forecast
│   └── tables/               # stats, params, AIC/BIC, diagnostics
├── config.py                 # tickers, dates, orders, forecast horizon
├── main.py                   # full research pipeline
└── requirements.txt
```

## Setup

```bash
cd garch_global_indices
pip install -r requirements.txt
python main.py
```

First run downloads daily adjusted closes via `yfinance` and caches them under `data/global_indices_adj_close.csv`.

If Yahoo Finance rate-limits your IP, the loader automatically falls back to a **synthetic GARCH-like index panel** so the full research workflow still runs offline. Delete the cache and re-run later for live data:

```bash
del data\global_indices_adj_close.csv
python main.py
```

## Model

The **GARCH(p, q)** conditional variance is

$$
\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2
$$

- **ω** — constant (long-run level contribution)  
- **ARCH terms (α)** — reaction to recent squared shocks / “news”  
- **GARCH terms (β)** — persistence of past conditional variance  

For the workhorse **GARCH(1,1)**, **volatility persistence** is measured by **α + β** (values near 1 imply slow mean reversion of volatility).

Estimation uses **maximum likelihood** via the [`arch`](https://arch.readthedocs.io/) package. Log returns are scaled by 100 (percent) for stable MLE, as is common in applied work.

## Workflow (`main.py`)

1. **Import** adjusted closes for the four indices  
2. **Returns** — log returns and simple returns  
3. **EDA** — prices, returns, rolling volatility, distributions (volatility clustering)  
4. **Stationarity** — ADF and KPSS on returns  
5. **Fit GARCH(1,1)** — Normal innovations  
6. **Model selection** — compare orders `(1,1), (1,2), (2,1), (2,2)` with **AIC / BIC**  
7. **Student-t** — optional heavier tails vs Normal  
8. **Diagnostics** — Ljung–Box on standardized residuals and squared residuals  
9. **Forecast** — multi-step conditional SD (blue = history, red = forecast)  
10. **Compare** persistence **α + β** across regions  

## Research questions

| Question | Where to look |
|----------|----------------|
| Which index has the highest sample volatility? | `summary_statistics.csv` (`ann_vol`) |
| Which market has the most persistent volatility (α+β closest to 1)? | `garch11_normal_params.csv`, `06_persistence_comparison.png` |
| Clear spikes in known crises (e.g. COVID-2020)? | `02_log_returns.png`, `05_conditional_volatility.png` |
| Does Student-t beat Normal? | `normal_vs_student_t.csv` |
| Are std. residuals still autocorrelated? | `ljung_box_std_residuals.csv` |
| How do dynamics differ by region (US / EU / JP / UK)? | persistence + conditional-vol figures |

## Expected outputs

**Figures**

- Normalized prices, log returns, return distributions  
- Rolling realized volatility  
- GARCH conditional volatility (all markets)  
- Persistence bar chart (α + β)  
- Forecast plot in the classic “Prediction based on GARCH model” style  

**Tables**

- Summary statistics  
- Stationarity tests  
- GARCH parameters (ω, α, β, α+β)  
- AIC / BIC across orders  
- Normal vs Student-t comparison  
- Ljung–Box diagnostics  
- `research_answers.json` — short Q&A snapshot from the latest run  

## Tunable parameters (`config.py`)

| Parameter | Default | Notes |
|-----------|---------|--------|
| `START_DATE` | `2015-01-01` | sample start |
| `RETURN_SCALE` | `100` | percent log returns |
| `ROLLING_WINDOW` | `21` | ~1 month realized vol |
| `GARCH_ORDERS` | `(1,1)…(2,2)` | IC grid |
| `FORECAST_HORIZON` | `30` | trading days |
| `COMPARE_STUDENT_T` | `True` | fat-tail comparison |

## Dependencies

`pandas`, `numpy`, `yfinance`, `matplotlib`, `seaborn`, `arch`, `statsmodels`, `scipy`

## License

MIT — feel free to use for coursework and portfolio demos.

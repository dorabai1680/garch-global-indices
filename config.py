"""Default settings for the GARCH global-indices study."""

from __future__ import annotations

# Yahoo Finance tickers for major equity indices
INDICES: dict[str, str] = {
    "S&P 500": "^GSPC",
    "EURO STOXX 50": "^STOXX50E",
    "Nikkei 225": "^N225",
    "FTSE 100": "^FTSE",
}

START_DATE = "2015-01-01"
END_DATE = None  # None = through most recent available close

# Scale log returns by 100 (percent) for numerically stable GARCH MLE
RETURN_SCALE = 100.0

# Rolling window (trading days) for realized / rolling volatility EDA
ROLLING_WINDOW = 21

# GARCH(1,1) is the baseline; grid used for AIC/BIC model selection
GARCH_ORDERS: list[tuple[int, int]] = [(1, 1), (1, 2), (2, 1), (2, 2)]

# Forecast horizon (trading days) for conditional-volatility prediction plot
FORECAST_HORIZON = 30

# Optional Student-t vs Normal comparison
COMPARE_STUDENT_T = True

# VaR backtest (optional)
VAR_CONFIDENCE = 0.05
VAR_WINDOW = 252

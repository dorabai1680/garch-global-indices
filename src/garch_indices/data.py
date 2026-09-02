from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import IndexConfig


def fetch_yahoo_prices(index: IndexConfig, start: date, end: date) -> pd.DataFrame:
    via_yfinance = _fetch_with_yfinance(index, start, end)
    return via_yfinance if via_yfinance is not None else _fetch_with_chart_api(index, start, end)


def _fetch_with_yfinance(index: IndexConfig, start: date, end: date) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    raw = yf.download(index.ticker, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"Yahoo Finance returned no prices for {index.ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(-1, axis=1)
    column = "Adj Close" if "Adj Close" in raw.columns else "Close"
    frame = raw.reset_index().rename(columns={"Date": "date", column: "adj_close"})
    if isinstance(frame["adj_close"], pd.DataFrame):
        frame["adj_close"] = frame["adj_close"].iloc[:, 0]
    frame["index"], frame["ticker"] = index.name, index.ticker
    return frame[["date", "index", "ticker", "adj_close"]].dropna().sort_values("date").reset_index(drop=True)


def _fetch_with_chart_api(index: IndexConfig, start: date, end: date) -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(index.ticker)}"
        f"?period1={_unix_timestamp(start)}&period2={_unix_timestamp(end)}&interval=1d&events=history"
    )
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance error for {index.ticker}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise ValueError(f"Yahoo Finance returned no prices for {index.ticker}")
    indicators = result.get("indicators", {})
    quote_data = (indicators.get("adjclose") or indicators.get("quote") or [{}])[0]
    closes = quote_data.get("adjclose") or quote_data.get("close") or []
    frame = pd.DataFrame({
        "date": pd.to_datetime(result.get("timestamp") or [], unit="s").tz_localize(None),
        "index": index.name,
        "ticker": index.ticker,
        "adj_close": closes,
    })
    return frame.dropna().sort_values("date").reset_index(drop=True)


def load_or_download_prices(indices, start: date, end: date, data_dir: Path, *, force_download=False):
    data_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for index in indices:
        cache_path = data_dir / f"{_slug(index.name)}.csv"
        if cache_path.exists() and not force_download:
            cached = _normalize_price_frame(pd.read_csv(cache_path), index)
            mask = (cached["date"].dt.date >= start) & (cached["date"].dt.date < end)
            cached = cached.loc[mask].reset_index(drop=True)
            if cached.empty:
                raise ValueError(f"Cached data for {index.name} does not cover the requested period")
            result[index.name] = cached
        else:
            prices = fetch_yahoo_prices(index, start, end)
            prices.to_csv(cache_path, index=False)
            result[index.name] = prices
    return result


def generate_demo_prices(indices, start: date, end: date):
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 61:
        raise ValueError("The selected period must contain at least 61 business days.")
    parameters = {
        "S&P 500": (0.035, 0.055, 0.925),
        "EURO STOXX 50": (0.045, 0.070, 0.900),
        "Nikkei 225": (0.050, 0.065, 0.910),
        "FTSE 100": (0.030, 0.050, 0.935),
    }
    rng = np.random.default_rng(20260727)
    generated = {}
    for index in indices:
        omega, alpha, beta = parameters[index.name]
        returns = np.zeros(len(dates))
        sigma2 = np.full(len(dates), omega / max(1 - alpha - beta, 0.02))
        shocks = rng.standard_normal(len(dates))
        for t in range(1, len(dates)):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
            returns[t] = np.sqrt(max(sigma2[t], 1e-8)) * shocks[t]
        generated[index.name] = pd.DataFrame({
            "date": dates, "index": index.name, "ticker": index.ticker,
            "adj_close": 100 * np.exp(np.cumsum(returns / 100)),
        })
    return generated


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values("date").copy()
    frame["return_pct"] = np.log(frame["adj_close"]).diff() * 100
    return frame.dropna(subset=["return_pct"]).reset_index(drop=True)


def _normalize_price_frame(frame: pd.DataFrame, index: IndexConfig) -> pd.DataFrame:
    normalized = frame.copy()
    normalized = normalized.rename(columns={"Date": "date", "Adj Close": "adj_close"})
    if "adj_close" not in normalized and "Close" in normalized:
        normalized = normalized.rename(columns={"Close": "adj_close"})
    if not {"date", "adj_close"}.issubset(normalized.columns):
        raise ValueError(f"{index.name} cache must contain date/adj_close or Yahoo Date/Adj Close columns.")
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["index"], normalized["ticker"] = index.name, index.ticker
    return normalized[["date", "index", "ticker", "adj_close"]].dropna().sort_values("date").reset_index(drop=True)


def _unix_timestamp(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _slug(value: str) -> str:
    return value.lower().replace("&", "and").replace(" ", "_")

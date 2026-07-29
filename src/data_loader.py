"""Download and cache daily adjusted closes for global equity indices."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# Eastmoney symbols via akshare (works when Yahoo is blocked / rate-limited)
AKSHARE_SYMBOLS: dict[str, str] = {
    "S&P 500": "标普500",
    "EURO STOXX 50": "欧洲斯托克50",
    "Nikkei 225": "日经225",
    "FTSE 100": "英国富时100",
}


def _download_akshare(tickers: dict[str, str], start: str) -> pd.DataFrame:
    """Fetch daily closes via akshare ``index_global_hist_em``."""
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise ImportError("akshare is required for the non-Yahoo fallback") from exc

    frames: dict[str, pd.Series] = {}
    for name in tickers:
        em_sym = AKSHARE_SYMBOLS.get(name)
        if em_sym is None:
            continue
        df = ak.index_global_hist_em(symbol=em_sym)
        cols = df.columns.tolist()
        date_col, close_col = cols[0], cols[4]  # 日期, 最新价/收盘
        s = df[[date_col, close_col]].copy()
        s[date_col] = pd.to_datetime(s[date_col])
        s = s.set_index(date_col)[close_col].astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.name = name
        frames[name] = s
    if not frames:
        return pd.DataFrame()
    prices = pd.DataFrame(frames).sort_index()
    return prices.loc[start:]


def _extract_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            closes = raw["Close"].copy()
        else:
            closes = raw.xs("Close", axis=1, level=0, drop_level=True)
    else:
        closes = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
        if closes.shape[1] == 1 and len(symbols) == 1:
            closes.columns = symbols
    return closes


def _download_one(symbol: str, start: str, end: str | None, retries: int = 3) -> pd.Series:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            part = _extract_close(raw, [symbol])
            if not part.empty and part.shape[1] >= 1:
                s = part.iloc[:, 0].dropna()
                s.name = symbol
                if len(s) > 50:
                    return s
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(2.0 * (attempt + 1))
    if last_exc:
        warnings.warn(f"Failed to download {symbol}: {last_exc}")
    return pd.Series(dtype=float, name=symbol)


def make_synthetic_index_prices(
    names: list[str],
    start: str = "2015-01-01",
    n_days: int = 2200,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Synthetic equity-index levels with GARCH(1,1)-like return dynamics.

    Used when Yahoo Finance is rate-limited so the full research pipeline
    still runs offline. Persistence α+β differs by market for demo contrast.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # (omega, alpha, beta, mu, start_level) — higher α+β ⇒ more persistence
    presets = {
        "S&P 500": (0.02, 0.08, 0.90, 0.03, 2000.0),
        "EURO STOXX 50": (0.03, 0.10, 0.86, 0.02, 3000.0),
        "Nikkei 225": (0.04, 0.12, 0.84, 0.02, 16000.0),
        "FTSE 100": (0.025, 0.09, 0.88, 0.025, 6500.0),
    }
    defaults = [
        (0.03, 0.10, 0.85, 0.02, 1000.0),
        (0.04, 0.12, 0.82, 0.015, 1200.0),
        (0.02, 0.07, 0.91, 0.03, 1500.0),
        (0.035, 0.11, 0.84, 0.02, 800.0),
    ]

    data: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        omega, alpha, beta, mu, level0 = presets.get(name, defaults[i % len(defaults)])
        eps = np.zeros(n_days)
        sig2 = np.zeros(n_days)
        sig2[0] = omega / max(1e-8, (1 - alpha - beta))
        for t in range(n_days):
            if t > 0:
                sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
            eps[t] = rng.normal(0, np.sqrt(max(sig2[t], 1e-8)))
        # percent shocks → price path; inject a COVID-like spike cluster
        rets = mu / 252 + eps / 100.0
        crisis = slice(int(0.45 * n_days), int(0.48 * n_days))
        rets[crisis] *= 2.5
        prices = level0 * np.exp(np.cumsum(rets))
        data[name] = prices

    return pd.DataFrame(data, index=idx)


def download_prices(
    tickers: dict[str, str],
    start: str = "2015-01-01",
    end: str | None = None,
    cache_path: Path | None = None,
    allow_synthetic: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices from Yahoo Finance (one ticker at a time).

    Falls back to synthetic GARCH-like index levels if live download fails.
    """
    if cache_path is not None and cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices.sort_index()

    if yf is None:
        raise ImportError("yfinance is required; pip install yfinance")

    series_list: list[pd.Series] = []
    for name, symbol in tickers.items():
        s = _download_one(symbol, start=start, end=end)
        if not s.empty:
            s.name = name
            series_list.append(s)
        time.sleep(1.5)

    if series_list:
        closes = pd.concat(series_list, axis=1).sort_index()
        closes = closes.dropna(how="all")
        keep = [c for c in closes.columns if closes[c].notna().sum() > 200]
        closes = closes[keep]
        if len(keep) >= 2:
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                closes.to_csv(cache_path)
            return closes

    # Fallback 1: akshare / Eastmoney (often reachable when Yahoo is blocked)
    try:
        closes = _download_akshare(tickers, start=start)
        keep = [c for c in closes.columns if closes[c].notna().sum() > 200]
        closes = closes[keep]
        if len(keep) >= 2:
            warnings.warn(
                "Yahoo Finance unavailable; loaded live prices via akshare (Eastmoney).",
                UserWarning,
                stacklevel=2,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                closes.to_csv(cache_path)
            return closes
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"akshare fallback failed: {exc}", UserWarning, stacklevel=2)

    if not allow_synthetic:
        raise RuntimeError(
            "No usable price data from Yahoo or akshare. "
            "Check network access / rate limits, or delete a stale cache and retry."
        )

    warnings.warn(
        "Live data unavailable; using synthetic GARCH index panel. "
        "Delete the cache CSV and re-run later for live data.",
        UserWarning,
        stacklevel=2,
    )
    closes = make_synthetic_index_prices(list(tickers.keys()), start=start)
    if cache_path is not None:
        synth_path = cache_path.with_name(cache_path.stem + "_synthetic.csv")
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        closes.to_csv(synth_path)
        closes.to_csv(cache_path)
    return closes


def log_returns(prices: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """Log returns r_t = scale * ln(P_t / P_{t-1})."""
    return scale * np.log(prices / prices.shift(1)).dropna(how="all")


def simple_returns(prices: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """Simple returns R_t = scale * (P_t / P_{t-1} - 1)."""
    return scale * prices.pct_change().dropna(how="all")


def load_dataset(
    tickers: dict[str, str],
    start: str = "2015-01-01",
    end: str | None = None,
    cache_path: Path | None = None,
    return_scale: float = 100.0,
    allow_synthetic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (prices, log_returns, simple_returns)."""
    prices = download_prices(
        tickers,
        start=start,
        end=end,
        cache_path=cache_path,
        allow_synthetic=allow_synthetic,
    )
    lr = log_returns(prices, scale=return_scale)
    sr = simple_returns(prices, scale=return_scale)
    return prices, lr, sr

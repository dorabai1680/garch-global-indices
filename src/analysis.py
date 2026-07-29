"""Stationarity tests, GARCH estimation, persistence, and VaR helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class GarchFitResult:
    name: str
    order: tuple[int, int]
    dist: str
    omega: float
    alphas: list[float]
    betas: list[float]
    persistence: float
    aic: float
    bic: float
    loglik: float
    nu: float | None
    conditional_vol: pd.Series
    std_resid: pd.Series
    resid: pd.Series
    model_result: Any


def adf_test(series: pd.Series, significance: float = 0.05) -> dict[str, float | bool | str]:
    """Augmented Dickey–Fuller test for a unit root (H0: non-stationary)."""
    clean = series.dropna()
    stat, pvalue, usedlag, nobs, crit, _ = adfuller(clean, autolag="AIC")
    return {
        "test": "ADF",
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "used_lag": int(usedlag),
        "nobs": int(nobs),
        "reject_unit_root": bool(pvalue < significance),
        "stationary_at_5pct": bool(pvalue < significance),
        "crit_1pct": float(crit["1%"]),
        "crit_5pct": float(crit["5%"]),
        "crit_10pct": float(crit["10%"]),
    }


def kpss_test(series: pd.Series, significance: float = 0.05) -> dict[str, float | bool | str]:
    """KPSS test (H0: stationary)."""
    clean = series.dropna()
    stat, pvalue, lags, crit = kpss(clean, regression="c", nlags="auto")
    return {
        "test": "KPSS",
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "lags": int(lags),
        "reject_stationarity": bool(pvalue < significance),
        "stationary_at_5pct": bool(pvalue >= significance),
        "crit_1pct": float(crit["1%"]),
        "crit_5pct": float(crit["5%"]),
        "crit_10pct": float(crit["10%"]),
    }


def stationarity_table(returns: pd.DataFrame) -> pd.DataFrame:
    """ADF + KPSS summary for each return series."""
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        adf = adf_test(s)
        kps = kpss_test(s)
        rows.append(
            {
                "index": col,
                "adf_stat": adf["statistic"],
                "adf_pvalue": adf["pvalue"],
                "adf_stationary": adf["stationary_at_5pct"],
                "kpss_stat": kps["statistic"],
                "kpss_pvalue": kps["pvalue"],
                "kpss_stationary": kps["stationary_at_5pct"],
            }
        )
    return pd.DataFrame(rows).set_index("index")


def summary_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for log returns (already scaled, e.g. percent)."""
    out = returns.describe().T
    out["skew"] = returns.skew()
    out["kurtosis"] = returns.kurtosis()  # excess kurtosis (Fisher)
    out["ann_vol"] = returns.std() * np.sqrt(252)
    return out


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    mean: str = "Zero",
) -> GarchFitResult:
    """
    Fit a GARCH(p, q) model by MLE via the `arch` package.

      σ²_t = ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}

    Persistence for GARCH(1,1) is α + β; for higher orders it is Σα + Σβ.
    """
    series = returns.dropna()
    am = arch_model(series, mean=mean, vol="GARCH", p=p, q=q, dist=dist, rescale=False)
    res = am.fit(disp="off", show_warning=False)

    params = res.params
    omega = float(params.get("omega", np.nan))
    alphas = [float(params[k]) for k in params.index if k.startswith("alpha")]
    betas = [float(params[k]) for k in params.index if k.startswith("beta")]
    persistence = float(sum(alphas) + sum(betas))
    nu = float(params["nu"]) if "nu" in params.index else None

    cond_vol = pd.Series(res.conditional_volatility, index=series.index, name=f"{series.name}_cond_vol")
    std_resid = pd.Series(res.std_resid, index=series.index, name=f"{series.name}_std_resid")
    resid = pd.Series(res.resid, index=series.index, name=f"{series.name}_resid")

    return GarchFitResult(
        name=str(series.name),
        order=(p, q),
        dist=dist,
        omega=omega,
        alphas=alphas,
        betas=betas,
        persistence=persistence,
        aic=float(res.aic),
        bic=float(res.bic),
        loglik=float(res.loglikelihood),
        nu=nu,
        conditional_vol=cond_vol,
        std_resid=std_resid,
        resid=resid,
        model_result=res,
    )


def select_garch_order(
    returns: pd.Series,
    orders: list[tuple[int, int]],
    dist: str = "normal",
) -> tuple[pd.DataFrame, GarchFitResult]:
    """Fit several (p, q) orders; return IC table and best-by-AIC fit."""
    rows = []
    fits: list[GarchFitResult] = []
    for p, q in orders:
        try:
            fit = fit_garch(returns, p=p, q=q, dist=dist)
            fits.append(fit)
            rows.append(
                {
                    "index": returns.name,
                    "p": p,
                    "q": q,
                    "dist": dist,
                    "aic": fit.aic,
                    "bic": fit.bic,
                    "loglik": fit.loglik,
                    "persistence": fit.persistence,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "index": returns.name,
                    "p": p,
                    "q": q,
                    "dist": dist,
                    "aic": np.nan,
                    "bic": np.nan,
                    "loglik": np.nan,
                    "persistence": np.nan,
                    "error": str(exc),
                }
            )
    table = pd.DataFrame(rows)
    if not fits:
        raise RuntimeError(f"No GARCH model converged for {returns.name}")
    best = min(fits, key=lambda f: f.aic)
    return table, best


def fit_all_indices(
    returns: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> dict[str, GarchFitResult]:
    """Fit the same GARCH(p,q) specification to every column."""
    return {col: fit_garch(returns[col], p=p, q=q, dist=dist) for col in returns.columns}


def parameters_table(fits: dict[str, GarchFitResult]) -> pd.DataFrame:
    """ω, α, β, α+β (and ν if Student-t) for each index."""
    rows = []
    for name, fit in fits.items():
        row = {
            "index": name,
            "order": f"GARCH{fit.order}",
            "dist": fit.dist,
            "omega": fit.omega,
            "alpha": fit.alphas[0] if fit.alphas else np.nan,
            "beta": fit.betas[0] if fit.betas else np.nan,
            "persistence_alpha_plus_beta": fit.persistence,
            "aic": fit.aic,
            "bic": fit.bic,
            "loglik": fit.loglik,
        }
        if fit.nu is not None:
            row["nu"] = fit.nu
        # Extra alphas/betas for higher-order models
        for i, a in enumerate(fit.alphas):
            row[f"alpha[{i+1}]"] = a
        for j, b in enumerate(fit.betas):
            row[f"beta[{j+1}]"] = b
        rows.append(row)
    return pd.DataFrame(rows).set_index("index")


def ljung_box_std_resid(fit: GarchFitResult, lags: int = 10) -> dict[str, float]:
    """Ljung–Box on standardized residuals and squared standardized residuals."""
    sr = fit.std_resid.dropna()
    lb = acorr_ljungbox(sr, lags=[lags], return_df=True)
    lb2 = acorr_ljungbox(sr**2, lags=[lags], return_df=True)
    return {
        "index": fit.name,
        "lb_resid_stat": float(lb["lb_stat"].iloc[0]),
        "lb_resid_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "lb_sq_resid_stat": float(lb2["lb_stat"].iloc[0]),
        "lb_sq_resid_pvalue": float(lb2["lb_pvalue"].iloc[0]),
        "lags": lags,
    }


def forecast_conditional_vol(fit: GarchFitResult, horizon: int = 30) -> pd.Series:
    """h-step ahead conditional volatility forecast (annualization not applied)."""
    fcast = fit.model_result.forecast(horizon=horizon, reindex=False)
    # variance forecast: columns h.1 ... h.h
    var = fcast.variance.iloc[-1]
    vol = np.sqrt(var.values.astype(float))
    last_date = fit.conditional_vol.index[-1]
    # Use business-day offsets for the forecast axis
    idx = pd.bdate_range(start=last_date, periods=horizon + 1, freq="B")[1:]
    return pd.Series(vol, index=idx[: len(vol)], name="forecast_vol")


def rolling_var_backtest(
    returns: pd.Series,
    confidence: float = 0.05,
    window: int = 252,
) -> pd.DataFrame:
    """
    Simple 1-day VaR backtest using rolling GARCH(1,1) Normal quantile.

    Slow but illustrative; uses expanding-window fits every day after `window`.
    For speed we refit every 5 days and forward-fill VaR.
    """
    series = returns.dropna()
    dates = series.index[window:]
    records = []
    last_var = np.nan
    for i, dt in enumerate(dates):
        if i % 5 == 0 or np.isnan(last_var):
            hist = series.loc[:dt].iloc[:-1].tail(window)
            try:
                fit = fit_garch(hist, p=1, q=1, dist="normal")
                sigma = float(fit.conditional_vol.iloc[-1])
                # one-step variance mean-reversion update using last residual
                # Use model forecast horizon=1
                f = fit.model_result.forecast(horizon=1)
                sigma_f = float(np.sqrt(f.variance.iloc[-1, 0]))
                from scipy.stats import norm

                last_var = float(norm.ppf(confidence) * sigma_f)
            except Exception:  # noqa: BLE001
                last_var = np.nan
        realized = float(series.loc[dt])
        records.append(
            {
                "date": dt,
                "return": realized,
                "var": last_var,
                "breach": bool(realized < last_var) if not np.isnan(last_var) else False,
            }
        )
    df = pd.DataFrame(records).set_index("date")
    if len(df):
        df.attrs["hit_rate"] = float(df["breach"].mean())
        df.attrs["expected_rate"] = confidence
    return df

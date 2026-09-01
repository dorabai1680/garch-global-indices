from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd

from .garch import GarchResult


def build_summary(results: dict[str, GarchResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "index": name,
                "omega": result.omega,
                "alpha_arch": result.alpha,
                "beta_garch": result.beta,
                "persistence_alpha_plus_beta": result.persistence,
                "half_life_days": result.half_life_days,
                "aic": result.aic,
                "bic": result.bic,
            }
        )
    return pd.DataFrame(rows).sort_values("persistence_alpha_plus_beta", ascending=False)


def plot_conditional_volatility(series: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    for name, frame in series.items():
        axis.plot(frame["date"], frame["conditional_volatility"], label=name, linewidth=1.3)
    axis.set_title("GARCH(1,1) Conditional Volatility")
    axis.set_ylabel("Daily volatility (%)")
    axis.set_xlabel("Date")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_persistence(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(summary["index"], summary["persistence_alpha_plus_beta"], color=["#245b86", "#d05a48", "#2d7d64", "#8a6f2a"])
    axis.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
    axis.set_ylim(0, max(1.05, float(summary["persistence_alpha_plus_beta"].max()) + 0.02))
    axis.set_title("Volatility Persistence by Index")
    axis.set_ylabel("alpha + beta")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_markdown_report(summary: pd.DataFrame, output_path: Path, *, demo_mode: bool) -> None:
    top = summary.iloc[0]
    source_note = "deterministic demo data" if demo_mode else "Yahoo Finance adjusted close data"
    markdown = f"""# GARCH(1,1) Global Index Volatility Report

Data source: {source_note}

## Key Finding

The most persistent volatility process in this run is **{top['index']}** with alpha + beta = **{top['persistence_alpha_plus_beta']:.3f}**.

## Model

Daily log returns are modeled with:

```text
r_t = mu + epsilon_t
sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2
```

The persistence metric is `alpha + beta`. Values close to 1 indicate volatility shocks fade slowly.

## Summary

{_markdown_table(summary)}

## Generated Charts

"""
    output_path.write_text(markdown, encoding="utf-8")

def evaluate_forecasts(
    prices_by_index: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    initial_train: int = 252,
    ewma_lambda: float = 0.94,
    step: int = 5,
) -> pd.DataFrame:
    """Run a simple out-of-sample 1-step-ahead forecast comparison.

    The `step` parameter controls evaluation frequency (e.g., step=5 evaluates every
    5th observation) to reduce the number of GARCH refits and speed up evaluation.
    """
    rows = []
    for name, prices in prices_by_index.items():
        returns = prices.copy()
        if "return_pct" not in returns.columns:
            # expects a column 'return_pct' (decimal, not %)
            returns["return_pct"] = (returns["adj_close"].pct_change())
        series = returns["return_pct"].dropna().to_numpy()
        n = series.size
        if n < 60:
            continue

        train0 = min(initial_train, max(60, n // 3))
        realized = []
        forecasts = {"historical": [], "ewma": [], "garch": []}

        # Precompute EWMA initial variance on first train0
        for t in range(train0, n - 1, step):
            train = series[:t]
            # realized variance at t+1 (one-step ahead) use squared return
            rv = float(series[t + 1] ** 2)
            realized.append(rv)

            # Historical rolling window (last 20 days)
            window = min(20, train.size)
            hist_var = float(train[-window:].var(ddof=1)) if train.size > 1 else float(train.var())
            forecasts["historical"].append(max(hist_var, 1e-12))

            # EWMA: compute on training window
            ewma_var = train[-1] ** 2 if train.size == 1 else float(((1 - ewma_lambda) * (train ** 2)).sum())
            # a simple recursive EWMA estimate (one-liner fallback)
            s = train[0] ** 2
            for x in train[1:]:
                s = ewma_lambda * s + (1 - ewma_lambda) * (x ** 2)
            ewma_var = float(s)
            forecasts["ewma"].append(max(ewma_var, 1e-12))

            # GARCH: fit on train and compute 1-step forecast
            try:
                from .garch import fit_garch11

                gr = fit_garch11(train)
                last_sigma2 = float(gr.conditional_volatility[-1] ** 2)
                last_resid = float((train[-1] - gr.mu) ** 2)
                garch_forecast = float(gr.omega + gr.alpha * (train[-1] - gr.mu) ** 2 + gr.beta * last_sigma2)
                forecasts["garch"].append(max(garch_forecast, 1e-12))
            except Exception:
                forecasts["garch"].append(float("nan"))

        # compute metrics
        import numpy as _np

        def _metrics(farr):
            f = _np.asarray(farr, dtype=float)
            r = _np.asarray(realized, dtype=float)
            mask = _np.isfinite(f) & _np.isfinite(r)
            if not mask.any():
                return {"rmse": _np.nan, "mae": _np.nan, "qlike": _np.nan}
            f = f[mask]
            r = r[mask]
            rmse = float(_np.sqrt(_np.mean((f - r) ** 2)))
            mae = float(_np.mean(_np.abs(f - r)))
            # QLIKE: mean( log(f) + r / f )
            qlike = float(_np.mean(_np.log(f) + r / f))
            return {"rmse": rmse, "mae": mae, "qlike": qlike}

        for method in ("historical", "ewma", "garch"):
            mets = _metrics(forecasts[method])
            rows.append(
                {
                    "index": name,
                    "method": method,
                    "rmse": mets["rmse"],
                    "mae": mets["mae"],
                    "qlike": mets["qlike"],
                }
            )

        df = pd.DataFrame(rows)
    if not df.empty:
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path / "forecast_evaluation.csv", index=False)

        # also write aggregated summary (mean metrics per index and method)
        summary = (
            df.groupby(["index", "method"]).agg({"rmse": "mean", "mae": "mean", "qlike": "mean"}).reset_index()
        )
        summary.to_csv(output_path / "results_summary.csv", index=False)
    return df

def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")

    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

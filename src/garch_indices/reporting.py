from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd

from .garch import GarchResult


def build_summary(results: dict[str, GarchResult]) -> pd.DataFrame:
    rows = [{
        "index": name, "omega": result.omega, "alpha_arch": result.alpha,
        "beta_garch": result.beta, "persistence_alpha_plus_beta": result.persistence,
        "half_life_days": result.half_life_days, "aic": result.aic, "bic": result.bic,
    } for name, result in results.items()]
    return pd.DataFrame(rows).sort_values("persistence_alpha_plus_beta", ascending=False)


def plot_conditional_volatility(series, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    for name, frame in series.items():
        axis.plot(frame["date"], frame["conditional_volatility"], label=name, linewidth=1.3)
    axis.set(title="GARCH(1,1) Conditional Volatility", ylabel="Daily volatility (%)", xlabel="Date")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_persistence(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(summary["index"], summary["persistence_alpha_plus_beta"])
    axis.axhline(1, color="#333333", linewidth=1, linestyle="--")
    axis.set_ylim(0, max(1.05, float(summary["persistence_alpha_plus_beta"].max()) + 0.02))
    axis.set(title="Volatility Persistence by Index", ylabel="alpha + beta")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_markdown_report(summary: pd.DataFrame, output_path: Path, *, demo_mode: bool,
                          evaluation: pd.DataFrame | None = None) -> None:
    top = summary.iloc[0]
    source = "deterministic demo data" if demo_mode else "Yahoo Finance adjusted close data"
    evaluation_section = ""
    if evaluation is not None and not evaluation.empty:
        winners = []
        for index_name, group in evaluation.groupby("index"):
            best = {metric: group.loc[group[metric].idxmin(), "method"] for metric in ("RMSE", "MAE", "QLIKE")}
            winners.append(f"- **{index_name}**: " + ", ".join(f"{metric} — {method}" for metric, method in best.items()))
        evaluation_section = f"""

## Forecast Evaluation

One-day-ahead forecasts use an expanding training window. GARCH parameters are
re-estimated every 20 holdout observations; no future conditional volatility is used.
Lower values are better.

{_markdown_table(evaluation)}

### Best Method by Metric

{chr(10).join(winners)}
"""
    markdown = f"""# GARCH(1,1) Global Index Volatility Report

Data source: {source}

## Key Finding

The most persistent volatility process is **{top['index']}**, with alpha + beta = **{top['persistence_alpha_plus_beta']:.3f}**.

## Model

```text
r_t = mu + epsilon_t
sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2
```

## Summary

{_markdown_table(summary)}

## Generated Charts

- `conditional_volatility.png`
- `persistence.png`
{evaluation_section}
"""
    output_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in display.astype(str).values.tolist())
    return "\n".join(lines)

"""Charts for EDA, GARCH conditional volatility, persistence, and forecasts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis import GarchFitResult

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prices(prices: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    # Normalize to 100 at start for visual comparison
    normed = prices / prices.iloc[0] * 100
    normed.plot(ax=ax)
    ax.set_title("Global Indices — Normalized Price (start = 100)")
    ax.set_ylabel("Index level")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, path)


def plot_returns(returns: pd.DataFrame, path: Path) -> None:
    n = returns.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, returns.columns):
        ax.plot(returns.index, returns[col], lw=0.6, color="steelblue")
        ax.set_ylabel(col, fontsize=9)
        ax.axhline(0, color="k", lw=0.4)
    axes[0].set_title("Daily Log Returns (%)")
    axes[-1].set_xlabel("Date")
    _save(fig, path)


def plot_return_distributions(returns: pd.DataFrame, path: Path) -> None:
    melted = returns.melt(var_name="index", value_name="ret").dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=melted, x="ret", hue="index", element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title("Return Distributions")
    ax.set_xlabel("Daily log return (%)")
    _save(fig, path)


def plot_rolling_volatility(returns: pd.DataFrame, window: int, path: Path) -> None:
    roll = returns.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 5))
    roll.plot(ax=ax)
    ax.set_title(f"Rolling Annualized Volatility ({window}-day)")
    ax.set_ylabel("σ (ann.)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, path)


def plot_conditional_volatility(fits: dict[str, GarchFitResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, fit in fits.items():
        ax.plot(fit.conditional_vol.index, fit.conditional_vol, lw=0.9, label=name)
    ax.set_title("GARCH Conditional Volatility")
    ax.set_ylabel("Conditional SD (% daily)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, path)


def plot_persistence_comparison(params: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = params["persistence_alpha_plus_beta"].sort_values(ascending=False)
    colors = sns.color_palette("muted", n_colors=len(s))
    bars = ax.bar(s.index.astype(str), s.values, color=colors)
    ax.axhline(1.0, color="crimson", ls="--", lw=1, label="α+β = 1 (IGARCH boundary)")
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Persistence (α + β)")
    ax.set_title("Volatility Persistence Across Markets")
    ax.legend(loc="lower right")
    for bar, val in zip(bars, s.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.4f}", ha="center", fontsize=9)
    _save(fig, path)


def plot_garch_forecast(
    fit: GarchFitResult,
    forecast: pd.Series,
    path: Path,
    history_points: int = 1000,
) -> None:
    """
    Blue = historical conditional SD; red = forecast — matching the classic
    'Prediction based on GARCH model' style chart.
    """
    hist = fit.conditional_vol.dropna().iloc[-history_points:]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(len(hist)), hist.values, color="royalblue", lw=1.0, label="Conditional SD")
    # Connect last hist point to first forecast
    x0 = len(hist) - 1
    x_f = np.arange(x0 + 1, x0 + 1 + len(forecast))
    ax.plot(
        np.concatenate([[x0], x_f]),
        np.concatenate([[hist.values[-1]], forecast.values]),
        color="crimson",
        lw=1.8,
        label="Forecast",
    )
    ax.set_title("Prediction based on GARCH model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Conditional SD")
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_std_residuals(fit: GarchFitResult, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sr = fit.std_resid.dropna()
    axes[0].plot(sr.index, sr.values, lw=0.5, color="steelblue")
    axes[0].set_title(f"{fit.name} — Standardized Residuals")
    axes[0].axhline(0, color="k", lw=0.4)
    sns.histplot(sr, kde=True, ax=axes[1], color="steelblue", stat="density")
    axes[1].set_title("Distribution of Std. Residuals")
    _save(fig, path)

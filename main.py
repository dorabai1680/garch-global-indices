"""
GARCH Model for Global Indices
==============================

End-to-end pipeline:
  1. Download major equity-index prices via yfinance
  2. Compute log / simple returns
  3. EDA: prices, returns, rolling vol, distributions
  4. Stationarity tests (ADF, KPSS)
  5. Fit GARCH(1,1) by MLE; optional order & Student-t selection
  6. Compare volatility persistence (α + β) across markets
  7. Conditional-volatility plots + multi-step forecasts
  8. Residual diagnostics and optional VaR sketch

Usage
-----
    cd garch_global_indices
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.analysis import (  # noqa: E402
    fit_all_indices,
    fit_garch,
    forecast_conditional_vol,
    ljung_box_std_resid,
    parameters_table,
    select_garch_order,
    stationarity_table,
    summary_stats,
)
from src.data_loader import load_dataset  # noqa: E402
from src.utils import ensure_dirs, print_section, save_json  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_conditional_volatility,
    plot_garch_forecast,
    plot_persistence_comparison,
    plot_prices,
    plot_return_distributions,
    plot_returns,
    plot_rolling_volatility,
    plot_std_residuals,
)

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs(DATA_DIR, FIG_DIR, TAB_DIR)

    print_section("1. Download index prices")
    cache = DATA_DIR / "global_indices_adj_close.csv"
    prices, log_ret, simple_ret = load_dataset(
        config.INDICES,
        start=config.START_DATE,
        end=config.END_DATE,
        cache_path=cache,
        return_scale=config.RETURN_SCALE,
    )
    # Align on common trading days (intersection) for fair cross-market comparison
    log_ret = log_ret.dropna(how="any")
    simple_ret = simple_ret.reindex(log_ret.index)
    prices = prices.reindex(log_ret.index)
    print(f"Prices: {prices.shape[0]} days × {prices.shape[1]} indices")
    print(f"Range:  {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"Tickers: {config.INDICES}")

    print_section("2. Summary statistics (log returns, %)")
    stats = summary_stats(log_ret)
    stats.to_csv(TAB_DIR / "summary_statistics.csv")
    print(stats[["mean", "std", "min", "max", "skew", "kurtosis", "ann_vol"]].round(4))

    print_section("3. Exploratory plots")
    plot_prices(prices, FIG_DIR / "01_normalized_prices.png")
    plot_returns(log_ret, FIG_DIR / "02_log_returns.png")
    plot_return_distributions(log_ret, FIG_DIR / "03_return_distributions.png")
    plot_rolling_volatility(log_ret, config.ROLLING_WINDOW, FIG_DIR / "04_rolling_volatility.png")
    print(f"Figures written to {FIG_DIR}")

    print_section("4. Stationarity (ADF & KPSS)")
    st = stationarity_table(log_ret)
    st.to_csv(TAB_DIR / "stationarity_tests.csv")
    print(st.round(4))

    print_section("5. Fit GARCH(1,1) — Normal innovations (MLE)")
    fits_normal = fit_all_indices(log_ret, p=1, q=1, dist="normal")
    params_n = parameters_table(fits_normal)
    params_n.to_csv(TAB_DIR / "garch11_normal_params.csv")
    print(params_n[["omega", "alpha", "beta", "persistence_alpha_plus_beta", "aic", "bic"]].round(6))

    print_section("6. Model-order selection by AIC / BIC")
    ic_frames = []
    best_by_aic: dict[str, object] = {}
    for col in log_ret.columns:
        table, best = select_garch_order(log_ret[col], config.GARCH_ORDERS, dist="normal")
        ic_frames.append(table)
        best_by_aic[col] = best
        print(
            f"  {col}: best GARCH{best.order}  "
            f"AIC={best.aic:.2f}  persistence={best.persistence:.4f}"
        )
    ic_all = pd.concat(ic_frames, ignore_index=True)
    ic_all.to_csv(TAB_DIR / "model_selection_aic_bic.csv", index=False)

    fits_t = {}
    if config.COMPARE_STUDENT_T:
        print_section("7. Student-t vs Normal (GARCH(1,1))")
        fits_t = fit_all_indices(log_ret, p=1, q=1, dist="t")
        params_t = parameters_table(fits_t)
        params_t.to_csv(TAB_DIR / "garch11_student_t_params.csv")
        cmp_rows = []
        for name in log_ret.columns:
            cmp_rows.append(
                {
                    "index": name,
                    "aic_normal": fits_normal[name].aic,
                    "aic_student_t": fits_t[name].aic,
                    "bic_normal": fits_normal[name].bic,
                    "bic_student_t": fits_t[name].bic,
                    "nu": fits_t[name].nu,
                    "t_preferred_by_aic": fits_t[name].aic < fits_normal[name].aic,
                }
            )
        cmp = pd.DataFrame(cmp_rows).set_index("index")
        cmp.to_csv(TAB_DIR / "normal_vs_student_t.csv")
        print(cmp.round(4))

    print_section("8. Residual diagnostics (Ljung–Box on std. residuals)")
    lb_rows = [ljung_box_std_resid(f, lags=10) for f in fits_normal.values()]
    lb = pd.DataFrame(lb_rows).set_index("index")
    lb.to_csv(TAB_DIR / "ljung_box_std_residuals.csv")
    print(lb.round(4))

    print_section("9. Visualizations — conditional vol, persistence, forecast")
    plot_conditional_volatility(fits_normal, FIG_DIR / "05_conditional_volatility.png")
    plot_persistence_comparison(params_n, FIG_DIR / "06_persistence_comparison.png")

    # Focus forecast on S&P 500 (or first available)
    focus = "S&P 500" if "S&P 500" in fits_normal else next(iter(fits_normal))
    focus_fit = fits_normal[focus]
    fcast = forecast_conditional_vol(focus_fit, horizon=config.FORECAST_HORIZON)
    fcast.to_csv(TAB_DIR / f"forecast_vol_{focus.replace(' ', '_').replace('&', 'and')}.csv")
    plot_garch_forecast(
        focus_fit,
        fcast,
        FIG_DIR / "07_garch_forecast.png",
        history_points=min(1000, len(focus_fit.conditional_vol)),
    )
    plot_std_residuals(focus_fit, FIG_DIR / f"08_std_residuals_{focus.replace(' ', '_')}.png")

    # One forecast panel per index for the README gallery
    for name, fit in fits_normal.items():
        fc = forecast_conditional_vol(fit, horizon=config.FORECAST_HORIZON)
        safe = name.replace(" ", "_").replace("&", "and")
        plot_garch_forecast(fit, fc, FIG_DIR / f"07_forecast_{safe}.png", history_points=1000)

    print_section("10. Research Q&A snapshot")
    # Highest unconditional / sample vol
    ann = stats["ann_vol"].sort_values(ascending=False)
    most_vol = ann.index[0]
    # Highest persistence
    pers = params_n["persistence_alpha_plus_beta"].sort_values(ascending=False)
    most_pers = pers.index[0]

    answers = {
        "highest_sample_ann_vol": {"index": most_vol, "ann_vol": float(ann.iloc[0])},
        "most_persistent_volatility": {
            "index": most_pers,
            "alpha_plus_beta": float(pers.iloc[0]),
        },
        "persistence_ranking": pers.round(6).to_dict(),
        "ann_vol_ranking": ann.round(4).to_dict(),
        "student_t_helps": (
            {
                name: bool(fits_t[name].aic < fits_normal[name].aic)
                for name in fits_t
            }
            if fits_t
            else None
        ),
        "ljung_box_std_resid_ok_5pct": {
            idx: bool(row["lb_resid_pvalue"] > 0.05 and row["lb_sq_resid_pvalue"] > 0.05)
            for idx, row in lb.iterrows()
        },
        "data_range": {
            "start": str(prices.index.min().date()),
            "end": str(prices.index.max().date()),
            "n_obs": int(len(prices)),
        },
    }
    save_json(answers, TAB_DIR / "research_answers.json")
    print(f"Highest sample ann. vol : {most_vol} ({ann.iloc[0]:.2f}%)")
    print(f"Most persistent (α+β)   : {most_pers} ({pers.iloc[0]:.4f})")
    print(f"\nAll tables → {TAB_DIR}")
    print(f"All figures → {FIG_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()

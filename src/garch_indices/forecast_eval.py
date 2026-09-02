"""Leakage-free, expanding-window volatility forecast evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .garch import fit_garch11


def qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    forecast = np.maximum(np.asarray(forecast, dtype=float), 1e-10)
    realized = np.maximum(np.asarray(realized, dtype=float), 0)
    return float(np.mean(realized / forecast + np.log(forecast)))


def evaluate_index(frame: pd.DataFrame, *, window=20, ewma_lambda=0.94, test_frac=0.2,
                   refit_every=20, max_iterations=300):
    """Evaluate one-day-ahead forecasts using only information available at each origin."""
    frame = frame.sort_values("date").reset_index(drop=True)
    returns = frame["return_pct"].astype(float).to_numpy()
    realized = returns**2
    split = int((1 - test_frac) * len(frame))
    if split < max(window + 5, 60) or len(frame) - split < 1:
        return None
    if not 0 < ewma_lambda < 1 or refit_every < 1:
        raise ValueError("ewma_lambda must be in (0, 1) and refit_every must be positive")

    ewma = float(pd.Series(realized[:split]).ewm(alpha=1 - ewma_lambda, adjust=False).mean().iloc[-1])
    forecasts = {"Historical": [], "EWMA": [], "GARCH(1,1)": []}
    actual, model, garch_variance = [], None, None
    for i in range(split, len(frame)):
        if model is None or (i - split) % refit_every == 0:
            model = fit_garch11(returns[:i], max_iterations=max_iterations)
            last_residual = returns[i - 1] - model.mu
            garch_variance = model.omega + model.alpha * last_residual**2 + model.beta * model.conditional_volatility[-1]**2
        forecasts["Historical"].append(float(np.mean(realized[i - window:i])))
        forecasts["EWMA"].append(ewma)
        forecasts["GARCH(1,1)"].append(float(garch_variance))
        actual.append(float(realized[i]))
        ewma = ewma_lambda * ewma + (1 - ewma_lambda) * realized[i]
        residual = returns[i] - model.mu
        garch_variance = model.omega + model.alpha * residual**2 + model.beta * garch_variance

    actual_array = np.asarray(actual)
    results = {}
    for method, values in forecasts.items():
        predicted = np.asarray(values)
        results[method] = {
            "RMSE": float(np.sqrt(np.mean((actual_array - predicted)**2))),
            "MAE": float(np.mean(np.abs(actual_array - predicted))),
            "QLIKE": qlike(actual_array, predicted),
        }
    return results


def evaluate_frame(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    rows = []
    for index_name, subset in frame.groupby("index"):
        result = evaluate_index(subset, **kwargs)
        if result:
            for method, metrics in result.items():
                rows.append({"index": index_name, "method": method, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/returns_and_volatility.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/forecast_evaluation.csv"))
    parser.add_argument("--refit-every", type=int, default=20)
    args = parser.parse_args()
    result = evaluate_frame(pd.read_csv(args.input, parse_dates=["date"]), refit_every=args.refit_every)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote forecast evaluation to {args.output.resolve()}")


if __name__ == "__main__":
    main()

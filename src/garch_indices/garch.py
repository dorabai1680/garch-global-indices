from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GarchResult:
    omega: float
    alpha: float
    beta: float
    mu: float
    log_likelihood: float
    aic: float
    bic: float
    persistence: float
    half_life_days: float | None
    conditional_volatility: np.ndarray


def fit_garch11(returns_pct: np.ndarray, *, max_iterations: int = 450) -> GarchResult:
    returns = np.asarray(returns_pct, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 60:
        raise ValueError("At least 60 daily returns are required for a stable GARCH fit.")
    mu = float(np.mean(returns))
    residuals = returns - mu
    variance = float(np.var(residuals, ddof=1))
    start = _to_theta(max(variance * 0.05, 1e-6), 0.06, 0.90)
    objective = lambda theta: _negative_log_likelihood(theta, residuals)
    theta = _nelder_mead(objective, start, max_iterations=max_iterations)
    omega, alpha, beta = _from_theta(theta)
    sigma2 = _conditional_variance(residuals, omega, alpha, beta)
    log_likelihood = -objective(theta)
    persistence = alpha + beta
    half_life = float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else None
    return GarchResult(
        omega, alpha, beta, mu, float(log_likelihood),
        float(8 - 2 * log_likelihood),
        float(np.log(returns.size) * 4 - 2 * log_likelihood),
        float(persistence), half_life, np.sqrt(sigma2),
    )


def _negative_log_likelihood(theta: np.ndarray, residuals: np.ndarray) -> float:
    omega, alpha, beta = _from_theta(theta)
    sigma2 = _conditional_variance(residuals, omega, alpha, beta)
    if not np.all(np.isfinite(sigma2)):
        return float("inf")
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + residuals**2 / sigma2))


def _conditional_variance(residuals, omega, alpha, beta):
    sigma2 = np.empty_like(residuals)
    sigma2[0] = max(float(np.var(residuals, ddof=1)), omega / max(1 - alpha - beta, 1e-6), 1e-8)
    for i in range(1, residuals.size):
        sigma2[i] = max(omega + alpha * residuals[i - 1] ** 2 + beta * sigma2[i - 1], 1e-10)
    return sigma2


def _from_theta(theta):
    omega = float(np.exp(theta[0]))
    alpha_raw, beta_raw = float(np.exp(theta[1])), float(np.exp(theta[2]))
    denominator = 1 + alpha_raw + beta_raw
    return omega, 0.999 * alpha_raw / denominator, 0.999 * beta_raw / denominator


def _to_theta(omega, alpha, beta):
    slack = max(0.999 - alpha - beta, 1e-4)
    return np.array([np.log(omega), np.log(alpha / slack), np.log(beta / slack)])


def _nelder_mead(objective, start, *, max_iterations, tolerance=1e-7):
    simplex = [start]
    for i in range(start.size):
        vertex = start.copy()
        vertex[i] += 0.18 if start[i] == 0 else 0.08 * abs(start[i])
        simplex.append(vertex)
    simplex = np.array(simplex)
    values = np.array([objective(vertex) for vertex in simplex])
    for _ in range(max_iterations):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if np.std(values) < tolerance:
            break
        centroid, worst = simplex[:-1].mean(axis=0), simplex[-1]
        reflected = centroid + (centroid - worst)
        reflected_value = objective(reflected)
        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
        elif reflected_value < values[0]:
            expanded = centroid + 2 * (reflected - centroid)
            expanded_value = objective(expanded)
            simplex[-1], values[-1] = (expanded, expanded_value) if expanded_value < reflected_value else (reflected, reflected_value)
        else:
            contracted = centroid + 0.5 * (worst - centroid)
            contracted_value = objective(contracted)
            if contracted_value < values[-1]:
                simplex[-1], values[-1] = contracted, contracted_value
            else:
                simplex = simplex[0] + 0.5 * (simplex - simplex[0])
                values = np.array([objective(vertex) for vertex in simplex])
    return simplex[int(np.argmin(values))]


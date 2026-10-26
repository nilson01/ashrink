"""Shared numerical input and mixture helpers."""

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


def vector(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty real numeric vector")
    array = np.array(array, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def fit_weights(loglik, *, penalty=None, initial=None, maxiter=10000, tol=1e-8):
    """Fit finite mixture weights with a Dirichlet log penalty.

    penalty contains nonnegative pseudo-counts (Dirichlet parameters minus one).
    Diagnostics distinguish the stopping rule from a global convergence claim.
    """
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    loglik = np.asarray(loglik, dtype=float)
    if loglik.ndim != 2 or np.isnan(loglik).any() or np.isposinf(loglik).any():
        raise ValueError("invalid component log likelihoods")
    n, k = loglik.shape
    if not np.all(np.isfinite(logsumexp(loglik, axis=1))):
        raise ValueError("no component supports one or more observations")
    pseudo = np.zeros(k) if penalty is None else np.asarray(penalty, dtype=float)
    if pseudo.shape != (k,) or not np.isfinite(pseudo).all() or (pseudo < 0).any():
        raise ValueError("penalty must be a nonnegative finite vector of component size")
    weights = np.ones(k) / k if initial is None else vector(initial, "initial")
    if weights.shape != (k,) or (weights <= 0).any():
        raise ValueError("initial weights must be positive with one entry per component")
    weights = weights / weights.sum()
    objective = -np.inf
    converged = False
    history = []
    for iteration in range(1, min(maxiter, 200) + 1):
        with np.errstate(divide="ignore"):
            logw = np.log(weights)
        mixture = logsumexp(loglik + logw, axis=1)
        responsibilities = np.exp(loglik + logw - mixture[:, None])
        positive = pseudo > 0
        objective = float(mixture.sum() + np.dot(pseudo[positive], logw[positive]))
        history.append(objective)
        update = (responsibilities.sum(axis=0) + pseudo) / (n + pseudo.sum())
        delta = float(np.max(np.abs(update - weights)))
        weights = update
        if delta < tol:
            converged = True
            break
    row_max = np.max(loglik, axis=1)
    scaled = np.exp(loglik - row_max[:, None])
    positive = pseudo > 0

    def objective_gradient(w):
        marginal = scaled @ w
        if (marginal <= 0).any() or (w[positive] <= 0).any():
            return np.inf, np.zeros(k)
        objective = -(np.log(marginal).sum() + np.dot(pseudo[positive], np.log(w[positive]))) / n
        gradient = -(scaled.T @ (1 / marginal)) / n
        gradient[positive] -= pseudo[positive] / w[positive] / n
        return float(objective), gradient

    def optimality(w):
        marginal = scaled @ w
        gradient = (scaled / marginal[:, None]).sum(axis=0)
        active = (w > 1e-10) | positive
        gradient[positive] += pseudo[positive] / w[positive]
        target = n + pseudo.sum()
        return max(
            float(np.max(np.abs(gradient[active] - target))),
            float(np.max(np.maximum(gradient[~active] - target, 0), initial=0)),
        ) / max(1, target)

    residual = optimality(weights)
    optimizer = "EM"
    polish_success = None
    remaining = maxiter - iteration
    if remaining > 0 and residual > max(1e-7, 10 * tol):
        old_objective = objective_gradient(weights)[0]

        def callback(w):
            value = objective_gradient(w)[0]
            if np.isfinite(value):
                history.append(float(-n * value + row_max.sum()))

        polished = minimize(
            objective_gradient,
            weights,
            jac=True,
            method="SLSQP",
            bounds=[(1e-14 if v > 0 else 0.0, 1.0) for v in pseudo],
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)},
            callback=callback,
            options={"maxiter": remaining, "ftol": min(1e-15, tol * 0.001)},
        )
        candidate = np.maximum(polished.x, 0.0)
        candidate /= candidate.sum()
        candidate_objective = objective_gradient(candidate)[0]
        iteration += int(polished.nit)
        polish_success = bool(polished.success)
        if np.isfinite(candidate_objective) and candidate_objective <= old_objective + 1e-12:
            weights = candidate
            optimizer = "EM+SLSQP"
        residual = optimality(weights)
    converged = residual <= max(1e-7, 10 * tol)
    with np.errstate(divide="ignore"):
        logw = np.log(weights)
    mixture = logsumexp(loglik + logw, axis=1)
    resp = np.exp(loglik + logw - mixture[:, None])
    history.append(float(mixture.sum() + np.dot(pseudo[positive], logw[positive])))
    return (
        weights,
        resp,
        {
            "converged": bool(converged),
            "iterations": iteration,
            "weight_change": delta,
            "kkt_residual": residual,
            "log_likelihood": float(mixture.sum()),
            "objective_history": history,
            "optimizer": optimizer,
            "polish_success": polish_success,
            "tolerance": float(tol),
        },
    )


def tail_average(probability):
    """Mean posterior error among entries up to each local-probability cutoff.

    Equal probabilities receive the same value, including every tied entry.
    """
    values = np.asarray(probability, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    means = np.cumsum(sorted_values) / np.arange(1, len(values) + 1)
    ends = np.r_[np.flatnonzero(np.diff(sorted_values)) + 1, len(values)]
    starts = np.r_[0, ends[:-1]]
    result = np.empty_like(values)
    for start, end in zip(starts, ends):
        result[order[start:end]] = means[end - 1]
    return result

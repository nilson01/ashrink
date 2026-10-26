"""Gaussian adaptive-shrinkage regression with independent random intercepts.

The random-intercept likelihood is the usual linear Gaussian mixed model.
Independent variational coefficient and group factors extend the normal-mixture
updates of Kim et al. (2024). These are approximate marginal distributions.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import logsumexp, ndtri

from ._utils import readonly
from .regression import (
    RegressionResult,
    _boolean,
    _elbo,
    _freeze,
    _labels,
    _numeric,
    _plain,
    _positive,
    _posterior_variance,
)

__all__ = ["fit_mixed_regression", "MixedRegressionResult"]


def _groups(groups, n):
    """Preserve first-occurrence order; group identities are nonempty strings."""
    if isinstance(groups, (str, bytes)):
        raise ValueError("groups must contain one nonempty string per observation")
    try:
        values = tuple(groups)
    except TypeError as exc:
        raise ValueError("groups must contain one nonempty string per observation") from exc
    if len(values) != n or any(not isinstance(g, str) or not g.strip() for g in values):
        raise ValueError("groups must contain one nonempty string per observation")
    labels = tuple(dict.fromkeys(values))
    lookup = {g: j for j, g in enumerate(labels)}
    return labels, np.array([lookup[g] for g in values], dtype=int)


@dataclass(frozen=True)
class MixedRegressionResult:
    """Fitted fixed-effect factors and Gaussian group-intercept factors.

    Fixed effects use original predictor units. All intervals are approximate
    marginal credible intervals conditional on fitted hyperparameters. They
    exclude intercept uncertainty and covariance between coefficient factors.
    Group identifiers are case-sensitive nonempty strings.
    """

    coefficient_result: RegressionResult
    random_effects: np.ndarray
    random_effect_sd: np.ndarray
    group_labels: tuple
    group_indices: np.ndarray
    group_variance: float
    diagnostics: object

    def __post_init__(self):
        if not isinstance(self.coefficient_result, RegressionResult):
            raise ValueError("coefficient_result must be a Gaussian RegressionResult")
        groups = tuple(self.group_labels)
        if (
            not groups
            or any(not isinstance(g, str) or not g.strip() for g in groups)
            or len(set(groups)) != len(groups)
        ):
            raise ValueError("group_labels must contain distinct nonempty strings")
        mean = _numeric(self.random_effects, "random_effects", 1)
        sd = _numeric(self.random_effect_sd, "random_effect_sd", 1)
        if mean.shape != (len(groups),) or sd.shape != mean.shape or (sd <= 0).any():
            raise ValueError("random effects and positive SDs must have one entry per group")
        raw = np.asarray(self.group_indices)
        if raw.dtype.kind not in "iu" or raw.shape != (self.coefficient_result.n_observations,):
            raise ValueError("group_indices must contain one integer per training row")
        if (raw < 0).any() or (raw >= len(groups)).any() or len(np.unique(raw)) != len(groups):
            raise ValueError("group_indices must reference every fitted group")
        tau2 = _positive(self.group_variance, "group_variance")
        if _plain(dict(self.diagnostics)) != _plain(self.coefficient_result.diagnostics):
            raise ValueError("outer and coefficient diagnostics must agree")
        object.__setattr__(self, "group_labels", groups)
        object.__setattr__(self, "random_effects", readonly(mean))
        object.__setattr__(self, "random_effect_sd", readonly(sd))
        object.__setattr__(self, "group_indices", readonly(raw))
        object.__setattr__(self, "group_variance", tau2)
        object.__setattr__(self, "diagnostics", _freeze(dict(self.diagnostics)))

    def __getattr__(self, name):
        # Reuse coefficient distributions without duplicating numerical methods.
        result = object.__getattribute__(self, "coefficient_result")
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(result, name)

    @property
    def model(self):
        return "gaussian_mixed_regression"

    @property
    def family(self):
        return self.model

    @property
    def n_groups(self):
        return len(self.group_labels)

    @property
    def population_fitted_values(self):
        return readonly(self.fitted_values - self.random_effects[self.group_indices])

    def predict(self, X, groups=None, *, allow_new_groups=False):
        """Predict population means, or conditional means for supplied groups.

        With groups=None, no group effect is added. Unknown group labels raise
        unless allow_new_groups=True, which assigns their prior mean zero.
        Neither choice adds observation noise or supplies prediction intervals.
        """
        allow_new_groups = _boolean(allow_new_groups, "allow_new_groups")
        prediction = self.coefficient_result.predict(X)
        if groups is None:
            return prediction
        labels, indices = _groups(groups, len(prediction))
        lookup = dict(zip(self.group_labels, self.random_effects))
        unknown = [label for label in labels if label not in lookup]
        if unknown and not allow_new_groups:
            raise ValueError("groups contain unseen labels; use allow_new_groups=True explicitly")
        offsets = np.array([lookup.get(label, 0.0) for label in labels])
        prediction = prediction + offsets[indices]
        if not np.isfinite(prediction).all():
            raise ValueError("conditional prediction overflowed")
        return prediction

    def random_effect_interval(self, level=0.95):
        """Approximate marginal group-intercept intervals, shape (n_groups,2)."""
        if (
            isinstance(level, (bool, np.bool_))
            or not np.isscalar(level)
            or not np.isfinite(level)
            or not 0 < level < 1
        ):
            raise ValueError("level must be between zero and one")
        return self.random_effects[:, None] + self.random_effect_sd[:, None] * ndtri(
            [(1 - level) / 2, (1 + level) / 2]
        )

    def random_effect_summary(self, level=0.95):
        bounds = self.random_effect_interval(level)
        return {
            "group": self.group_labels,
            "posterior_mean": self.random_effects.copy(),
            "posterior_sd": self.random_effect_sd.copy(),
            "lower": bounds[:, 0],
            "upper": bounds[:, 1],
        }

    def random_effect_intervals(self, level=0.95):
        """Alias for random_effect_interval, with one interval per fitted group."""
        return self.random_effect_interval(level)

    def summary(self, level=0.95):
        """Return fixed-effect reporting columns; group summaries are separate."""
        return self.coefficient_result.summary(level)

    def format_summary(self, level=0.95, digits=4):
        fixed = self.coefficient_result.format_summary(level, digits)
        return (
            fixed.replace(
                "Gaussian adaptive-shrinkage regression",
                "Gaussian adaptive-shrinkage random-intercept regression",
                1,
            )
            + f"\nGroups: {self.n_groups}; random-intercept variance: {self.group_variance:.{digits}g}"
        )

    def __str__(self):
        return self.format_summary()

    def to_dict(self):
        """Serialize fitted factors, including training outcomes via residuals."""
        return {
            "schema_version": 1,
            "model": self.model,
            "coefficient_result": self.coefficient_result.to_dict(),
            "random_effects": self.random_effects.tolist(),
            "random_effect_sd": self.random_effect_sd.tolist(),
            "group_labels": list(self.group_labels),
            "group_indices": self.group_indices.tolist(),
            "group_variance": self.group_variance,
            "diagnostics": _plain(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, document):
        """Restore stored factors without optimization or executable objects."""
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != 1
            or document.get("model") != "gaussian_mixed_regression"
        ):
            raise ValueError("unsupported mixed regression result schema")
        try:
            return cls(
                RegressionResult.from_dict(document["coefficient_result"]),
                document["random_effects"],
                document["random_effect_sd"],
                document["group_labels"],
                document["group_indices"],
                document["group_variance"],
                document["diagnostics"],
            )
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("invalid mixed regression result document") from exc

    def save(self, path, *, overwrite=False):
        from .io import save_result

        return save_result(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path):
        from .io import load_result

        result = load_result(path)
        if not isinstance(result, cls):
            raise ValueError("the file does not contain a mixed regression result")
        return result


def fit_mixed_regression(
    X,
    y,
    groups,
    *,
    scales=None,
    point_mass=True,
    fit_intercept=True,
    standardize=True,
    estimate_weights=True,
    estimate_residual_variance=True,
    estimate_group_variance=True,
    initial_weights=None,
    initial_residual_variance=None,
    initial_group_variance=None,
    initial_coefficients=None,
    maxiter=1000,
    tol=1e-6,
    labels=None,
):
    """Fit Gaussian regression with shrunk coefficients and random intercepts.

    y_i = alpha + x_i b + u_groups[i] + error_i, with independent
    u_g ~ N(0,tau^2) and errors ~ N(0,sigma^2). On transformed predictors,
    b_j follows sum_k pi_k N(0,sigma^2*scales[k]^2), including a zero atom
    when requested. Group labels must be nonempty strings; convert numeric
    identifiers to strings before fitting. No random slopes or GLMM is fitted.

    Coordinate ascent uses independent coefficient and group variational
    factors. Each estimate_* flag can fix its supplied starting parameter.
    The intercept is unpenalized; centering and RMS scaling match fit_regression.
    The default grid is 2**(0.05*k)-1 for k=0,...,19. Initial coefficients
    and returned coefficient summaries use original predictor units.
    """
    design, response = _numeric(X, "X", 2), _numeric(y, "y", 1)
    n, p = design.shape
    if n < 2 or p < 1 or response.shape != (n,):
        raise ValueError("X must have shape (n,p), n >= 2 and p >= 1; y must have length n")
    group_labels, indices = _groups(groups, n)
    ng = np.bincount(indices, minlength=len(group_labels)).astype(float)
    fit_intercept = _boolean(fit_intercept, "fit_intercept")
    standardize = _boolean(standardize, "standardize")
    point_mass = _boolean(point_mass, "point_mass")
    estimate_weights = _boolean(estimate_weights, "estimate_weights")
    estimate_residual_variance = _boolean(estimate_residual_variance, "estimate_residual_variance")
    estimate_group_variance = _boolean(estimate_group_variance, "estimate_group_variance")
    labels = _labels(labels, p)
    if (
        isinstance(maxiter, (bool, np.bool_))
        or not isinstance(maxiter, (int, np.integer))
        or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer")
    tol = _positive(tol, "tol")
    x_offset = design.mean(axis=0) if fit_intercept else np.zeros(p)
    alpha = float(response.mean()) if fit_intercept else 0.0
    centered = design - x_offset
    if not np.isfinite(centered).all():
        raise ValueError("centering overflowed; rescale X")
    magnitude = np.max(np.abs(centered), axis=0)
    if (magnitude == 0).any():
        raise ValueError("X contains a zero-information predictor after centering")
    rms = magnitude * np.sqrt(np.mean((centered / magnitude) ** 2, axis=0))
    x_scale = rms if standardize else np.ones(p)
    transformed = np.array(centered / x_scale, order="F")
    d = np.einsum("ij,ij->j", transformed, transformed)
    if not np.isfinite(d).all() or (d <= 0).any() or not np.isfinite(1 / d).all():
        raise ValueError("predictor information is outside floating-point range")
    grid = 2 ** (0.05 * np.arange(20)) - 1 if scales is None else _numeric(scales, "scales", 1)
    if (
        grid.size == 0
        or (grid < 0).any()
        or (not point_mass and scales is not None and (grid == 0).any())
    ):
        raise ValueError("scales must be nonempty, nonnegative, and zero requires point_mass=True")
    grid = np.unique(grid[grid > 0])
    if point_mass:
        grid = np.r_[0.0, grid]
    if grid.size == 0:
        raise ValueError("the prior must have at least one component")
    with np.errstate(over="ignore", under="ignore"):
        grid_variance = grid**2
    if not np.isfinite(grid_variance).all() or ((grid > 0) & (grid_variance == 0)).any():
        raise ValueError("squared scales are outside floating-point range")
    k = len(grid)
    weights = (
        np.full(k, 1 / k)
        if initial_weights is None
        else _numeric(initial_weights, "initial_weights", 1)
    )
    if (
        weights.shape != (k,)
        or (weights < 0).any()
        or not np.isfinite(weights.sum())
        or weights.sum() <= 0
    ):
        raise ValueError("initial_weights must have k nonnegative entries and positive sum")
    weights /= weights.sum()
    mean = (
        np.zeros(p)
        if initial_coefficients is None
        else _numeric(initial_coefficients, "initial_coefficients", 1)
    )
    if mean.shape != (p,):
        raise ValueError("initial_coefficients must have length p")
    mean *= x_scale
    sigma2 = (
        float(np.mean((response - alpha) ** 2))
        if initial_residual_variance is None
        else _positive(initial_residual_variance, "initial_residual_variance")
    )
    sigma2 = _positive(sigma2, "initial_residual_variance")
    tau2 = (
        sigma2 / 2
        if initial_group_variance is None
        else _positive(initial_group_variance, "initial_group_variance")
    )
    tau2 = _positive(tau2, "initial_group_variance")
    u = np.zeros(len(ng))
    vu = np.full(len(ng), tau2)
    positive = grid_variance > 0
    phi = np.empty((p, k))
    means, variances = np.zeros((p, k)), np.zeros((p, k))
    history, converged = [], False
    for iteration in range(1, maxiter + 1):
        old_mean, old_u = mean.copy(), u.copy()
        old_sigma2, old_tau2, old_weights = sigma2, tau2, weights.copy()
        residual = response - alpha - transformed @ mean - u[indices]
        with np.errstate(divide="ignore"):
            log_weights = np.log(weights)
        for j in range(p):
            column = transformed[:, j]
            partial = residual + column * mean[j]
            estimate = float(column @ partial / d[j])
            total = sigma2 * (1 / d[j] + grid_variance)
            if not np.isfinite(total).all() or (total <= 0).any():
                raise ValueError("component variances are outside floating-point range")
            with np.errstate(over="ignore"):
                logp = log_weights - 0.5 * (np.log(2 * np.pi) + np.log(total) + estimate**2 / total)
            normalization = logsumexp(logp)
            if not np.isfinite(normalization):
                raise ValueError("coefficient likelihood underflowed; rescale y")
            phi[j] = np.exp(logp - normalization)
            ratio = np.zeros(k)
            ratio[positive] = grid_variance[positive] / (grid_variance[positive] + 1 / d[j])
            means[j], variances[j] = ratio * estimate, sigma2 * ratio / d[j]
            if (variances[j, positive] <= 0).any():
                raise ValueError("positive-component variance underflowed")
            mean[j] = phi[j] @ means[j]
            residual = partial - column * mean[j]
        # Conditional Gaussian group factors; no coefficient uncertainty term
        # enters their mean update under the independent-factor approximation.
        vu = 1 / (ng / sigma2 + 1 / tau2)
        u = (
            vu
            * np.bincount(indices, weights=response - alpha - transformed @ mean, minlength=len(ng))
            / sigma2
        )
        if (vu <= 0).any() or not np.isfinite(vu).all() or not np.isfinite(u).all():
            raise ValueError("group posterior factors are outside floating-point range")
        if fit_intercept:
            alpha = float(np.mean(response - transformed @ mean - u[indices]))
        residual = response - alpha - transformed @ mean - u[indices]
        if estimate_weights:
            weights = phi.mean(axis=0)
            weights /= weights.sum()
        if estimate_group_variance:
            tau2 = _positive(float(np.mean(u**2 + vu)), "updated group variance")
        if estimate_residual_variance:
            vb = _posterior_variance(phi, means, variances, mean)
            prior_quadratic = float(
                np.sum(
                    phi[:, positive]
                    * (means[:, positive] ** 2 + variances[:, positive])
                    / grid_variance[None, positive]
                )
            )
            sigma2 = _positive(
                float(
                    (residual @ residual + d @ vb + ng @ vu + prior_quadratic)
                    / (n + phi[:, positive].sum())
                ),
                "updated residual variance",
            )
        # _elbo includes E||X(b-Eb)||^2 and the augmented mixture KL.
        # Add group uncertainty in the expected likelihood, then subtract
        # KL[N(u_g,vu_g) || N(0,tau2)] for every group.
        group_kl = 0.5 * np.sum(np.log(tau2) - np.log(vu) + (vu + u**2) / tau2 - 1)
        objective = (
            _elbo(residual, d, means, variances, phi, weights, grid_variance, sigma2)
            - 0.5 * (ng @ vu) / sigma2
            - group_kl
        )
        if not np.isfinite(objective):
            raise ValueError("the mixed-model variational objective is nonfinite")
        history.append(float(objective))
        if len(history) > 1:
            change = history[-1] - history[-2]
            if change < -1e-8 * (1 + abs(history[-2])):
                raise RuntimeError("the mixed-model ELBO decreased beyond roundoff tolerance")
            if abs(change) <= tol * (1 + abs(history[-2])):
                converged = True
                break
    if not converged:
        warnings.warn(
            "Mixed regression reached maxiter before the ELBO stopping rule; inspect diagnostics",
            RuntimeWarning,
            stacklevel=2,
        )
    fitted = alpha + transformed @ mean + u[indices]
    diagnostics = {
        "converged": converged,
        "iterations": iteration,
        "optimizer": "cyclic variational coordinate ascent",
        "objective": "ELBO",
        "objective_history": history,
        "elbo": history[-1],
        "tolerance": tol,
        "coefficient_change_transformed": float(np.max(np.abs(mean - old_mean))),
        "random_effect_change": float(np.max(np.abs(u - old_u))),
        "relative_residual_variance_change": float(abs(sigma2 - old_sigma2) / old_sigma2),
        "relative_group_variance_change": float(abs(tau2 - old_tau2) / old_tau2),
        "weight_change": float(np.max(np.abs(weights - old_weights))),
        "estimate_weights": estimate_weights,
        "estimate_residual_variance": estimate_residual_variance,
        "estimate_group_variance": estimate_group_variance,
        "approximation": "independent coefficient and group factors",
        "hyperparameter_uncertainty": "not integrated",
        "intercept_uncertainty": "not integrated",
        "group_structure": "independent random intercepts",
    }
    coefficient_result = RegressionResult(
        phi,
        means / x_scale[:, None],
        variances / x_scale[:, None] ** 2,
        weights,
        grid,
        sigma2,
        x_offset,
        x_scale,
        alpha,
        fitted,
        response - fitted,
        labels,
        fit_intercept,
        standardize,
        diagnostics,
    )
    return MixedRegressionResult(
        coefficient_result, u, np.sqrt(vu), group_labels, indices, tau2, diagnostics
    )

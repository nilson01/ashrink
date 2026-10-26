"""Fixed-pattern multivariate normal empirical Bayes shrinkage.

Each row is an independent vector of noisy effects. The prior is a finite
mixture of zero-mean multivariate normals, including singular patterns.
Covariance patterns and residual correlations are supplied, not learned.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, ndtr, ndtri

from ._utils import fit_weights, readonly, tail_average

__all__ = ["fit_multivariate", "MultivariateResult"]


def _array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or (ndim is not None and raw.ndim != ndim):
        raise ValueError(f"{name} must be a real numeric array")
    result = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _covariance(value, name, *, positive=False):
    """Check symmetry and remove relative roundoff in PSD eigenvalues."""
    value = _array(value, name, 2)
    if value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    scale = max(float(np.max(np.abs(value))), np.finfo(float).tiny)
    if not np.allclose(value, value.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError(f"{name} must be symmetric")
    value = (value + value.T) / 2
    if positive:
        try:
            np.linalg.cholesky(value)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be positive definite") from exc
        return value
    eigenvalues, vectors = np.linalg.eigh(value)
    tolerance = 1e-12 * max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if eigenvalues.min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    # Exact zero marginal variances carry genuine prior atoms. Preserve them.
    zero = np.diag(value) == 0
    eigenvalues[eigenvalues <= tolerance] = 0
    value = (vectors * eigenvalues) @ vectors.T
    value[zero, :] = 0
    value[:, zero] = 0
    return (value + value.T) / 2


def _errors(effects, standard_errors, error_covariance, correlation):
    n, d = effects.shape
    if error_covariance is not None:
        if standard_errors is not None or correlation is not None:
            raise ValueError(
                "error_covariance cannot be combined with standard_errors or correlation"
            )
        errors = _array(error_covariance, "error_covariance")
        if errors.shape == (d, d):
            errors = np.broadcast_to(errors, (n, d, d)).copy()
        elif errors.shape != (n, d, d):
            raise ValueError("error_covariance must have shape (d,d) or (n,d,d)")
        return np.stack([_covariance(v, "error_covariance", positive=True) for v in errors])
    if standard_errors is None:
        raise ValueError("provide standard_errors or error_covariance")
    se = _array(standard_errors, "standard_errors", 2)
    if se.shape != effects.shape or (se <= 0).any():
        raise ValueError("standard_errors must be positive with the same shape as effects")
    corr = (
        np.eye(d) if correlation is None else _covariance(correlation, "correlation", positive=True)
    )
    if corr.shape != (d, d) or not np.allclose(np.diag(corr), 1, rtol=1e-10, atol=1e-12):
        raise ValueError("correlation must have shape (d,d) and a unit diagonal")
    return se[:, :, None] * corr[None, :, :] * se[:, None, :]


def _components(d, errors, covariances, scales, point_mass):
    if not isinstance(point_mass, (bool, np.bool_)):
        raise ValueError("point_mass must be boolean")
    canonical = covariances is None
    if canonical:
        patterns = [np.eye(d), np.ones((d, d))]
        patterns.extend(np.diag(np.eye(d)[j]) for j in range(d))
        patterns = np.stack(patterns)
    else:
        patterns = _array(covariances, "covariances", 3)
        if patterns.shape[0] == 0 or patterns.shape[1:] != (d, d):
            raise ValueError("covariances must have nonempty shape (k,d,d)")
        patterns = np.stack([_covariance(v, "covariances") for v in patterns])
    if scales is None:
        grid = (
            np.array([0.1, 0.25, 0.5, 1, 2, 4])
            * np.median(np.sqrt(np.diagonal(errors, axis1=1, axis2=2)))
            if canonical
            else np.ones(1)
        )
    else:
        grid = _array(scales, "scales", 1)
        if grid.size == 0 or (grid < 0).any():
            raise ValueError("scales must be a nonempty nonnegative vector")
    if not np.isfinite(grid**2).all():
        raise ValueError("scales are too large to square")
    components = []
    if point_mass:
        components.append(np.zeros((d, d)))
    for pattern in patterns:
        for scale in grid:
            candidate = pattern * (scale * scale)
            if not np.isfinite(candidate).all():
                raise ValueError("scaled covariance overflowed")
            # Remove duplicate components, including all supplied zero atoms.
            if not any(np.array_equal(candidate, old) for old in components):
                components.append(candidate)
    if not components:
        raise ValueError("the prior must contain at least one component")
    return np.stack(components)


def _calculations(effects, errors, components):
    n, d = effects.shape
    k = len(components)
    loglik = np.empty((n, k))
    means = np.empty((n, k, d))
    covs = np.empty((n, k, d, d))
    roots = []
    for prior in components:
        vals, vecs = np.linalg.eigh(prior)
        positive = vals > 1e-12 * max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
        roots.append(vecs[:, positive] * np.sqrt(vals[positive]))
    for i in range(n):
        error_factor = cho_factor(errors[i], lower=True)
        error_y = cho_solve(error_factor, effects[i])
        for j, (prior, root) in enumerate(zip(components, roots)):
            total = errors[i] + prior
            try:
                factor = cho_factor(total, lower=True)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "component covariance cannot be factored at floating-point precision"
                ) from exc
            loglik[i, j] = -0.5 * (
                d * np.log(2 * np.pi)
                + 2 * np.log(np.diag(factor[0])).sum()
                + effects[i] @ cho_solve(factor, effects[i])
            )
            # This form never inverts a singular prior and avoids subtraction
            # of almost equal covariance matrices for very weak information.
            if root.shape[1]:
                inner = np.eye(root.shape[1]) + root.T @ cho_solve(error_factor, root)
                post = root @ np.linalg.solve(inner, root.T)
                post = (post + post.T) / 2
                mean = post @ error_y
                zero = np.diag(prior) == 0
                post[zero, :] = 0
                post[:, zero] = 0
                mean[zero] = 0
            else:
                post = np.zeros((d, d))
                mean = np.zeros(d)
            means[i, j], covs[i, j] = mean, post
    return loglik, means, covs


def _freeze_diagnostics(diagnostics):
    def freeze(value):
        if isinstance(value, dict):
            return MappingProxyType({key: freeze(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise ValueError("diagnostics must contain JSON-compatible values")

    return freeze(dict(diagnostics))


def _plain_diagnostics(diagnostics):
    def plain(value):
        if isinstance(value, (dict, MappingProxyType)):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        return value

    return plain(diagnostics)


@dataclass(frozen=True)
class MultivariateResult:
    """Posterior summaries for fixed-pattern multivariate shrinkage.

    ``lfdr`` is the marginal probability of an exactly zero effect, which may
    include condition-specific atoms. ``joint_null_probability`` refers only
    to the component whose whole effect vector is zero. Fitted-prior summaries
    do not integrate uncertainty in the estimated mixture proportions.
    """

    effects: np.ndarray
    error_covariance: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    responsibilities: np.ndarray
    component_means: np.ndarray
    component_covariances: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    posterior_sd: np.ndarray
    probability_nonpositive: np.ndarray
    probability_nonnegative: np.ndarray
    lfdr: np.ndarray
    lfsr: np.ndarray
    qvalue: np.ndarray
    svalue: np.ndarray
    joint_null_probability: np.ndarray
    diagnostics: object
    family: str = "multivariate_normal"

    @property
    def mean(self):
        return self.posterior_mean

    @property
    def sd(self):
        return self.posterior_sd

    @property
    def covariance(self):
        return self.posterior_covariance

    @property
    def standard_errors(self):
        return readonly(np.sqrt(np.diagonal(self.error_covariance, axis1=1, axis2=2)))

    def cdf(self, value):
        """Marginal posterior P(effect <= value), including zero atoms."""
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("value must be real numeric")
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            values = np.full(self.effects.shape, values)
        if values.shape != self.effects.shape or np.isnan(values).any():
            raise ValueError("value must be scalar or have the shape of effects, without NaN")
        var = np.diagonal(self.component_covariances, axis1=2, axis2=3)
        sd = np.sqrt(np.maximum(var, 0))
        z = np.divide(
            values[:, None, :] - self.component_means, sd, out=np.zeros_like(sd), where=sd > 0
        )
        probs = np.where(sd > 0, ndtr(z), values[:, None, :] >= self.component_means)
        return np.clip(np.einsum("nk,nkd->nd", self.responsibilities, probs), 0, 1)

    @property
    def mixture_weights(self):
        return self.weights

    @property
    def null_probability(self):
        return self.lfdr

    @property
    def prior_null_probability(self):
        """Fitted-prior zero-atom probability for each condition."""
        zero = np.diagonal(self.covariances, axis1=1, axis2=2) == 0
        return readonly(self.weights @ zero.astype(float))

    @property
    def prior_mean(self):
        return readonly(np.zeros(self.effects.shape[1]))

    def _prior_marginal(self, value, condition, density):
        d = self.effects.shape[1]
        if (
            isinstance(condition, (bool, np.bool_))
            or not isinstance(condition, (int, np.integer))
            or not 0 <= condition < d
        ):
            raise ValueError("condition must identify an existing matrix column")
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("value must be real numeric")
        values = np.asarray(value, dtype=float)
        if np.isnan(values).any():
            raise ValueError("value cannot contain NaN")
        variances = self.covariances[:, condition, condition]
        continuous = variances > 0
        scales = np.sqrt(variances[continuous])
        z = values[..., None] / scales
        if density:
            result = (np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * scales)) @ self.weights[
                continuous
            ]
        else:
            result = ndtr(z) @ self.weights[continuous]
            result = result + (values >= 0) * self.weights[~continuous].sum()
        return result

    def prior_density(self, value, *, condition=0):
        """Continuous marginal fitted-prior density, excluding zero atoms.

        ``value`` can have any shape. ``condition`` selects a matrix column.
        The excluded probability is ``prior_null_probability[condition]``.
        """
        return self._prior_marginal(value, condition, True)

    def prior_cdf(self, value, *, condition=0):
        """Marginal fitted-prior CDF for one condition, including zero atoms."""
        return np.clip(self._prior_marginal(value, condition, False), 0, 1)

    def shrink(self, new_effects, standard_errors=None, *, error_covariance=None, correlation=None):
        """Evaluate new independent rows using the exact fitted prior.

        New rows do not update covariance patterns or mixture weights.
        Supply their standard errors or error covariances explicitly.
        """
        effects = _array(new_effects, "new_effects", 2)
        if effects.shape[0] < 1 or effects.shape[1] != self.effects.shape[1]:
            raise ValueError(
                "new_effects must have nonempty rows and the fitted number of conditions"
            )
        errors = _errors(effects, standard_errors, error_covariance, correlation)
        calculations = _calculations(effects, errors, self.covariances)
        diagnostics = {
            "optimizer": "none",
            "prior_fixed": True,
            "source_fit_converged": self.diagnostics.get("converged"),
            "iterations": 0,
            "objective_history": [],
            "covariance_patterns": "fixed",
            "rank_tolerance": 1e-12,
            "intervals": "marginal",
            "prior_uncertainty": "not integrated",
        }
        return _result(effects, errors, self.covariances, self.weights, diagnostics, calculations)

    def quantile(self, probability):
        """Marginal generalized-inverse CDF, not a simultaneous region.

        A scalar returns (n,d); a one-dimensional probability vector returns
        (levels,n,d). Endpoints are infinite for unbounded marginals.
        """
        raw = np.asarray(probability)
        if raw.dtype.kind not in "iuf" or raw.ndim > 1:
            raise ValueError("probability must be a numeric scalar or vector")
        levels = np.atleast_1d(np.asarray(probability, dtype=float))
        if levels.size == 0 or not np.isfinite(levels).all() or ((levels < 0) | (levels > 1)).any():
            raise ValueError("probabilities must be finite and between zero and one")
        var = np.diagonal(self.component_covariances, axis1=2, axis2=3)
        sd = np.sqrt(np.maximum(var, 0))
        outputs = np.empty((len(levels),) + self.effects.shape)
        for i in range(self.effects.shape[0]):
            active = self.responsibilities[i] > 0
            weight = self.responsibilities[i, active]
            for j in range(self.effects.shape[1]):
                means = self.component_means[i, active, j]
                sds = sd[i, active, j]
                continuous = sds > 0

                def local_cdf(x):
                    prob = np.empty_like(weight)
                    prob[continuous] = ndtr((x - means[continuous]) / sds[continuous])
                    prob[~continuous] = x >= means[~continuous]
                    return float(weight @ prob)

                atom_weight = float(weight[~continuous].sum())
                left_zero = float(weight[continuous] @ ndtr(-means[continuous] / sds[continuous]))
                for pindex, p in enumerate(levels):
                    if not continuous.any():
                        # Under this zero-mean prior every singular marginal
                        # is an atom at zero; condition-specific atoms count.
                        result = 0.0
                    elif p == 0:
                        result = -np.inf
                    elif p == 1:
                        result = np.inf
                    elif atom_weight > 0 and left_zero <= p <= left_zero + atom_weight:
                        result = 0.0
                    else:
                        component_quantiles = means + sds * ndtri(p)
                        lo, hi = float(component_quantiles.min()), float(component_quantiles.max())
                        for _ in range(100):
                            mid = lo / 2 + hi / 2
                            if local_cdf(mid) < p:
                                lo = mid
                            else:
                                hi = mid
                        result = hi
                    outputs[pindex, i, j] = result
        return outputs[0] if raw.ndim == 0 else outputs

    def interval(self, level=0.95):
        """Equal-tailed marginal credible intervals, with shape (n,d,2)."""
        if (
            isinstance(level, (bool, np.bool_))
            or not np.isscalar(level)
            or not np.isfinite(level)
            or not 0 < level < 1
        ):
            raise ValueError("level must be between zero and one")
        quantiles = self.quantile([(1 - level) / 2, (1 + level) / 2])
        return np.moveaxis(quantiles, 0, -1)

    def sample(self, size=1, seed=None):
        """Draw conditional posterior samples with shape (size,n,d)."""
        if (
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size < 1
        ):
            raise ValueError("size must be a positive integer")
        rng = np.random.default_rng(seed)
        n, d = self.effects.shape
        result = np.empty((size, n, d))
        for i in range(n):
            allocation = rng.choice(len(self.weights), size=size, p=self.responsibilities[i])
            for k in np.unique(allocation):
                selected = allocation == k
                vals, vecs = np.linalg.eigh(self.component_covariances[i, k])
                vals[vals <= 1e-12 * max(float(np.max(np.abs(vals))), np.finfo(float).tiny)] = 0
                root = vecs * np.sqrt(np.maximum(vals, 0))
                result[selected, i] = (
                    self.component_means[i, k]
                    + rng.standard_normal((int(selected.sum()), d)) @ root.T
                )
        return result

    def summary(self, level=0.95):
        """Return numeric (n,d) reporting columns without requiring pandas."""
        bounds = self.interval(level)
        return {
            "posterior_mean": self.posterior_mean.copy(),
            "posterior_sd": self.posterior_sd.copy(),
            "lower": bounds[:, :, 0],
            "upper": bounds[:, :, 1],
            "lfdr": self.lfdr.copy(),
            "null_probability": self.lfdr.copy(),
            "probability_nonpositive": self.probability_nonpositive.copy(),
            "probability_nonnegative": self.probability_nonnegative.copy(),
            "lfsr": self.lfsr.copy(),
            "qvalue": self.qvalue.copy(),
            "svalue": self.svalue.copy(),
        }

    def format_summary(self):
        """Return a short human-readable description of the result."""
        n, d = self.effects.shape
        status = "yes" if self.diagnostics.get("converged") else "not established"
        if self.diagnostics.get("prior_fixed"):
            status = "not run; prior fixed"
        return (
            f"Fixed-pattern multivariate shrinkage\nRows: {n}; conditions: {d}; components: {len(self.weights)}\n"
            f"Optimizer stopping rule met: {status}\n"
            "Intervals and sign summaries are marginal and conditional on the fitted prior.\n"
            "Use to_dataframe() for row-by-condition posterior summaries."
        )

    def to_dataframe(self, level=0.95):
        """Return a long table; install the separate ``table`` extra for pandas."""
        import pandas as pd

        n, d = self.effects.shape
        intervals = self.interval(level)
        return pd.DataFrame(
            {
                "row": np.repeat(np.arange(n), d),
                "condition": np.tile(np.arange(d), n),
                "effect": self.effects.ravel(),
                "standard_error": self.standard_errors.ravel(),
                "posterior_mean": self.posterior_mean.ravel(),
                "posterior_sd": self.posterior_sd.ravel(),
                "lower": intervals[:, :, 0].ravel(),
                "upper": intervals[:, :, 1].ravel(),
                "lfdr": self.lfdr.ravel(),
                "lfsr": self.lfsr.ravel(),
                "qvalue": self.qvalue.ravel(),
                "svalue": self.svalue.ravel(),
            }
        )

    def to_dict(self):
        """Return JSON-compatible inputs and fitted weights; no executable data."""
        return {
            "schema_version": 1,
            "model": "multivariate",
            "family": self.family,
            "effects": self.effects.tolist(),
            "error_covariance": self.error_covariance.tolist(),
            "covariances": self.covariances.tolist(),
            "weights": self.weights.tolist(),
            "diagnostics": _plain_diagnostics(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value):
        """Reconstruct summaries with fixed weights, without optimization."""
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("model") != "multivariate"
            or value.get("family") != "multivariate_normal"
        ):
            raise ValueError("unsupported multivariate result schema")
        effects = _array(value.get("effects"), "effects", 2)
        if min(effects.shape, default=0) < 1 or effects.shape[1] < 2:
            raise ValueError("effects must have nonempty shape (n,d), d >= 2")
        errors = _errors(effects, None, value.get("error_covariance"), None)
        components = _array(value.get("covariances"), "covariances", 3)
        if components.shape[0] < 1 or components.shape[1:] != (effects.shape[1],) * 2:
            raise ValueError("invalid component dimensions")
        components = np.stack([_covariance(c, "covariances") for c in components])
        weight = _array(value.get("weights"), "weights", 1)
        if (
            weight.shape != (len(components),)
            or (weight < 0).any()
            or not np.isclose(weight.sum(), 1, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError("weights must be nonnegative and sum to one")
        diagnostics = value.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            raise ValueError("diagnostics must be a mapping")
        calculations = _calculations(effects, errors, components)
        return _result(
            effects, errors, components, weight / weight.sum(), diagnostics, calculations
        )


def _result(effects, errors, components, weights, diagnostics, calculations):
    loglik, means, covs = calculations
    with np.errstate(divide="ignore"):
        log_joint = loglik + np.log(weights)
    resp = np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])
    mean = np.einsum("nk,nkd->nd", resp, means)
    centered = means - mean[:, None, :]
    covariance = np.einsum("nk,nkde->nde", resp, covs) + np.einsum(
        "nk,nkd,nke->nde", resp, centered, centered
    )
    sd = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0))
    component_sd = np.sqrt(np.maximum(np.diagonal(covs, axis1=2, axis2=3), 0))
    zero = np.diagonal(components, axis1=1, axis2=2) == 0
    score = np.divide(-means, component_sd, out=np.zeros_like(means), where=component_sd > 0)
    nonpositive = np.where(zero[None, :, :], 1, ndtr(score))
    nonnegative = np.where(zero[None, :, :], 1, ndtr(-score))
    pminus = np.clip(np.einsum("nk,nkd->nd", resp, nonpositive), 0, 1)
    pplus = np.clip(np.einsum("nk,nkd->nd", resp, nonnegative), 0, 1)
    lfdr = np.clip(resp @ zero.astype(float), 0, 1)
    lfsr = np.minimum(pminus, pplus)
    qvalue = np.column_stack([tail_average(lfdr[:, j]) for j in range(effects.shape[1])])
    svalue = np.column_stack([tail_average(lfsr[:, j]) for j in range(effects.shape[1])])
    joint = resp[:, np.all(components == 0, axis=(1, 2))].sum(axis=1)
    arrays = (
        effects,
        errors,
        components,
        weights,
        resp,
        means,
        covs,
        mean,
        covariance,
        sd,
        pminus,
        pplus,
        lfdr,
        lfsr,
        qvalue,
        svalue,
        joint,
    )
    return MultivariateResult(*(readonly(v) for v in arrays), _freeze_diagnostics(diagnostics))


def fit_multivariate(
    effects,
    standard_errors=None,
    *,
    error_covariance=None,
    correlation=None,
    covariances=None,
    scales=None,
    point_mass=True,
    null_weight=1,
    maxiter=10000,
    tol=1e-8,
):
    """Fit a finite mixture of fixed multivariate normal covariance patterns.

    Parameters
    ----------
    effects : array, shape (n,d)
        Finite effect estimates; d must be at least two. Rows are independent.
    standard_errors : array, shape (n,d)
        Positive standard errors. Used with a supplied residual correlation
        matrix or the identity. Alternatively supply ``error_covariance``.
    error_covariance : array, shape (d,d) or (n,d,d)
        Positive definite observation-error covariance. Cannot be combined
        with standard errors or a correlation matrix.
    correlation : array, shape (d,d)
        Positive definite residual correlation, treated as known.
    covariances : array, shape (k,d,d)
        Positive semidefinite prior patterns; singular patterns are allowed.
        Defaults to identity, equal-effect and single-condition patterns.
    scales : nonnegative vector
        Standard-deviation multipliers: components are scale**2 times each
        pattern. With supplied patterns the default is [1]. Canonical patterns
        use [0.1,0.25,0.5,1,2,4] times the median standard error.
    point_mass : bool
        Include one all-zero component. Zero components supplied through the
        grid are retained even when False; duplicates are removed exactly.
    null_weight : float >= 1
        Dirichlet parameter for the joint null; 1 gives maximum likelihood.
        A value above one requires a joint-null component.
    maxiter, tol : int, float
        EM iteration limit and absolute mixture-weight stopping tolerance.

    Notes
    -----
    Mixture weights are estimated on all supplied rows. Do not select only
    significant rows before learning the prior. Posterior summaries condition
    on the fitted weights, patterns and error covariance. No covariance-pattern
    learning or missing-condition imputation is performed. PSD eigenvalues at
    most 1e-12 times their largest absolute eigenvalue are treated as roundoff.
    """
    effects = _array(effects, "effects", 2)
    if effects.shape[0] < 1 or effects.shape[1] < 2:
        raise ValueError("effects must have nonempty shape (n,d), d >= 2")
    errors = _errors(effects, standard_errors, error_covariance, correlation)
    components = _components(effects.shape[1], errors, covariances, scales, point_mass)
    if (
        isinstance(null_weight, (bool, np.bool_))
        or not np.isscalar(null_weight)
        or not np.isfinite(null_weight)
        or null_weight < 1
    ):
        raise ValueError("null_weight must be finite and at least one")
    penalty = np.zeros(len(components))
    null = np.all(components == 0, axis=(1, 2))
    if null_weight > 1 and not null.any():
        raise ValueError("null_weight > 1 requires a joint-null component")
    penalty[null] = null_weight - 1
    calculations = _calculations(effects, errors, components)
    weights, _, diagnostics = fit_weights(
        calculations[0], penalty=penalty, maxiter=maxiter, tol=tol
    )
    diagnostics = dict(
        diagnostics,
        covariance_patterns="fixed",
        rank_tolerance=1e-12,
        intervals="marginal",
        prior_uncertainty="not integrated",
        null_weight=float(null_weight),
    )
    return _result(effects, errors, components, weights, diagnostics, calculations)

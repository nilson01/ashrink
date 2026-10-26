"""Gaussian regression with a learned normal-mixture shrinkage prior.

The coordinate-ascent variational algorithm follows Kim et al. (2024),
Sections 3.1--3.5 and Appendix E. Coefficient distributions are approximate
marginals, conditional on fitted mixture weights and residual variance.
"""

from dataclasses import dataclass
from types import MappingProxyType
import warnings

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

from ._utils import readonly, tail_average

__all__ = ["fit_regression", "RegressionResult"]


def _numeric(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or (ndim is not None and raw.ndim != ndim):
        raise ValueError(f"{name} must be a real numeric array")
    array = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _positive(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 0:
        raise ValueError(f"{name} must be a positive finite number")
    scalar = float(raw)
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return scalar


def _labels(value, size):
    if value is None:
        return tuple(f"x{j + 1}" for j in range(size))
    if isinstance(value, (str, bytes)):
        raise ValueError("labels must contain one distinct nonempty string per predictor")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError("labels must contain one string per predictor") from exc
    if len(result) != size or any(not isinstance(v, str) or not v.strip() for v in result):
        raise ValueError("labels must contain one nonempty string per predictor")
    if len(set(result)) != size:
        raise ValueError("labels must be distinct")
    return result


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("diagnostics must contain finite JSON numbers")
        return value
    raise ValueError("diagnostics must contain JSON-compatible values")


def _plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _posterior_variance(phi, means, variances, mean):
    # Centering the component means avoids cancellation in E(b^2)-E(b)^2.
    return np.sum(phi * (variances + (means - mean[:, None]) ** 2), axis=1)


def _elbo(residual, d, means, variances, phi, weights, grid_variance, sigma2):
    """Augmented-mixture ELBO, Kim et al. (2024), Equation (60)."""
    mean = np.sum(phi * means, axis=1)
    variance = _posterior_variance(phi, means, variances, mean)
    rss = float(residual @ residual + d @ variance)
    likelihood = -0.5 * (len(residual) * (np.log(2 * np.pi) + np.log(sigma2)) + rss / sigma2)
    nonzero_phi = phi > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        categorical = np.sum(
            np.where(nonzero_phi, phi * (np.log(phi) - np.log(weights)[None, :]), 0)
        )
    positive = grid_variance > 0
    if positive.any():
        prior_variance = sigma2 * grid_variance[positive]
        # Positive-component posterior variances remain positive even when their
        # responsibilities are zero, so this KL is well defined at boundaries.
        normal = 0.5 * (
            np.log(prior_variance)[None, :]
            - np.log(variances[:, positive])
            + (variances[:, positive] + means[:, positive] ** 2) / prior_variance[None, :]
            - 1
        )
        gaussian = float(np.sum(phi[:, positive] * normal))
    else:
        gaussian = 0.0
    return float(likelihood - categorical - gaussian)


@dataclass(frozen=True)
class RegressionResult:
    """Variational coefficient distributions for Gaussian regression.

    Coefficients and their component distributions use the original predictor
    units. ``scales`` instead describes the prior on the transformed design:
    component k has variance ``residual_variance * scales[k]**2``. Intercept
    estimation and hyperparameter uncertainty are not included in the marginal
    coefficient intervals. Correlated predictors can have underestimated joint
    uncertainty under the factorized approximation.
    """

    responsibilities: np.ndarray
    component_means: np.ndarray
    component_variances: np.ndarray
    weights: np.ndarray
    scales: np.ndarray
    residual_variance: float
    x_offset: np.ndarray
    x_scale: np.ndarray
    y_offset: float
    fitted_values: np.ndarray
    residuals: np.ndarray
    labels: tuple
    fit_intercept: bool
    standardize: bool
    diagnostics: object
    family: str = "gaussian_regression"

    def __post_init__(self):
        phi = _numeric(self.responsibilities, "responsibilities", 2)
        if phi.shape[0] < 1 or phi.shape[1] < 1 or (phi < 0).any():
            raise ValueError("responsibilities must have nonempty nonnegative shape (p,k)")
        p, k = phi.shape
        if not np.allclose(phi.sum(axis=1), 1, rtol=1e-10, atol=1e-12):
            raise ValueError("responsibility rows must sum to one")
        means = _numeric(self.component_means, "component_means", 2)
        variances = _numeric(self.component_variances, "component_variances", 2)
        if means.shape != phi.shape or variances.shape != phi.shape or (variances < 0).any():
            raise ValueError("component means and nonnegative variances must have shape (p,k)")
        weights = _numeric(self.weights, "weights", 1)
        scales = _numeric(self.scales, "scales", 1)
        if (
            weights.shape != (k,)
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("weights must be nonnegative, sum to one, and have length k")
        if scales.shape != (k,) or (scales < 0).any() or len(np.unique(scales)) != k:
            raise ValueError("scales must be distinct nonnegative values with length k")
        if ((phi > 0) & (weights[None, :] == 0)).any():
            raise ValueError("positive responsibilities require positive mixture weights")
        atom = scales == 0
        if (means[:, atom] != 0).any() or (variances[:, atom] != 0).any():
            raise ValueError("zero prior components must remain atoms at zero")
        if (variances[:, ~atom] <= 0).any():
            raise ValueError("positive prior components require positive posterior variances")
        offset = _numeric(self.x_offset, "x_offset", 1)
        scale = _numeric(self.x_scale, "x_scale", 1)
        if offset.shape != (p,) or scale.shape != (p,) or (scale <= 0).any():
            raise ValueError("x_offset and positive x_scale must have length p")
        fitted = _numeric(self.fitted_values, "fitted_values", 1)
        residual = _numeric(self.residuals, "residuals", 1)
        if fitted.size < 2 or residual.shape != fitted.shape:
            raise ValueError("fitted_values and residuals must have the same length n >= 2")
        yo = _numeric(self.y_offset, "y_offset", 0)
        sigma2 = _positive(self.residual_variance, "residual_variance")
        intercept = _boolean(self.fit_intercept, "fit_intercept")
        standardize = _boolean(self.standardize, "standardize")
        if not intercept and (float(yo) != 0 or (offset != 0).any()):
            raise ValueError("a fit without intercept must use zero offsets")
        if not standardize and (scale != 1).any():
            raise ValueError("an unstandardized fit must use unit x_scale")
        if self.family != "gaussian_regression":
            raise ValueError("family must be gaussian_regression")
        for name, value in (
            ("responsibilities", phi),
            ("component_means", means),
            ("component_variances", variances),
            ("weights", weights),
            ("scales", scales),
            ("x_offset", offset),
            ("x_scale", scale),
            ("fitted_values", fitted),
            ("residuals", residual),
        ):
            object.__setattr__(self, name, readonly(value))
        object.__setattr__(self, "residual_variance", sigma2)
        object.__setattr__(self, "y_offset", float(yo))
        object.__setattr__(self, "fit_intercept", intercept)
        object.__setattr__(self, "standardize", standardize)
        object.__setattr__(self, "labels", _labels(self.labels, p))
        object.__setattr__(self, "diagnostics", _freeze(dict(self.diagnostics)))

    @property
    def model(self):
        return self.family

    @property
    def posterior_mean(self):
        return readonly(np.sum(self.responsibilities * self.component_means, axis=1))

    @property
    def coefficients(self):
        return self.posterior_mean

    @property
    def coef_(self):
        return self.posterior_mean

    @property
    def mean(self):
        return self.posterior_mean

    @property
    def posterior_sd(self):
        variance = _posterior_variance(
            self.responsibilities,
            self.component_means,
            self.component_variances,
            self.posterior_mean,
        )
        return readonly(np.sqrt(variance))

    @property
    def sd(self):
        return self.posterior_sd

    @property
    def intercept(self):
        return float(self.y_offset - self.x_offset @ self.posterior_mean)

    @property
    def intercept_(self):
        return self.intercept

    @property
    def mixture_weights(self):
        return self.weights

    @property
    def lfdr(self):
        return readonly(self.responsibilities[:, self.scales == 0].sum(axis=1))

    @property
    def null_probability(self):
        return self.lfdr

    @property
    def probability_nonpositive(self):
        return readonly(self.cdf(0))

    @property
    def probability_nonnegative(self):
        continuous = self.component_variances > 0
        z = np.divide(
            self.component_means,
            np.sqrt(self.component_variances),
            out=np.zeros_like(self.component_means),
            where=continuous,
        )
        return readonly(np.sum(self.responsibilities * np.where(continuous, ndtr(z), 1), axis=1))

    @property
    def lfsr(self):
        return readonly(np.minimum(self.probability_nonpositive, self.probability_nonnegative))

    @property
    def qvalue(self):
        return readonly(tail_average(self.lfdr))

    @property
    def svalue(self):
        return readonly(tail_average(self.lfsr))

    @property
    def n_features(self):
        return self.responsibilities.shape[0]

    @property
    def n_observations(self):
        return self.fitted_values.size

    def predict(self, X):
        """Predict the outcome mean for a finite (n_new,p) numeric matrix.

        The fitted centering and scaling are already incorporated in the
        original-unit coefficients and intercept. No prediction interval or
        extra observation noise is included.
        """
        design = _numeric(X, "X", 2)
        if design.shape[1] != self.n_features:
            raise ValueError("X must have one column per fitted predictor, in fitted order")
        result = design @ self.posterior_mean + self.intercept
        if not np.isfinite(result).all():
            raise ValueError("prediction overflowed; check predictor magnitudes")
        return result

    def cdf(self, value):
        """Approximate marginal P(b_j <= value), including zero atoms."""
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("value must be a real numeric scalar or vector")
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            values = np.full(self.n_features, values)
        if values.shape != (self.n_features,) or np.isnan(values).any():
            raise ValueError("value must be scalar or length p, without NaN")
        sd = np.sqrt(self.component_variances)
        continuous = sd > 0
        z = np.divide(
            values[:, None] - self.component_means,
            sd,
            out=np.zeros_like(sd),
            where=continuous,
        )
        probability = np.where(continuous, ndtr(z), values[:, None] >= 0)
        return np.clip(np.sum(self.responsibilities * probability, axis=1), 0, 1)

    def quantile(self, probability):
        """Approximate marginal quantiles; scalar gives (p,), vector gives (levels,p)."""
        raw = np.asarray(probability)
        if raw.dtype.kind not in "iuf" or raw.ndim > 1:
            raise ValueError("probability must be a numeric scalar or vector")
        levels = np.atleast_1d(np.asarray(probability, dtype=float))
        if not levels.size or not np.isfinite(levels).all() or ((levels < 0) | (levels > 1)).any():
            raise ValueError("probabilities must be finite and between zero and one")
        result = np.empty((levels.size, self.n_features))
        for j in range(self.n_features):
            active = self.responsibilities[j] > 0
            weights = self.responsibilities[j, active]
            means = self.component_means[j, active]
            sd = np.sqrt(self.component_variances[j, active])
            continuous = sd > 0
            atom = float(weights[~continuous].sum())
            left_zero = float(weights[continuous] @ ndtr(-means[continuous] / sd[continuous]))
            for i, q in enumerate(levels):
                if not continuous.any():
                    result[i, j] = 0.0
                elif q == 0:
                    result[i, j] = -np.inf
                elif q == 1:
                    result[i, j] = np.inf
                elif atom > 0 and left_zero <= q <= left_zero + atom:
                    result[i, j] = 0.0
                else:
                    component_quantiles = means[continuous] + sd[continuous] * ndtri(q)
                    lo, hi = float(component_quantiles.min()), float(component_quantiles.max())
                    if atom:
                        lo, hi = min(lo, 0.0), max(hi, 0.0)
                    for _ in range(100):
                        mid = lo / 2 + hi / 2
                        local = float(
                            weights[continuous] @ ndtr((mid - means[continuous]) / sd[continuous])
                        )
                        if atom and mid >= 0:
                            local += atom
                        if local < q:
                            lo = mid
                        else:
                            hi = mid
                    result[i, j] = hi
        return result[0] if raw.ndim == 0 else result

    def interval(self, level=0.95):
        """Equal-tailed approximate marginal credible intervals, shape (p,2)."""
        if (
            isinstance(level, (bool, np.bool_))
            or not np.isscalar(level)
            or not np.isfinite(level)
            or not 0 < level < 1
        ):
            raise ValueError("level must be between zero and one")
        return self.quantile([(1 - level) / 2, (1 + level) / 2]).T

    def sample(self, size=1, seed=None):
        """Draw (size,p) coefficients independently from variational factors."""
        if (
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size < 1
        ):
            raise ValueError("size must be a positive integer")
        rng = np.random.default_rng(seed)
        output = np.empty((size, self.n_features))
        for j in range(self.n_features):
            allocation = rng.choice(len(self.weights), size=size, p=self.responsibilities[j])
            output[:, j] = self.component_means[j, allocation] + np.sqrt(
                self.component_variances[j, allocation]
            ) * rng.standard_normal(size)
        return output

    def summary(self, level=0.95):
        """Return coefficient reporting columns; all uncertainty is approximate."""
        bounds = self.interval(level)
        return {
            "posterior_mean": self.posterior_mean.copy(),
            "posterior_sd": self.posterior_sd.copy(),
            "lower": bounds[:, 0],
            "upper": bounds[:, 1],
            "lfdr": self.lfdr.copy(),
            "null_probability": self.lfdr.copy(),
            "probability_nonpositive": self.probability_nonpositive.copy(),
            "probability_nonnegative": self.probability_nonnegative.copy(),
            "lfsr": self.lfsr.copy(),
            "qvalue": self.qvalue.copy(),
            "svalue": self.svalue.copy(),
        }

    def format_summary(self, level=0.95, digits=4):
        """Return a compact coefficient table and the optimizer status."""
        if (
            isinstance(digits, (bool, np.bool_))
            or not isinstance(digits, (int, np.integer))
            or not 1 <= digits <= 12
        ):
            raise ValueError("digits must be an integer between 1 and 12")
        columns = self.summary(level)
        status = "yes" if self.diagnostics.get("converged") else "no"
        rows = [
            "Gaussian adaptive-shrinkage regression",
            f"Observations: {self.n_observations}; predictors: {self.n_features}; ELBO stopping rule met: {status}",
            f"Intercept: {self.intercept:.{digits}g}; residual variance: {self.residual_variance:.{digits}g}",
            f"Approximate {100 * level:g}% marginal credible intervals; fitted hyperparameters fixed.",
            "Predictor  Mean  SD  Lower  Upper  P(b=0)  lfsr",
        ]
        for j, label in enumerate(self.labels):
            values = [
                columns[key][j]
                for key in ("posterior_mean", "posterior_sd", "lower", "upper", "lfdr", "lfsr")
            ]
            rows.append(f"{label}  " + "  ".join(f"{v:.{digits}g}" for v in values))
        return "\n".join(rows)

    def __str__(self):
        return self.format_summary()

    def to_dataframe(self, level=0.95):
        """Return coefficient summaries; requires the separate ``table`` extra."""
        import pandas as pd

        return pd.DataFrame({"predictor": self.labels, **self.summary(level)})

    def to_dict(self):
        """Return versioned variational factors and prediction transformations.

        Training fitted values and residuals are included. Saving this result
        therefore retains information about training outcomes.
        """
        return {
            "schema_version": 1,
            "model": "gaussian_regression",
            "responsibilities": self.responsibilities.tolist(),
            "component_means": self.component_means.tolist(),
            "component_variances": self.component_variances.tolist(),
            "weights": self.weights.tolist(),
            "scales": self.scales.tolist(),
            "residual_variance": self.residual_variance,
            "x_offset": self.x_offset.tolist(),
            "x_scale": self.x_scale.tolist(),
            "y_offset": self.y_offset,
            "fitted_values": self.fitted_values.tolist(),
            "residuals": self.residuals.tolist(),
            "labels": list(self.labels),
            "fit_intercept": self.fit_intercept,
            "standardize": self.standardize,
            "diagnostics": _plain(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, document):
        """Restore saved factors without fitting or recomputing the posterior."""
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != 1
            or document.get("model") != "gaussian_regression"
        ):
            raise ValueError("unsupported Gaussian regression result schema")
        names = (
            "responsibilities",
            "component_means",
            "component_variances",
            "weights",
            "scales",
            "residual_variance",
            "x_offset",
            "x_scale",
            "y_offset",
            "fitted_values",
            "residuals",
            "labels",
            "fit_intercept",
            "standardize",
            "diagnostics",
        )
        try:
            return cls(**{name: document[name] for name in names})
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("invalid Gaussian regression result document") from exc

    def save(self, path, *, overwrite=False):
        """Save JSON using the shared persistence interface."""
        from .io import save_result

        return save_result(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path):
        """Load a Gaussian regression result using the shared JSON reader."""
        from .io import load_result

        result = load_result(path)
        if not isinstance(result, cls):
            raise ValueError("the file does not contain a Gaussian regression result")
        return result


def fit_regression(
    X,
    y,
    *,
    scales=None,
    point_mass=True,
    fit_intercept=True,
    standardize=True,
    estimate_weights=True,
    estimate_residual_variance=True,
    initial_coefficients=None,
    initial_weights=None,
    initial_residual_variance=None,
    maxiter=1000,
    tol=1e-6,
    labels=None,
):
    """Fit Gaussian multiple regression with a normal-mixture shrinkage prior.

    For transformed predictors, y = X b + error with error variance sigma^2,
    and b_j has prior sum_k pi_k N(0, sigma^2 * scales[k]^2). A scale of zero
    denotes a point mass at zero. Supplied ``scales`` are standard deviations
    relative to the residual SD, not component variances. The default uses
    scales 2**(0.05*k)-1 for k=0,...,19, following mr.ash's 20-point grid.
    Duplicate scales are removed and the grid is sorted. Initial weights refer
    to this final order, including the leading zero atom when requested.

    ``X`` is a finite dense (n,p) array and ``y`` has length n. Standardization
    divides each centered column by its root mean square (divisor n). With no
    intercept, columns and y are not centered. Coefficients, intervals and
    ``initial_coefficients`` use original predictor units. Columns with zero
    information after centering are rejected; the intercept is fitted separately.

    Each sweep updates coefficient factors, then mixture weights and residual
    variance while holding those factors fixed (Kim et al., 2024, Appendix E).
    Set either ``estimate_*`` flag to False to retain its initial value.
    The residual-variance update divides its expected quadratic terms by n plus
    the sum of posterior probabilities of nonzero coefficients. This is the
    fixed-factor update of Kim et al. (2024), Appendix E.3. ELBO convergence is a
    local stopping criterion, not
    evidence of a global optimum or calibrated variable selection.
    """
    design = _numeric(X, "X", 2)
    response = _numeric(y, "y", 1)
    n, p = design.shape
    if n < 2 or p < 1 or response.shape != (n,):
        raise ValueError("X must have shape (n,p), n >= 2 and p >= 1; y must have length n")
    fit_intercept = _boolean(fit_intercept, "fit_intercept")
    standardize = _boolean(standardize, "standardize")
    estimate_weights = _boolean(estimate_weights, "estimate_weights")
    estimate_residual_variance = _boolean(estimate_residual_variance, "estimate_residual_variance")
    point_mass = _boolean(point_mass, "point_mass")
    labels = _labels(labels, p)
    if (
        isinstance(maxiter, (bool, np.bool_))
        or not isinstance(maxiter, (int, np.integer))
        or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer")
    tol = _positive(tol, "tol")
    x_offset = np.mean(design, axis=0) if fit_intercept else np.zeros(p)
    y_offset = float(np.mean(response)) if fit_intercept else 0.0
    centered = design - x_offset
    yc = response - y_offset
    if not np.isfinite(centered).all() or not np.isfinite(yc).all():
        raise ValueError("centering overflowed; rescale the inputs")
    # Stable column norms avoid overflow during the usual squaring operation.
    magnitude = np.max(np.abs(centered), axis=0)
    if (magnitude == 0).any():
        raise ValueError("X contains a zero-information predictor after centering")
    rms = magnitude * np.sqrt(np.mean((centered / magnitude) ** 2, axis=0))
    x_scale = rms if standardize else np.ones(p)
    transformed = np.array(centered / x_scale, order="F")
    d = np.einsum("ij,ij->j", transformed, transformed)
    if not np.isfinite(d).all() or (d <= 0).any() or not np.isfinite(1 / d).all():
        raise ValueError("predictor information is outside floating-point range; rescale X")
    if scales is None:
        grid = 2 ** (0.05 * np.arange(20)) - 1
    else:
        grid = _numeric(scales, "scales", 1)
        if not grid.size or (grid < 0).any():
            raise ValueError("scales must be a nonempty nonnegative vector")
        if not point_mass and (grid == 0).any():
            raise ValueError("zero scales require point_mass=True")
    grid = np.unique(grid)
    grid = grid[grid > 0]
    if point_mass:
        grid = np.r_[0.0, grid]
    if not grid.size:
        raise ValueError("the prior must have at least one component")
    with np.errstate(over="ignore", under="ignore"):
        grid_variance = grid**2
    if not np.isfinite(grid_variance).all() or ((grid > 0) & (grid_variance == 0)).any():
        raise ValueError("squared scales are outside floating-point range")
    k = len(grid)
    if initial_weights is None:
        weights = np.full(k, 1 / k)
    else:
        weights = _numeric(initial_weights, "initial_weights", 1)
        if (
            weights.shape != (k,)
            or (weights < 0).any()
            or weights.sum() <= 0
            or not np.isfinite(weights.sum())
        ):
            raise ValueError(
                "initial_weights must have one nonnegative entry per component and positive sum"
            )
        weights /= weights.sum()
    if initial_coefficients is None:
        mean = np.zeros(p)
    else:
        initial = _numeric(initial_coefficients, "initial_coefficients", 1)
        if initial.shape != (p,):
            raise ValueError("initial_coefficients must have length p")
        mean = initial * x_scale
    if not np.isfinite(mean).all():
        raise ValueError("transformed initial coefficients overflowed")
    residual = yc - transformed @ mean
    sigma2 = (
        float(np.mean(yc**2))
        if initial_residual_variance is None
        else _positive(initial_residual_variance, "initial_residual_variance")
    )
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError(
            "provide positive initial_residual_variance when the default outcome variance is zero or nonfinite"
        )
    positive = grid_variance > 0
    phi = np.empty((p, k))
    means = np.zeros((p, k))
    variances = np.zeros((p, k))
    history = []
    converged = False
    previous = None
    coefficient_change = 0.0
    variance_change = 0.0
    weight_change = 0.0
    for iteration in range(1, maxiter + 1):
        old_mean = mean.copy()
        old_weights = weights.copy()
        old_sigma2 = sigma2
        with np.errstate(divide="ignore"):
            log_weights = np.log(weights)
        for j in range(p):
            column = transformed[:, j]
            partial = residual + column * mean[j]
            estimate = float(column @ partial / d[j])
            total = sigma2 * (1 / d[j] + grid_variance)
            if not np.isfinite(total).all() or (total <= 0).any():
                raise ValueError("component marginal variances are outside floating-point range")
            z = estimate / np.sqrt(total)
            with np.errstate(over="ignore"):
                log_probability = log_weights - 0.5 * (np.log(2 * np.pi) + np.log(total) + z**2)
            normalizer = logsumexp(log_probability)
            if not np.isfinite(normalizer):
                raise ValueError(
                    "coefficient likelihood underflowed; rescale y or use a wider prior grid"
                )
            phi[j] = np.exp(log_probability - normalizer)
            ratio = np.zeros(k)
            ratio[positive] = grid_variance[positive] / (grid_variance[positive] + 1 / d[j])
            means[j] = ratio * estimate
            variances[j] = sigma2 * ratio / d[j]
            if (variances[j, positive] <= 0).any():
                raise ValueError("positive-component posterior variance underflowed")
            mean[j] = float(phi[j] @ means[j])
            residual = partial - column * mean[j]
        # Refresh once per sweep to prevent residual-update roundoff accumulating.
        residual = yc - transformed @ mean
        if estimate_weights:
            weights = phi.mean(axis=0)
            weights /= weights.sum()
        if estimate_residual_variance:
            variance = _posterior_variance(phi, means, variances, mean)
            prior_quadratic = float(
                np.sum(
                    phi[:, positive]
                    * (means[:, positive] ** 2 + variances[:, positive])
                    / grid_variance[None, positive]
                )
            )
            sigma2 = float(
                (residual @ residual + d @ variance + prior_quadratic)
                / (n + phi[:, positive].sum())
            )
            if not np.isfinite(sigma2) or sigma2 <= 0:
                raise ValueError(
                    "residual variance update is zero or nonfinite; use a fixed positive variance or check the data"
                )
        objective = _elbo(residual, d, means, variances, phi, weights, grid_variance, sigma2)
        if not np.isfinite(objective):
            raise ValueError("the variational objective is nonfinite; rescale the inputs")
        history.append(objective)
        coefficient_change = float(np.max(np.abs(mean - old_mean)))
        variance_change = float(abs(sigma2 - old_sigma2) / old_sigma2)
        weight_change = float(np.max(np.abs(weights - old_weights)))
        if previous is not None:
            change = objective - previous
            if change < -1e-8 * (1 + abs(previous)):
                raise RuntimeError("the variational objective decreased beyond roundoff tolerance")
            if abs(change) <= tol * (1 + abs(previous)):
                converged = True
                break
        previous = objective
    if not converged:
        warnings.warn(
            "Gaussian regression reached maxiter before the ELBO stopping rule; inspect diagnostics",
            RuntimeWarning,
            stacklevel=2,
        )
    coefficients = mean / x_scale
    fitted = y_offset + centered @ coefficients
    diagnostics = {
        "converged": converged,
        "iterations": iteration,
        "optimizer": "cyclic variational coordinate ascent",
        "objective": "ELBO",
        "objective_history": history,
        "elbo": history[-1],
        "elbo_change": None if len(history) < 2 else history[-1] - history[-2],
        "coefficient_change_transformed": coefficient_change,
        "relative_residual_variance_change": variance_change,
        "weight_change": weight_change,
        "tolerance": tol,
        "estimate_weights": estimate_weights,
        "estimate_residual_variance": estimate_residual_variance,
        "approximation": "independent coefficient factors",
        "hyperparameter_uncertainty": "not integrated",
        "intercept_uncertainty": "not integrated",
        "residual_variance_divisor": "n + sum of posterior nonzero probabilities",
    }
    return RegressionResult(
        responsibilities=phi,
        component_means=means / x_scale[:, None],
        component_variances=variances / x_scale[:, None] ** 2,
        weights=weights,
        scales=grid,
        residual_variance=sigma2,
        x_offset=x_offset,
        x_scale=x_scale,
        y_offset=y_offset,
        fitted_values=fitted,
        residuals=response - fitted,
        labels=labels,
        fit_intercept=fit_intercept,
        standardize=standardize,
        diagnostics=diagnostics,
    )

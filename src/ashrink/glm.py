"""Variational shrinkage for logistic and Poisson regression.

Logistic fitting uses the Jaakkola--Jordan quadratic lower bound. Poisson
fitting evaluates the expected log likelihood under independent mixture-normal
coefficient factors. Neither route gives an exact joint coefficient posterior.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import expit, gammaln, logsumexp

from ._utils import readonly
from .regression import (
    RegressionResult,
    _boolean,
    _freeze,
    _labels,
    _numeric,
    _plain,
    _positive,
    _posterior_variance,
)

__all__ = ["fit_logistic_regression", "fit_poisson_regression", "GLMResult"]


def _row_vector(value, name, n, *, positive=False, integer=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim > 1:
        raise ValueError(f"{name} must be a numeric scalar or length-n vector")
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(n, result)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and scalar or length n")
    if positive and (result <= 0).any():
        raise ValueError(f"{name} must be positive")
    if integer and ((result < 0).any() or (result != np.floor(result)).any()):
        raise ValueError(f"{name} must contain nonnegative integers")
    return np.array(result, copy=True)


def _grid(scales, point_mass):
    point_mass = _boolean(point_mass, "point_mass")
    if scales is None:
        values = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    else:
        values = _numeric(scales, "scales", 1)
        if values.size == 0 or (values < 0).any():
            raise ValueError("scales must be a nonempty nonnegative vector")
        if not point_mass and (values == 0).any():
            raise ValueError("zero scales require point_mass=True")
    values = np.unique(values[values > 0])
    if point_mass:
        values = np.r_[0.0, values]
    if not values.size:
        raise ValueError("the prior must have at least one component")
    with np.errstate(over="ignore", under="ignore"):
        variance = values**2
    if not np.isfinite(variance).all() or ((values > 0) & (variance == 0)).any():
        raise ValueError("squared prior scales are outside floating-point range")
    return values, variance


def _prepare(X, y, *, informative, fit_intercept, standardize, labels):
    design = _numeric(X, "X", 2)
    response = _numeric(y, "y", 1)
    n, p = design.shape
    if n < 2 or p < 1 or response.shape != (n,):
        raise ValueError("X must have shape (n,p), n >= 2 and p >= 1; y must have length n")
    if informative.shape != (n,) or not informative.any():
        raise ValueError("at least one observation must provide likelihood information")
    fit_intercept = _boolean(fit_intercept, "fit_intercept")
    standardize = _boolean(standardize, "standardize")
    labels = _labels(labels, p)
    x_offset = design[informative].mean(axis=0) if fit_intercept else np.zeros(p)
    centered = design - x_offset
    if not np.isfinite(centered).all():
        raise ValueError("centering overflowed; rescale X")
    magnitude = np.max(np.abs(centered[informative]), axis=0)
    if (magnitude == 0).any():
        raise ValueError("X contains a zero-information predictor after centering")
    rms = magnitude * np.sqrt(np.mean((centered[informative] / magnitude) ** 2, axis=0))
    x_scale = rms if standardize else np.ones(p)
    transformed = np.array(centered / x_scale, order="F")
    if not np.isfinite(transformed).all():
        raise ValueError("standardization overflowed; rescale X")
    if not np.isfinite(transformed**2).all():
        raise ValueError("squared predictors overflowed; rescale X")
    return design, response, transformed, x_offset, x_scale, labels


def _controls(maxiter, tol, estimate_weights):
    if (
        isinstance(maxiter, (bool, np.bool_))
        or not isinstance(maxiter, (int, np.integer))
        or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer")
    return int(maxiter), _positive(tol, "tol"), _boolean(estimate_weights, "estimate_weights")


def _initialize(
    transformed, grid_variance, x_scale, initial_coefficients, initial_weights, curvature
):
    n, p = transformed.shape
    k = len(grid_variance)
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
    start = (
        np.zeros(p)
        if initial_coefficients is None
        else _numeric(initial_coefficients, "initial_coefficients", 1)
    )
    if start.shape != (p,):
        raise ValueError("initial_coefficients must have length p")
    start = start * x_scale
    if not np.isfinite(start).all():
        raise ValueError("transformed initial coefficients overflowed")
    positive = grid_variance > 0
    mass = float(weights[positive].sum())
    if mass == 0 and (start != 0).any():
        raise ValueError("nonzero initial coefficients require a positive continuous-prior weight")
    phi = np.broadcast_to(weights, (p, k)).copy()
    means = np.zeros((p, k))
    if mass > 0:
        means[:, positive] = start[:, None] / mass
    variances = np.zeros((p, k))
    d = (transformed**2).T @ curvature
    if not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError("each predictor must have positive finite likelihood information")
    variances[:, positive] = 1 / (d[:, None] + 1 / grid_variance[None, positive])
    if (variances[:, positive] <= 0).any():
        raise ValueError("positive-component variance underflowed")
    return weights, phi, means, variances


def _kl(phi, means, variances, weights, prior_variance):
    with np.errstate(divide="ignore", invalid="ignore"):
        categorical = np.sum(np.where(phi > 0, phi * (np.log(phi) - np.log(weights)[None, :]), 0))
    positive = prior_variance > 0
    gaussian = 0.0
    if positive.any():
        normal_kl = 0.5 * (
            np.log(prior_variance[positive])[None, :]
            - np.log(variances[:, positive])
            + (means[:, positive] ** 2 + variances[:, positive]) / prior_variance[None, positive]
            - 1
        )
        gaussian = float(np.sum(phi[:, positive] * normal_kl))
    return float(categorical + gaussian)


def _moments(phi, means, variances):
    mean = np.sum(phi * means, axis=1)
    variance = _posterior_variance(phi, means, variances, mean)
    return mean, variance


def _stopping(history, value, tol):
    if not np.isfinite(value):
        raise ValueError("the variational objective is nonfinite; rescale inputs or prior scales")
    if history:
        change = value - history[-1]
        if change < -1e-8 * (1 + abs(history[-1])):
            raise RuntimeError("the variational objective decreased beyond roundoff tolerance")
        stopping = abs(change) <= tol * (1 + abs(history[-1]))
    else:
        stopping = False
    history.append(float(value))
    return stopping


def _exp_checked(log_value, name):
    values = np.asarray(log_value, dtype=float)
    if not np.isfinite(values).all() or (values > np.log(np.finfo(float).max)).any():
        raise ValueError(f"{name} overflowed; rescale predictors, offsets or prior scales")
    result = np.exp(values)
    if (result == 0).any():
        raise ValueError(
            f"{name} underflowed; rescale predictors or offsets, or use link predictions"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is nonfinite")
    return result


@dataclass(frozen=True)
class GLMResult:
    """Approximate coefficients and plug-in predictions for a fitted GLM.

    Logistic coefficients are log odds per predictor unit; Poisson coefficients
    are log rates per predictor unit. Coefficient intervals condition on fitted
    mixture weights and the estimated intercept. They exclude hyperparameter
    and intercept uncertainty. Predictions apply the inverse link to the mean
    coefficients; they do not integrate the coefficient distributions.
    """

    model: str
    _coefficients: RegressionResult
    response: np.ndarray
    trials: np.ndarray
    exposure: np.ndarray
    offset: np.ndarray
    linear_predictor: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    diagnostics: object

    def __post_init__(self):
        if self.model not in ("logistic_regression", "poisson_regression"):
            raise ValueError("model must be logistic_regression or poisson_regression")
        if not isinstance(self._coefficients, RegressionResult):
            raise ValueError("coefficient factors must be a RegressionResult carrier")
        n = self._coefficients.n_observations
        response = _row_vector(self.response, "response", n, integer=True)
        trials = _row_vector(self.trials, "trials", n, integer=True)
        exposure = _row_vector(self.exposure, "exposure", n, positive=True)
        offset = _row_vector(self.offset, "offset", n)
        eta = _row_vector(self.linear_predictor, "linear_predictor", n)
        fitted = _row_vector(self.fitted_values, "fitted_values", n)
        residual = _row_vector(self.residuals, "residuals", n)
        if (fitted < 0).any() or not np.allclose(
            response - fitted, residual, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("residuals must equal response minus nonnegative fitted values")
        if self.model == "logistic_regression":
            if (response > trials).any() or trials.sum() <= 0 or (exposure != 1).any():
                raise ValueError("invalid binomial response, trials or exposure")
            expected = trials * expit(eta)
        else:
            if (trials != 1).any():
                raise ValueError("Poisson results use unit trial placeholders")
            expected = _exp_checked(eta, "Poisson fitted mean")
        if not np.allclose(expected, fitted, rtol=1e-10, atol=1e-12):
            raise ValueError("fitted values disagree with the response link")
        for name, value in (
            ("response", response),
            ("trials", trials),
            ("exposure", exposure),
            ("offset", offset),
            ("linear_predictor", eta),
            ("fitted_values", fitted),
            ("residuals", residual),
        ):
            object.__setattr__(self, name, readonly(value))
        object.__setattr__(self, "diagnostics", _freeze(dict(self.diagnostics)))

    @property
    def family(self):
        return self.model

    @property
    def posterior_mean(self):
        return self._coefficients.posterior_mean

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
        return self._coefficients.posterior_sd

    @property
    def sd(self):
        return self.posterior_sd

    @property
    def intercept(self):
        return self._coefficients.intercept

    @property
    def intercept_(self):
        return self.intercept

    @property
    def weights(self):
        return self._coefficients.weights

    @property
    def mixture_weights(self):
        return self.weights

    @property
    def scales(self):
        return self._coefficients.scales

    @property
    def responsibilities(self):
        return self._coefficients.responsibilities

    @property
    def component_means(self):
        return self._coefficients.component_means

    @property
    def component_variances(self):
        return self._coefficients.component_variances

    @property
    def labels(self):
        return self._coefficients.labels

    @property
    def x_offset(self):
        return self._coefficients.x_offset

    @property
    def x_scale(self):
        return self._coefficients.x_scale

    @property
    def fit_intercept(self):
        return self._coefficients.fit_intercept

    @property
    def standardize(self):
        return self._coefficients.standardize

    @property
    def n_features(self):
        return self._coefficients.n_features

    @property
    def n_observations(self):
        return self.response.size

    @property
    def lfdr(self):
        return self._coefficients.lfdr

    @property
    def null_probability(self):
        return self.lfdr

    @property
    def probability_nonpositive(self):
        return self._coefficients.probability_nonpositive

    @property
    def probability_nonnegative(self):
        return self._coefficients.probability_nonnegative

    @property
    def lfsr(self):
        return self._coefficients.lfsr

    @property
    def qvalue(self):
        return self._coefficients.qvalue

    @property
    def svalue(self):
        return self._coefficients.svalue

    def cdf(self, value):
        return self._coefficients.cdf(value)

    def quantile(self, probability):
        return self._coefficients.quantile(probability)

    def interval(self, level=0.95):
        """Approximate marginal coefficient credible intervals, shape (p,2)."""
        return self._coefficients.interval(level)

    def sample(self, size=1, seed=None):
        """Independent variational coefficient draws, shape (size,p)."""
        return self._coefficients.sample(size=size, seed=seed)

    def predict(self, X, *, scale="response", trials=None, exposure=1, offset=0):
        """Predict using mean coefficients and new-row offsets.

        ``scale='link'`` returns log odds or log expected counts. On the
        response scale logistic output is a probability unless ``trials`` is
        supplied, when it is expected successes. Poisson output is expected
        counts at the supplied exposure. New rows default to exposure one and
        offset zero; training offsets are never reused.
        """
        if scale not in ("response", "link"):
            raise ValueError("scale must be response or link")
        design = _numeric(X, "X", 2)
        if design.shape[1] != self.n_features:
            raise ValueError("X must have the fitted predictor columns in fitted order")
        n = design.shape[0]
        new_offset = _row_vector(offset, "offset", n)
        new_exposure = _row_vector(exposure, "exposure", n, positive=True)
        if self.model == "logistic_regression":
            if (new_exposure != 1).any():
                raise ValueError("logistic prediction does not use exposure")
            totals = None if trials is None else _row_vector(trials, "trials", n, integer=True)
            eta = self._coefficients.predict(design) + new_offset
            output = expit(eta)
            if totals is not None:
                output = output * totals
        else:
            if trials is not None:
                raise ValueError("Poisson prediction does not use trials")
            eta = self._coefficients.predict(design) + new_offset + np.log(new_exposure)
            output = _exp_checked(eta, "Poisson prediction") if scale == "response" else None
        if not np.isfinite(eta).all():
            raise ValueError("linear prediction overflowed")
        return eta if scale == "link" else output

    def summary(self, level=0.95):
        """Return approximate coefficient distributions on the link scale."""
        return self._coefficients.summary(level)

    def format_summary(self, level=0.95, digits=4):
        if (
            isinstance(digits, (bool, np.bool_))
            or not isinstance(digits, (int, np.integer))
            or not 1 <= digits <= 12
        ):
            raise ValueError("digits must be an integer between 1 and 12")
        columns = self.summary(level)
        name = "Binomial logistic" if self.model == "logistic_regression" else "Poisson log-link"
        status = "yes" if self.diagnostics.get("converged") else "no"
        rows = [
            f"{name} adaptive-shrinkage regression",
            f"Observations: {self.n_observations}; predictors: {self.n_features}; ELBO stopping rule met: {status}",
            f"Intercept: {self.intercept:.{digits}g}",
            f"Approximate {100 * level:g}% marginal credible intervals on the link scale.",
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
        """Return coefficient summaries; requires the ``table`` extra."""
        import pandas as pd

        return pd.DataFrame({"predictor": self.labels, **self.summary(level)})

    def to_dict(self):
        """Serialize factors and training responses without refitting."""
        distribution = self._coefficients.to_dict()
        for key in (
            "model",
            "schema_version",
            "residual_variance",
            "fitted_values",
            "residuals",
            "diagnostics",
        ):
            distribution.pop(key)
        return {
            "schema_version": 1,
            "model": self.model,
            "coefficient_distribution": distribution,
            "response": self.response.tolist(),
            "trials": self.trials.tolist(),
            "exposure": self.exposure.tolist(),
            "offset": self.offset.tolist(),
            "linear_predictor": self.linear_predictor.tolist(),
            "fitted_values": self.fitted_values.tolist(),
            "residuals": self.residuals.tolist(),
            "diagnostics": _plain(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, document):
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != 1
            or document.get("model") not in ("logistic_regression", "poisson_regression")
        ):
            raise ValueError("unsupported GLM result schema")
        try:
            factors = dict(document["coefficient_distribution"])
            # RegressionResult supplies distribution methods only. A unit
            # carrier scale has no interpretation as Gaussian observation noise.
            factors.update(
                schema_version=1,
                model="gaussian_regression",
                residual_variance=1.0,
                fitted_values=document["fitted_values"],
                residuals=document["residuals"],
                diagnostics=document["diagnostics"],
            )
            carrier = RegressionResult.from_dict(factors)
            names = (
                "response",
                "trials",
                "exposure",
                "offset",
                "linear_predictor",
                "fitted_values",
                "residuals",
                "diagnostics",
            )
            return cls(
                model=document["model"],
                _coefficients=carrier,
                **{key: document[key] for key in names},
            )
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("invalid GLM result document") from exc

    def save(self, path, *, overwrite=False):
        from .io import save_result

        return save_result(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path):
        from .io import load_result

        result = load_result(path)
        if not isinstance(result, cls):
            raise ValueError("the file does not contain a GLM result")
        return result


def _make_result(
    model,
    response,
    transformed,
    x_offset,
    x_scale,
    alpha,
    phi,
    means,
    variances,
    weights,
    scales,
    trials,
    exposure,
    offset,
    labels,
    fit_intercept,
    standardize,
    diagnostics,
):
    mean, _ = _moments(phi, means, variances)
    combined_offset = offset + np.log(exposure)
    eta = alpha + combined_offset + transformed @ mean
    fitted = (
        trials * expit(eta)
        if model == "logistic_regression"
        else _exp_checked(eta, "Poisson fitted mean")
    )
    residual = response - fitted
    carrier = RegressionResult(
        responsibilities=phi,
        component_means=means / x_scale[:, None],
        component_variances=variances / x_scale[:, None] ** 2,
        weights=weights,
        scales=scales,
        residual_variance=1.0,
        x_offset=x_offset,
        x_scale=x_scale,
        y_offset=alpha,
        fitted_values=fitted,
        residuals=residual,
        labels=labels,
        fit_intercept=fit_intercept,
        standardize=standardize,
        diagnostics=diagnostics,
    )
    return GLMResult(
        model=model,
        _coefficients=carrier,
        response=response,
        trials=trials,
        exposure=exposure,
        offset=offset,
        linear_predictor=eta,
        fitted_values=fitted,
        residuals=residual,
        diagnostics=diagnostics,
    )


def _diagnostics(history, converged, iteration, tol, estimate_weights, objective, **extra):
    return {
        "converged": bool(converged),
        "iterations": int(iteration),
        "optimizer": "cyclic variational coordinate ascent",
        "objective": objective,
        "objective_history": history,
        "elbo": history[-1],
        "elbo_change": None if len(history) < 2 else history[-1] - history[-2],
        "tolerance": tol,
        "estimate_weights": estimate_weights,
        "approximation": "independent mixture-normal coefficient factors",
        "hyperparameter_uncertainty": "not integrated",
        "intercept_uncertainty": "not integrated",
        "prior_scale": "absolute transformed-coefficient SD",
        **extra,
    }


def _jj_curvature(xi, trials):
    # lim_{xi -> 0} tanh(xi/2)/(4 xi) = 1/8.
    value = np.full_like(xi, 0.125)
    nonzero = xi > 1e-8
    value[nonzero] = np.tanh(xi[nonzero] / 2) / (4 * xi[nonzero])
    return 2 * trials * value


def _logistic_objective(
    response, trials, eta, eta_variance, phi, means, variances, weights, prior_variance
):
    second = eta**2 + eta_variance
    if not np.isfinite(second).all():
        raise ValueError("logistic predictor second moment overflowed")
    xi = np.sqrt(np.maximum(second, 0))
    logcosh = np.logaddexp(xi / 2, -xi / 2)
    const = gammaln(trials + 1) - gammaln(response + 1) - gammaln(trials - response + 1)
    bound = float(np.sum(const + (response - trials / 2) * eta - trials * logcosh))
    return bound - _kl(phi, means, variances, weights, prior_variance), xi


def fit_logistic_regression(
    X,
    y,
    *,
    trials=1,
    offset=0,
    scales=None,
    point_mass=True,
    fit_intercept=True,
    standardize=True,
    estimate_weights=True,
    initial_coefficients=None,
    initial_weights=None,
    initial_intercept=None,
    maxiter=1000,
    tol=1e-6,
    labels=None,
):
    """Fit binomial logistic regression with adaptive normal-mixture priors.

    ``y`` contains successes and ``trials`` contains totals; binary outcomes use
    trials one. The prior scales are absolute coefficient SDs after predictor
    transformation, with no Gaussian residual-variance factor. Scales are
    sorted and deduplicated; initial weights refer to that final grid, including
    a leading zero atom when requested. Defaults are 0.25, 0.5, 1, 2 and 4.
    Initial coefficients use original predictor units; initial_intercept is the
    intercept for the transformed design. Returned intercepts use original units.

    Zero-trial rows remain in the result but do not determine predictor centering
    or scaling. Inputs are finite dense arrays; missing values are rejected.
    An unpenalized intercept requires at least one success and one failure.
    Fitting maximizes a Jaakkola--Jordan lower bound using independent
    coefficient factors. Its intervals are approximate marginal credible
    intervals, conditional on fitted weights and intercept.
    """
    response = _numeric(y, "y", 1)
    n = response.size
    totals = _row_vector(trials, "trials", n, integer=True)
    response = _row_vector(response, "y", n, integer=True)
    if (response > totals).any() or totals.sum() <= 0:
        raise ValueError("successes cannot exceed trials, and total trials must be positive")
    raw_offset = _row_vector(offset, "offset", n)
    fit_intercept = _boolean(fit_intercept, "fit_intercept")
    standardize = _boolean(standardize, "standardize")
    if fit_intercept and (response.sum() == 0 or response.sum() == totals.sum()):
        raise ValueError("an unpenalized logistic intercept requires both successes and failures")
    _, response, design, x_offset, x_scale, labels = _prepare(
        X,
        response,
        informative=totals > 0,
        fit_intercept=fit_intercept,
        standardize=standardize,
        labels=labels,
    )
    maxiter, tol, estimate_weights = _controls(maxiter, tol, estimate_weights)
    scales, prior_variance = _grid(scales, point_mass)
    if initial_intercept is None:
        alpha = (
            float(np.log(response.sum()) - np.log((totals - response).sum()))
            if fit_intercept
            else 0.0
        )
    else:
        alpha = float(_numeric(initial_intercept, "initial_intercept", 0))
        if not fit_intercept and alpha != 0:
            raise ValueError("initial_intercept must be zero when fit_intercept=False")
    weights, phi, means, variances = _initialize(
        design,
        prior_variance,
        x_scale,
        initial_coefficients,
        initial_weights,
        totals / 4,
    )
    positive = prior_variance > 0
    mean, variance = _moments(phi, means, variances)
    eta = alpha + raw_offset + design @ mean
    value, xi = _logistic_objective(
        response, totals, eta, design**2 @ variance, phi, means, variances, weights, prior_variance
    )
    history = [float(value)]
    if not np.isfinite(value):
        raise ValueError("initial logistic bound is nonfinite")
    converged = False
    kappa = response - totals / 2
    for iteration in range(1, maxiter + 1):
        curvature = _jj_curvature(xi, totals)
        with np.errstate(divide="ignore"):
            log_weights = np.log(weights)
        eta = alpha + raw_offset + design @ mean
        for j in range(design.shape[1]):
            column = design[:, j]
            information = float(curvature @ (column**2))
            linear = float(column @ (kappa - curvature * (eta - column * mean[j])))
            variances[j, positive] = 1 / (information + 1 / prior_variance[positive])
            means[j, positive] = variances[j, positive] * linear
            log_probability = log_weights.copy()
            log_probability[positive] += 0.5 * (
                np.log(variances[j, positive])
                - np.log(prior_variance[positive])
                + means[j, positive] ** 2 / variances[j, positive]
            )
            normalizer = logsumexp(log_probability)
            if not np.isfinite(normalizer):
                raise ValueError("logistic coefficient update is nonfinite; rescale inputs")
            phi[j] = np.exp(log_probability - normalizer)
            new_mean = float(phi[j] @ means[j])
            eta += column * (new_mean - mean[j])
            mean[j] = new_mean
        eta_without_intercept = raw_offset + design @ mean
        if fit_intercept:
            alpha = float((kappa.sum() - curvature @ eta_without_intercept) / curvature.sum())
        if estimate_weights:
            weights = phi.mean(axis=0)
            weights /= weights.sum()
        mean, variance = _moments(phi, means, variances)
        eta = alpha + eta_without_intercept
        value, xi = _logistic_objective(
            response,
            totals,
            eta,
            design**2 @ variance,
            phi,
            means,
            variances,
            weights,
            prior_variance,
        )
        converged = _stopping(history, value, tol)
        if converged:
            break
    if not converged:
        warnings.warn(
            "Logistic regression reached maxiter before the ELBO stopping rule; inspect diagnostics",
            RuntimeWarning,
            stacklevel=2,
        )
    diagnostics = _diagnostics(
        history,
        converged,
        iteration,
        tol,
        estimate_weights,
        "Jaakkola-Jordan logistic-bound ELBO",
        auxiliary_xi=xi.tolist(),
    )
    return _make_result(
        "logistic_regression",
        response,
        design,
        x_offset,
        x_scale,
        alpha,
        phi,
        means,
        variances,
        weights,
        scales,
        totals,
        np.ones(n),
        raw_offset,
        labels,
        fit_intercept,
        standardize,
        diagnostics,
    )


def _log_mgf(column, phi, means, variances):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        terms = (
            np.log(phi)[None, :]
            + column[:, None] * means[None, :]
            + 0.5 * column[:, None] ** 2 * variances[None, :]
        )
    result = logsumexp(terms, axis=1)
    if not np.isfinite(result).all():
        raise ValueError(
            "a Poisson coefficient exponential moment is nonfinite; rescale inputs or priors"
        )
    return result


def _poisson_component(
    column, log_a, y_dot, prior_variance, initial_mean, initial_variance, maxiter, tol
):
    """Optimize a Gaussian factor by damped Newton steps in mean and variance.

    The negative objective is convex in these two variables for v > 0.
    Backtracking enforces positivity and rejects overflowed trial points.
    Scores subtract the common zero-atom expected likelihood for stability.
    """
    x = column
    x2 = x**2
    base = _exp_checked(log_a, "Poisson partial exponential moment")
    tau2 = prior_variance

    def evaluate(m, v):
        if not np.isfinite(m) or not np.isfinite(v) or v <= 0:
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            t = x * m + 0.5 * x2 * v
            log_rate = log_a + t
            if not np.isfinite(log_rate).all() or (log_rate > np.log(np.finfo(float).max)).any():
                return None
            rate = np.exp(log_rate)
            difference = rate - base
            close = np.abs(t) < 0.5
            difference[close] = base[close] * np.expm1(t[close])
            objective = (
                -y_dot * m
                + difference.sum()
                + 0.5 * ((m * m + v) / tau2 - 1 + np.log(tau2) - np.log(v))
            )
            gradient = np.array([rate @ x + m / tau2 - y_dot, 0.5 * (rate @ x2 + 1 / tau2 - 1 / v)])
            hessian = np.array(
                [
                    [rate @ x2 + 1 / tau2, 0.5 * (rate @ (x * x2))],
                    [0.5 * (rate @ (x * x2)), 0.25 * (rate @ (x2 * x2)) + 0.5 / v**2],
                ]
            )
        if (
            not np.isfinite(objective)
            or not np.isfinite(gradient).all()
            or not np.isfinite(hessian).all()
        ):
            return None
        return float(objective), gradient, hessian

    m, v = float(initial_mean), float(initial_variance)
    current = evaluate(m, v)
    if current is None:
        m = 0.0
        v = float(1 / (base @ x2 + 1 / tau2))
        current = evaluate(m, v)
    if current is None:
        raise ValueError("cannot initialize a finite Poisson component objective")
    for iteration in range(1, maxiter + 1):
        value, gradient, hessian = current
        scaled_gradient = max(abs(gradient[0]) * np.sqrt(v), abs(gradient[1]) * v)
        if scaled_gradient <= tol * (1 + abs(value)):
            return m, v, -value, iteration, "scaled_gradient"
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Poisson component Newton system is singular") from exc
        decrement = float(gradient @ direction)
        if not np.isfinite(decrement) or decrement < 0:
            raise RuntimeError("Poisson component Newton direction is invalid")
        step = min(1.0, 0.99 * v / direction[1]) if direction[1] > 0 else 1.0
        accepted = False
        for _ in range(60):
            proposal_m, proposal_v = m - step * direction[0], v - step * direction[1]
            proposal = evaluate(proposal_m, proposal_v)
            if proposal is not None:
                armijo = proposal[0] <= value - 1e-4 * step * decrement
                proposal_gradient = max(
                    abs(proposal[1][0]) * np.sqrt(proposal_v),
                    abs(proposal[1][1]) * proposal_v,
                )
                roundoff = 8 * np.finfo(float).eps * (1 + abs(value))
                # Near machine precision the objective can round upward even
                # when a Newton step improves the score equations. Accept only
                # an improving gradient and an objective change within roundoff.
                refinement = (
                    decrement <= roundoff
                    and proposal[0] <= value + roundoff
                    and proposal_gradient < scaled_gradient
                )
                if armijo or refinement:
                    m, v, current = proposal_m, proposal_v, proposal
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            # A machine-precision plateau can satisfy the local objective rule
            # even when a scaled gradient is slightly above its tolerance.
            if decrement <= tol**2 * (1 + abs(value)):
                return m, v, -value, iteration, "newton_decrement"
            raise RuntimeError("Poisson component line search failed")
    raise RuntimeError(
        f"Poisson component reached inner_maxiter; scaled gradient={scaled_gradient:.6g}, Newton decrement={decrement:.6g}"
    )


def _poisson_objective(
    response, eta, log_expected_rate, phi, means, variances, weights, prior_variance
):
    expected_rate = _exp_checked(log_expected_rate, "Poisson expected rate")
    likelihood = float(response @ eta - expected_rate.sum() - gammaln(response + 1).sum())
    return likelihood - _kl(phi, means, variances, weights, prior_variance)


def fit_poisson_regression(
    X,
    y,
    *,
    exposure=1,
    offset=0,
    scales=None,
    point_mass=True,
    fit_intercept=True,
    standardize=True,
    estimate_weights=True,
    initial_coefficients=None,
    initial_weights=None,
    initial_intercept=None,
    maxiter=1000,
    tol=1e-6,
    inner_maxiter=100,
    inner_tol=1e-8,
    labels=None,
):
    """Fit Poisson log-link regression with adaptive normal-mixture priors.

    Expected counts are exposure * exp(intercept + offset + X b). Exposures
    must be positive. Prior scales are absolute SDs on transformed coefficients;
    no Gaussian residual variance is fitted. Scales are sorted and deduplicated,
    and initial weights use that order with a leading zero atom when requested.
    Initial coefficients use original predictor units; initial_intercept is the
    intercept for the transformed design. Returned intercepts use original units.

    The algorithm uses the exact expected Poisson log likelihood within the
    independent mixture-normal approximation. Positive-component means and
    variances are optimized by damped Newton steps. Failed inner optimization
    and nonfinite exponential moments raise an error. Coefficient intervals
    remain approximate, conditional on fitted weights and intercept.
    """
    response = _numeric(y, "y", 1)
    n = response.size
    response = _row_vector(response, "y", n, integer=True)
    exposures = _row_vector(exposure, "exposure", n, positive=True)
    raw_offset = _row_vector(offset, "offset", n)
    combined_offset = raw_offset + np.log(exposures)
    if not np.isfinite(combined_offset).all():
        raise ValueError("combined exposure and offset must be finite")
    fit_intercept = _boolean(fit_intercept, "fit_intercept")
    standardize = _boolean(standardize, "standardize")
    if fit_intercept and response.sum() == 0:
        raise ValueError("an unpenalized Poisson intercept requires a positive count total")
    _, response, design, x_offset, x_scale, labels = _prepare(
        X,
        response,
        informative=np.ones(n, dtype=bool),
        fit_intercept=fit_intercept,
        standardize=standardize,
        labels=labels,
    )
    maxiter, tol, estimate_weights = _controls(maxiter, tol, estimate_weights)
    inner_maxiter, inner_tol, _ = _controls(inner_maxiter, inner_tol, True)
    scales, prior_variance = _grid(scales, point_mass)
    if initial_intercept is None:
        alpha = float(np.log(response.sum()) - logsumexp(combined_offset)) if fit_intercept else 0.0
    else:
        alpha = float(_numeric(initial_intercept, "initial_intercept", 0))
        if not fit_intercept and alpha != 0:
            raise ValueError("initial_intercept must be zero when fit_intercept=False")
    initial_rate = _exp_checked(alpha + combined_offset, "initial Poisson mean")
    weights, phi, means, variances = _initialize(
        design,
        prior_variance,
        x_scale,
        initial_coefficients,
        initial_weights,
        initial_rate,
    )
    positive = prior_variance > 0
    n, p = design.shape
    log_mgf = np.column_stack(
        [_log_mgf(design[:, j], phi[j], means[j], variances[j]) for j in range(p)]
    )
    log_total = log_mgf.sum(axis=1)
    mean, _ = _moments(phi, means, variances)
    value = _poisson_objective(
        response,
        alpha + combined_offset + design @ mean,
        alpha + combined_offset + log_total,
        phi,
        means,
        variances,
        weights,
        prior_variance,
    )
    if not np.isfinite(value):
        raise ValueError("initial Poisson ELBO is nonfinite")
    history = [float(value)]
    converged = False
    inner_iterations = 0
    max_inner_iterations = 0
    inner_stopping_counts = {"scaled_gradient": 0, "newton_decrement": 0}
    for iteration in range(1, maxiter + 1):
        with np.errstate(divide="ignore"):
            log_weights = np.log(weights)
        for j in range(p):
            column = design[:, j]
            log_a = alpha + combined_offset + log_total - log_mgf[:, j]
            score = np.zeros(len(scales))
            y_dot = float(response @ column)
            for k in np.flatnonzero(positive & (weights > 0)):
                try:
                    m, v, score[k], steps, stopping_rule = _poisson_component(
                        column,
                        log_a,
                        y_dot,
                        prior_variance[k],
                        means[j, k],
                        variances[j, k],
                        inner_maxiter,
                        inner_tol,
                    )
                except (ValueError, RuntimeError) as exc:
                    raise RuntimeError(
                        f"Poisson inner update failed at predictor {j}, component {k}: {exc}"
                    ) from exc
                means[j, k], variances[j, k] = m, v
                inner_stopping_counts[stopping_rule] += 1
                inner_iterations += steps
                max_inner_iterations = max(max_inner_iterations, steps)
            log_probability = log_weights + score
            normalizer = logsumexp(log_probability)
            if not np.isfinite(normalizer):
                raise ValueError("Poisson component responsibilities are nonfinite")
            phi[j] = np.exp(log_probability - normalizer)
            new_mgf = _log_mgf(column, phi[j], means[j], variances[j])
            log_total += new_mgf - log_mgf[:, j]
            log_mgf[:, j] = new_mgf
        log_total = log_mgf.sum(axis=1)
        if fit_intercept:
            alpha = float(np.log(response.sum()) - logsumexp(combined_offset + log_total))
        if estimate_weights:
            weights = phi.mean(axis=0)
            weights /= weights.sum()
        mean, _ = _moments(phi, means, variances)
        value = _poisson_objective(
            response,
            alpha + combined_offset + design @ mean,
            alpha + combined_offset + log_total,
            phi,
            means,
            variances,
            weights,
            prior_variance,
        )
        converged = _stopping(history, value, tol)
        if converged:
            break
    if not converged:
        warnings.warn(
            "Poisson regression reached maxiter before the ELBO stopping rule; inspect diagnostics",
            RuntimeWarning,
            stacklevel=2,
        )
    diagnostics = _diagnostics(
        history,
        converged,
        iteration,
        tol,
        estimate_weights,
        "Poisson expected-log-likelihood ELBO",
        inner_optimizer="damped Newton in component mean and variance",
        inner_stopping_counts=inner_stopping_counts,
        inner_iterations=inner_iterations,
        max_inner_iterations=max_inner_iterations,
        inner_tolerance=inner_tol,
        inner_maxiter=inner_maxiter,
    )
    return _make_result(
        "poisson_regression",
        response,
        design,
        x_offset,
        x_scale,
        alpha,
        phi,
        means,
        variances,
        weights,
        scales,
        np.ones(n),
        exposures,
        raw_offset,
        labels,
        fit_intercept,
        standardize,
        diagnostics,
    )

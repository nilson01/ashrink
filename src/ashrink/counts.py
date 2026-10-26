"""Empirical Bayes mixtures for Poisson rates and binomial probabilities.

These conjugate models use gamma mixtures for rates and beta mixtures for
probabilities. Mixture components share a supplied mode. Posterior probabilities condition on
estimated mixture weights; they do not include uncertainty in those weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln, gammaln, logsumexp, xlogy
from scipy.stats import beta, gamma, poisson, binom

from ._utils import fit_weights, readonly, tail_average, vector

__all__ = ["CountResult", "fit_poisson", "fit_binomial"]
_DEFAULT_STRENGTHS = np.power(2.0, np.arange(-4, 9, dtype=float))


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _integer_vector(value, name):
    value = vector(value, name)
    if np.any(value < 0) or np.any(value != np.floor(value)):
        raise ValueError(f"{name} must contain nonnegative integers")
    return value


def _broadcast_input(value, n, name):
    if np.ndim(value) == 0:
        return np.full(n, _scalar(value, name))
    value = vector(value, name)
    if value.shape != (n,):
        raise ValueError(f"{name} must be scalar or have one entry per observation")
    return value


def _setup(family, observations, auxiliary, null_value, strengths, point_mass):
    observations = _integer_vector(observations, "counts" if family == "poisson" else "successes")
    n = len(observations)
    auxiliary = _broadcast_input(auxiliary, n, "exposure" if family == "poisson" else "trials")
    null_value = _scalar(null_value, "null_value")
    if not isinstance(point_mass, (bool, np.bool_)):
        raise ValueError("point_mass must be Boolean")
    if family == "poisson":
        if null_value <= 0 or np.any(auxiliary <= 0):
            raise ValueError("null_value and exposure must be positive")
    else:
        if not 0 < null_value < 1:
            raise ValueError("null_value must lie strictly between zero and one")
        if np.any(auxiliary < observations) or np.any(auxiliary != np.floor(auxiliary)):
            raise ValueError("trials must be nonnegative integers at least as large as successes")
    strengths = vector(_DEFAULT_STRENGTHS if strengths is None else strengths, "strengths")
    if np.any(strengths <= 0) or len(np.unique(strengths)) != len(strengths):
        raise ValueError("strengths must be distinct positive values")
    a = 1 + null_value * strengths
    if family == "poisson":
        b = strengths
        loglik = (
            gammaln(observations[:, None] + a)
            - gammaln(a)
            - gammaln(observations[:, None] + 1)
            + a * np.log(b)
            + xlogy(observations[:, None], auxiliary[:, None])
            - (observations[:, None] + a) * np.log(auxiliary[:, None] + b)
        )
        atom = poisson.logpmf(observations, auxiliary * null_value)
    else:
        b = 1 + (1 - null_value) * strengths
        loglik = (
            gammaln(auxiliary[:, None] + 1)
            - gammaln(observations[:, None] + 1)
            - gammaln((auxiliary - observations)[:, None] + 1)
            + betaln(observations[:, None] + a, (auxiliary - observations)[:, None] + b)
            - betaln(a, b)
        )
        atom = binom.logpmf(observations, auxiliary, null_value)
    if point_mass:
        loglik = np.column_stack((atom, loglik))
    if not np.isfinite(loglik).all():
        raise ValueError(
            "component likelihoods are not finite; input magnitudes exceed numerical support"
        )
    return observations, auxiliary, null_value, strengths, a, b, loglik


class CountResult:
    """Fitted conjugate mixture and marginal posterior summaries.

    ``probability_below_null`` and ``probability_above_null`` are strict tail
    probabilities. ``lfsr`` includes equality in both directional error events.
    ``qvalue`` averages null probabilities at each tied null-probability cutoff;
    ``svalue`` does the same for lfsr. These are posterior summaries, not claims
    of frequentist error control.
    """

    def __init__(
        self,
        family,
        observations,
        auxiliary,
        null_value,
        strengths,
        point_mass,
        weights,
        diagnostics=None,
    ):
        if family not in ("poisson", "binomial"):
            raise ValueError("family must be poisson or binomial")
        y, aux, null, grid, a, b, loglik = _setup(
            family, observations, auxiliary, null_value, strengths, point_mass
        )
        weights = vector(weights, "mixture_weights")
        if (
            weights.shape != (loglik.shape[1],)
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("mixture_weights must be nonnegative and sum to one")
        weights /= weights.sum()
        with np.errstate(divide="ignore"):
            joint = loglik + np.log(weights)
        resp = np.exp(joint - logsumexp(joint, axis=1)[:, None])
        self.family = family
        self.observations, self.auxiliary = readonly(y), readonly(aux)
        self.null_value, self.point_mass = null, bool(point_mass)
        self.strengths, self.mixture_weights = readonly(grid), readonly(weights)
        self.posterior_weights = readonly(resp)
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)
        self._prior_a, self._prior_b = readonly(a), readonly(b)
        self._a = readonly(y[:, None] + a)
        self._b = readonly(aux[:, None] + b if family == "poisson" else (aux - y)[:, None] + b)
        if family == "poisson":
            mean, variance = self._a / self._b, self._a / self._b**2
        else:
            total = self._a + self._b
            mean = self._a / total
            variance = self._a * self._b / (total**2 * (total + 1))
        offset = int(self.point_mass)
        continuous = resp[:, offset:]
        atom = resp[:, 0] if offset else np.zeros(len(y))
        posterior_mean = continuous.__mul__(mean).sum(axis=1) + atom * null
        posterior_variance = (continuous * (variance + (mean - posterior_mean[:, None]) ** 2)).sum(
            axis=1
        ) + atom * (null - posterior_mean) ** 2
        self.posterior_mean = readonly(posterior_mean)
        self.posterior_sd = readonly(np.sqrt(np.maximum(posterior_variance, 0)))
        self.null_probability = readonly(atom)
        below = (continuous * self._component_cdf(null)).sum(axis=1)
        survival = (
            gamma.sf(null, self._a, scale=1 / self._b)
            if family == "poisson"
            else beta.sf(null, self._a, self._b)
        )
        above = (continuous * survival).sum(axis=1)
        self.probability_below_null, self.probability_above_null = readonly(below), readonly(above)
        self.lfsr = readonly(np.minimum(below + atom, above + atom))
        self.qvalue, self.svalue = readonly(tail_average(atom)), readonly(tail_average(self.lfsr))

    def shrink(self, observations, auxiliary=1):
        """Evaluate new observations using the fitted prior without refitting.

        ``auxiliary`` gives exposure for a Poisson result or trials for a
        binomial result. It may be scalar or one value per observation. Supply
        the appropriate new exposure times or trial counts explicitly when
        they differ from one. The original grid, null and mixture weights are
        reused; the original observations are not added to the new likelihood.
        """
        diagnostics = {
            "evaluation": "fixed_prior",
            "weights_refitted": False,
            "source_fit": dict(self.diagnostics),
        }
        return type(self)(
            self.family,
            observations,
            auxiliary,
            self.null_value,
            self.strengths,
            self.point_mass,
            self.mixture_weights,
            diagnostics,
        )

    @property
    def lfdr(self):
        return self.null_probability

    @property
    def counts(self):
        return self.observations

    @property
    def successes(self):
        return self.observations

    @property
    def exposure(self):
        return self.auxiliary

    @property
    def trials(self):
        return self.auxiliary

    def _component_cdf(self, value):
        if self.family == "poisson":
            return gamma.cdf(value, self._a, scale=1 / self._b)
        return beta.cdf(value, self._a, self._b)

    def cdf(self, value):
        """Return posterior P(theta <= value); scalar or one value per row."""
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            values = np.full(len(self.observations), float(values))
        if values.shape != self.observations.shape or np.isnan(values).any():
            raise ValueError("value must be scalar or one non-NaN value per observation")
        offset = int(self.point_mass)
        result = (self.posterior_weights[:, offset:] * self._component_cdf(values[:, None])).sum(
            axis=1
        )
        return result + self.null_probability * (values >= self.null_value)

    def quantile(self, prob):
        """Posterior generalized inverse CDF; levels by observations for 1D input."""
        probabilities = np.asarray(prob, dtype=float)
        scalar = probabilities.ndim == 0
        probabilities = np.atleast_1d(probabilities)
        if (
            probabilities.ndim != 1
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("prob must be a scalar or vector in [0, 1]")
        result = np.empty((len(probabilities), len(self.observations)))
        offset = int(self.point_mass)
        for j in range(len(self.observations)):
            weights = self.posterior_weights[j, offset:]
            atom = self.null_probability[j]
            a, b = self._a[j], self._b[j]
            distribution = gamma(a, scale=1 / b) if self.family == "poisson" else beta(a, b)
            left = float(np.dot(weights, distribution.cdf(self.null_value)))

            def equation(x):
                return float(np.dot(weights, distribution.cdf(x)) + atom * (x >= self.null_value))

            for i, p in enumerate(probabilities):
                if p == 0:
                    answer = self.null_value if atom == 1 else 0.0
                elif p == 1:
                    answer = (
                        self.null_value
                        if atom == 1
                        else (np.inf if self.family == "poisson" else 1.0)
                    )
                elif atom > 0 and left <= p <= left + atom:
                    answer = self.null_value
                else:
                    upper = (
                        1.0
                        if self.family == "binomial"
                        else max(self.null_value, float(np.max(distribution.ppf(p))))
                    )
                    if not np.isfinite(upper):
                        raise ArithmeticError("quantile bracket exceeds numerical support")
                    lower = 0.0
                    # Avoid solving across a discontinuity at the null atom.
                    if p < left:
                        upper = min(upper, self.null_value)

                        def target(x):
                            return float(np.dot(weights, distribution.cdf(x)) - p)
                    else:
                        lower = self.null_value if atom > 0 else 0.0

                        def target(x):
                            return equation(x) - p

                    answer = brentq(target, lower, upper, xtol=1e-12, rtol=1e-12)
                result[i, j] = answer
        return result[0] if scalar else result

    def interval(self, level=0.95):
        """Equal-tail posterior credible intervals, one row per observation."""
        level = _scalar(level, "level")
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        return self.quantile([(1 - level) / 2, (1 + level) / 2]).T

    def sample(self, size=1, seed=None):
        """Draw posterior samples; result shape is (size, observations)."""
        if (
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size < 1
        ):
            raise ValueError("size must be a positive integer")
        rng = np.random.default_rng(seed)
        result = np.empty((size, len(self.observations)))
        offset = int(self.point_mass)
        for j, weights in enumerate(self.posterior_weights):
            component = rng.choice(len(weights), size=size, p=weights)
            for k in np.unique(component):
                selected = component == k
                if offset and k == 0:
                    result[selected, j] = self.null_value
                elif self.family == "poisson":
                    result[selected, j] = rng.gamma(
                        self._a[j, k - offset], 1 / self._b[j, k - offset], selected.sum()
                    )
                else:
                    result[selected, j] = rng.beta(
                        self._a[j, k - offset], self._b[j, k - offset], selected.sum()
                    )
        return result

    @property
    def prior_mean(self):
        offset = int(self.point_mass)
        means = (
            self._prior_a / self._prior_b
            if self.family == "poisson"
            else self._prior_a / (self._prior_a + self._prior_b)
        )
        return float(
            np.dot(self.mixture_weights[offset:], means)
            + (self.mixture_weights[0] * self.null_value if offset else 0)
        )

    def prior_density(self, value):
        """Continuous prior density; the null atom is excluded."""
        values = np.asarray(value, dtype=float)
        if np.isnan(values).any():
            raise ValueError("value cannot contain NaN")
        distribution = (
            gamma(self._prior_a, scale=1 / self._prior_b)
            if self.family == "poisson"
            else beta(self._prior_a, self._prior_b)
        )
        return np.sum(
            distribution.pdf(values[..., None]) * self.mixture_weights[int(self.point_mass) :],
            axis=-1,
        )

    def prior_cdf(self, value):
        """Prior CDF, including the null atom."""
        values = np.asarray(value, dtype=float)
        if np.isnan(values).any():
            raise ValueError("value cannot contain NaN")
        distribution = (
            gamma(self._prior_a, scale=1 / self._prior_b)
            if self.family == "poisson"
            else beta(self._prior_a, self._prior_b)
        )
        result = np.sum(
            distribution.cdf(values[..., None]) * self.mixture_weights[int(self.point_mass) :],
            axis=-1,
        )
        return result + (
            self.mixture_weights[0] * (values >= self.null_value) if self.point_mass else 0
        )

    def summary(self, level=0.95):
        """Return numeric columns for reporting; no pandas dependency."""
        bounds = self.interval(level)
        return {
            "posterior_mean": self.posterior_mean.copy(),
            "posterior_sd": self.posterior_sd.copy(),
            "lower": bounds[:, 0],
            "upper": bounds[:, 1],
            "null_probability": self.null_probability.copy(),
            "probability_below_null": self.probability_below_null.copy(),
            "probability_above_null": self.probability_above_null.copy(),
            "lfsr": self.lfsr.copy(),
            "qvalue": self.qvalue.copy(),
            "svalue": self.svalue.copy(),
        }

    def to_dataframe(self, level=0.95):
        """Return summary columns as a pandas DataFrame (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(self.summary(level))

    def to_dict(self):
        return {
            "schema_version": 1,
            "result_type": "count",
            "model": self.family,
            "family": self.family,
            "observations": self.observations.tolist(),
            "auxiliary": self.auxiliary.tolist(),
            "null_value": self.null_value,
            "strengths": self.strengths.tolist(),
            "point_mass": self.point_mass,
            "mixture_weights": self.mixture_weights.tolist(),
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data):
        """Restore a stored mixture without optimizing weights again."""
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != 1
            or data.get("result_type") != "count"
        ):
            raise ValueError("unsupported count-result schema")
        if data.get("model") != data.get("family"):
            raise ValueError("inconsistent count-result family")
        return cls(
            data["model"],
            data["observations"],
            data["auxiliary"],
            data["null_value"],
            data["strengths"],
            data["point_mass"],
            data["mixture_weights"],
            data.get("diagnostics"),
        )

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _fit(
    family, observations, auxiliary, null_value, strengths, point_mass, null_weight, maxiter, tol
):
    y, aux, null, grid, _, _, loglik = _setup(
        family, observations, auxiliary, null_value, strengths, point_mass
    )
    null_weight = _scalar(null_weight, "null_weight")
    if null_weight < 1:
        raise ValueError("null_weight must be at least one")
    if not point_mass and null_weight != 1:
        raise ValueError("null_weight must equal one when point_mass is False")
    penalty = np.zeros(loglik.shape[1])
    if point_mass:
        penalty[0] = null_weight - 1
    weights, _, diagnostics = fit_weights(loglik, penalty=penalty, maxiter=maxiter, tol=tol)
    diagnostics["null_weight"] = null_weight
    return CountResult(family, y, aux, null, grid, point_mass, weights, diagnostics)


def fit_poisson(
    counts,
    exposure=1,
    *,
    null_value=1,
    strengths=None,
    point_mass=True,
    null_weight=1,
    maxiter=10000,
    tol=1e-8,
):
    """Shrink Poisson rates under a fitted mode-shared gamma mixture.

    ``counts[i] ~ Poisson(exposure[i] * rate[i])``. Continuous components are
    Gamma(shape=1+null_value*strength, rate=strength). The default grid is
    ``2**arange(-4, 9)``. ``null_weight-1`` is a nonnegative null-atom
    pseudo-count, not an observation weight. No missing values are accepted.
    """
    return _fit(
        "poisson", counts, exposure, null_value, strengths, point_mass, null_weight, maxiter, tol
    )


def fit_binomial(
    successes,
    trials,
    *,
    null_value=0.5,
    strengths=None,
    point_mass=True,
    null_weight=1,
    maxiter=10000,
    tol=1e-8,
):
    """Shrink binomial probabilities under a fitted mode-shared beta mixture.

    Components are Beta(1+null_value*strength, 1+(1-null_value)*strength).
    The default grid is ``2**arange(-4, 9)``. Trials may be scalar; zero trials
    are allowed only with zero successes and contribute no likelihood evidence.
    ``null_weight-1`` penalizes the null atom as a nonnegative pseudo-count.
    """
    return _fit(
        "binomial", successes, trials, null_value, strengths, point_mass, null_weight, maxiter, tol
    )

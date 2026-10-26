"""Scalar empirical-Bayes shrinkage with Gaussian or Student-t errors.

Mixture scales and modes are fixed while weights are fitted. Normal errors
use analytic component posteriors; Student-t errors use adaptive quadrature.
Ranked error summaries group tied local probabilities together.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
import json
import math
import warnings
import numpy as np
from scipy import integrate, optimize, special, stats
from ._utils import readonly, fit_weights, tail_average
from .priors import MixturePrior, normal_mixture, uniform_mixture


class _Atom:
    mean = 0.0
    variance = 0.0
    lower = upper = 0.0

    def cdf(self, x):
        return float(x >= 0)

    def ppf(self, p):
        return np.zeros_like(p, dtype=float)


class _Analytic:
    def __init__(self, distribution, lower=-np.inf, upper=np.inf):
        self.distribution = distribution
        self.lower, self.upper = lower, upper
        self.mean = float(distribution.mean())
        self.variance = float(distribution.var())
        if not np.isfinite(self.mean) or not np.isfinite(self.variance) or self.variance < -1e-10:
            raise ArithmeticError("nonfinite analytic posterior moments; rescale the inputs")
        self.variance = max(0.0, self.variance)

    def cdf(self, x):
        return float(self.distribution.cdf(x))

    def ppf(self, p):
        return self.distribution.ppf(p)


def _log_normal_interval(lo, hi):
    # Reflect positive intervals so subtraction occurs in the stable tail.
    if lo > 0:
        lo, hi = -hi, -lo
    log_hi, log_lo = special.log_ndtr(hi), special.log_ndtr(lo)
    if log_hi == log_lo:
        return -np.inf
    return float(log_hi + np.log(-np.expm1(log_lo - log_hi)))


class _Numerical:
    """A normalized, one-dimensional Student-t component posterior.

    Integration uses the prior's scale coordinate. Likelihood peaks and zero
    divide the integration range, preventing narrow central peaks from being
    skipped by a single integral over an infinite interval.
    """

    def __init__(self, effect, se, df, family, scale, lower, upper):
        self.scale = scale
        self.lower = -np.inf if family == "normal" else lower
        self.upper = np.inf if family == "normal" else upper
        self.lo, self.hi = self.lower / scale, self.upper / scale
        center, width = effect / scale, se / scale
        self.points = sorted(
            set(
                p
                for p in [
                    0.0,
                    -1.0,
                    1.0,
                    center,
                    center - width,
                    center + width,
                    center - 8 * width,
                    center + 8 * width,
                ]
                if np.isfinite(p) and self.lo < p < self.hi
            )
        )
        logconstant = float(stats.t.logpdf(0.0, df) - np.log(se))
        prior_constant = (
            -0.5 * math.log(2 * math.pi) if family == "normal" else -math.log(self.hi - self.lo)
        )

        def kernel(u):
            z = (effect - scale * u) / se
            logprior = prior_constant - 0.5 * u * u if family == "normal" else prior_constant
            # logaddexp avoids squaring extreme standardized residuals.
            logquadratic = (
                0.0 if z == 0 else float(np.logaddexp(0.0, 2 * math.log(abs(z)) - math.log(df)))
            )
            return logconstant - 0.5 * (df + 1) * logquadratic + logprior

        self.kernel = kernel
        candidates = (
            self.points
            + ([self.lo] if np.isfinite(self.lo) else [])
            + ([self.hi] if np.isfinite(self.hi) else [])
        )
        # Heavy-tailed likelihoods can produce more than one posterior mode.
        # Search between existing peak candidates instead of assuming unimodality.
        searchlo = self.lo if np.isfinite(self.lo) else min(-10.0, center - 1.0)
        searchhi = self.hi if np.isfinite(self.hi) else max(10.0, center + 1.0)
        searchpoints = sorted(set([searchlo, searchhi] + self.points))
        for a, b in zip(searchpoints[:-1], searchpoints[1:]):
            if b > a:
                optimum = optimize.minimize_scalar(
                    lambda u: -kernel(u), bounds=(a, b), method="bounded"
                )
                if optimum.success:
                    candidates.append(float(optimum.x))
        self.offset = max(kernel(u) for u in candidates)
        self.warning_count = 0
        self.error_max = 0.0
        self.normalizer, error = self._integral(self.lo, self.hi)
        if not np.isfinite(self.normalizer) or self.normalizer <= 0:
            raise ArithmeticError("Student-t posterior normalization failed; rescale inputs")
        self.relative_error = error / self.normalizer
        if self.relative_error > 1e-4:
            raise ArithmeticError("Student-t posterior integral did not achieve useful accuracy")
        m1, _ = self._integral(self.lo, self.hi, moment=1)
        mean_u = m1 / self.normalizer
        self.mean = self.scale * mean_u
        m2, _ = self._integral(self.lo, self.hi, moment=2, center=mean_u)
        rawvar = self.scale**2 * m2 / self.normalizer
        if rawvar < -1e-7 * max(1.0, self.mean**2):
            raise ArithmeticError("Student-t posterior variance is numerically negative")
        self.variance = max(0.0, rawvar)
        self.log_marginal = float(self.offset + np.log(self.normalizer))

    def _integral(self, lo, hi, moment=0, center=0.0):
        if lo >= hi:
            return 0.0, 0.0
        points = [lo] + [p for p in self.points if lo < p < hi] + [hi]
        total, error = 0.0, 0.0

        def f(u):
            logvalue = self.kernel(u) - self.offset
            # Avoid an infinity times zero when QUADPACK probes far tails.
            if logvalue < -745:
                return 0.0
            return float(math.exp(logvalue) * ((u - center) ** moment if moment else 1.0))

        for a, b in zip(points[:-1], points[1:]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", integrate.IntegrationWarning)
                value, err = integrate.quad(f, a, b, epsabs=1e-10, epsrel=1e-9, limit=200)
            self.warning_count += sum(
                issubclass(w.category, integrate.IntegrationWarning) for w in caught
            )
            total += value
            error += err
        self.error_max = max(self.error_max, error)
        return total, error

    def cdf(self, x):
        if x <= self.lower:
            return 0.0
        if x >= self.upper:
            return 1.0
        value, _ = self._integral(self.lo, x / self.scale)
        return float(np.clip(value / self.normalizer, 0, 1))

    def ppf(self, probabilities):
        def one(p):
            if p == 0:
                return self.lower
            if p == 1:
                return self.upper
            lo = (
                self.lower
                if np.isfinite(self.lower)
                else min(-self.scale, self.mean - 8 * np.sqrt(self.variance))
            )
            hi = (
                self.upper
                if np.isfinite(self.upper)
                else max(self.scale, self.mean + 8 * np.sqrt(self.variance))
            )
            for _ in range(100):
                if self.cdf(lo) <= p:
                    break
                lo = 2 * lo - self.scale
            for _ in range(100):
                if self.cdf(hi) >= p:
                    break
                hi = 2 * hi + self.scale
            return optimize.brentq(
                lambda v: self.cdf(v) - p, lo, hi, xtol=1e-11 * max(self.scale, 1e-8)
            )

        values = np.asarray(probabilities, dtype=float)
        return np.array([one(p) for p in values.ravel()]).reshape(values.shape)


def _components(effects, standard_errors, dfs, prior):
    n, k = len(effects), len(prior.weights)
    loglik = np.empty((n, k))
    components = []
    for i, (effect, se, df) in enumerate(zip(effects, standard_errors, dfs)):
        row = []
        for j, (scale, lower, upper) in enumerate(zip(prior.scales, prior.lower, prior.upper)):
            if scale == 0:
                loglik[i, j] = (
                    stats.norm.logpdf(effect, scale=se)
                    if np.isinf(df)
                    else stats.t.logpdf(effect / se, df) - np.log(se)
                )
                component = _Atom()
            elif np.isfinite(df):
                component = _Numerical(effect, se, df, prior.family, scale, lower, upper)
                loglik[i, j] = component.log_marginal
            elif prior.family == "normal":
                marginal_sd = np.hypot(se, scale)
                loglik[i, j] = stats.norm.logpdf(effect, scale=marginal_sd)
                shrinkage = (scale / marginal_sd) ** 2
                component = _Analytic(
                    stats.norm(loc=effect * shrinkage, scale=se * (scale / marginal_sd))
                )
            else:
                a, b = (lower - effect) / se, (upper - effect) / se
                loglik[i, j] = _log_normal_interval(a, b) - np.log(upper - lower)
                component = _Analytic(stats.truncnorm(a, b, loc=effect, scale=se), lower, upper)
            row.append(component)
        components.append(tuple(row))
    if not np.isfinite(loglik).any(axis=1).all():
        raise ArithmeticError("all component likelihoods vanished for an observation")
    return loglik, tuple(components)


def _inputs(effects, standard_errors, df, missing, labels):
    def raw(value, name):
        a = np.asarray(value)
        if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a nonempty numeric vector")
        return np.array(a, dtype=float, copy=True)

    effects, se = raw(effects, "effects"), raw(standard_errors, "standard_errors")
    if effects.shape != se.shape:
        raise ValueError("effects and standard_errors must have the same length")
    if np.any(se <= 0):
        raise ValueError("standard_errors must be positive; nonpositive values cannot be dropped")
    if missing not in ("raise", "drop"):
        raise ValueError("missing must be 'raise' or 'drop'")
    if df is None:
        dfs = np.full(len(effects), np.inf)
    else:
        original = np.asarray(df)
        if original.dtype.kind not in "iuf" or original.ndim > 1:
            raise ValueError("df must be a positive number or aligned vector")
        dfs = (
            np.full(len(effects), float(original))
            if original.ndim == 0
            else np.array(original, dtype=float, copy=True)
        )
        if dfs.shape != effects.shape:
            raise ValueError("df must be scalar or have one entry per effect")
        if np.any(dfs <= 0):
            raise ValueError("df must be positive; nonpositive values cannot be dropped")
    valid = np.isfinite(effects) & np.isfinite(se) & ~np.isnan(dfs)
    if missing == "raise" and not valid.all():
        raise ValueError(
            'nonfinite effects, standard_errors or missing df; use missing="drop" explicitly'
        )
    if not valid.any():
        raise ValueError("no complete observations remain")
    names = None
    if labels is not None:
        labels = np.asarray(labels, dtype=object)
        if labels.shape != effects.shape or not all(isinstance(x, str) for x in labels):
            raise ValueError("labels must contain one string per input row")
        names = tuple(labels[valid])
    return effects[valid], se[valid], dfs[valid], np.flatnonzero(valid), len(effects), names


def _automatic_scales(effects, se):
    minimum = float(np.min(se) / 10)
    if minimum <= 0:
        raise ValueError("standard errors are too small for the automatic grid; rescale inputs")
    signal = np.abs(effects) > se
    if not signal.any():
        maximum = 8 * minimum
    else:
        magnitude = np.abs(effects[signal])
        maximum = 2 * np.max(magnitude * np.sqrt(1 - (se[signal] / magnitude) ** 2))
    if not np.isfinite(maximum):
        raise ValueError("input magnitudes exceed the automatic grid range; rescale inputs")
    maximum = max(maximum, minimum)
    count = max(0, int(np.ceil(np.log(maximum / minimum) / np.log(np.sqrt(2)))))
    return maximum * np.sqrt(2.0) ** np.arange(-count, 1)


@dataclass(frozen=True, eq=False)
class ASHResult:
    """Scalar mixture posterior summaries conditional on a fitted prior.

    ``lfdr`` is the posterior zero-atom probability. ``lfsr`` is the smaller
    of P(beta <= 0 | data) and P(beta >= 0 | data), so the atom belongs to
    both error events. Strict negative/positive probabilities exclude it.
    Credible intervals condition on estimated weights, not uncertainty in
    those weights. Missing rows are absent and ``row_indices`` maps back to
    the original input. Public numeric arrays are read-only.
    """

    effects: np.ndarray
    standard_errors: np.ndarray
    df: np.ndarray
    prior: MixturePrior
    posterior_mean: np.ndarray
    posterior_sd: np.ndarray
    lfdr: np.ndarray
    probability_negative: np.ndarray
    probability_positive: np.ndarray
    lfsr: np.ndarray
    qvalue: np.ndarray
    svalue: np.ndarray
    component_probabilities: np.ndarray
    row_indices: np.ndarray
    original_size: int
    labels: tuple
    diagnostics: object
    _posterior: tuple = field(repr=False)

    def __post_init__(self):
        for name in (
            "effects",
            "standard_errors",
            "df",
            "posterior_mean",
            "posterior_sd",
            "lfdr",
            "probability_negative",
            "probability_positive",
            "lfsr",
            "qvalue",
            "svalue",
            "component_probabilities",
            "row_indices",
        ):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def model(self):
        return "gaussian" if np.isinf(self.df).all() else "student_t"

    @property
    def kind(self):
        return self.model

    @property
    def null_probability(self):
        return self.lfdr

    @property
    def prior_mean(self):
        return self.prior.mean

    def prior_density(self, x):
        return self.prior.density(x)

    def prior_cdf(self, x):
        return self.prior.cdf(x)

    def cdf(self, value):
        """Posterior P(beta <= value), at a scalar or one value per row."""
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            value = np.full(len(self.effects), float(value))
        if value.shape != self.effects.shape or np.isnan(value).any():
            raise ValueError("value must be scalar or one nonmissing value per retained row")
        return np.clip(
            np.array(
                [
                    sum(p * c.cdf(x) for p, c in zip(probs, row) if p > 0)
                    for x, probs, row in zip(value, self.component_probabilities, self._posterior)
                ]
            ),
            0,
            1,
        )

    def _quantile_row(self, probability, i):
        probs, row = self.component_probabilities[i], self._posterior[i]
        active = [(p, c) for p, c in zip(probs, row) if p > 0]
        lower, upper = min(c.lower for _, c in active), max(c.upper for _, c in active)
        if probability == 0:
            return lower
        if probability == 1:
            return upper
        atom = self.lfdr[i]
        left = self.probability_negative[i]
        if atom > 0 and left <= probability <= left + atom:
            return 0.0

        def cdf(x):
            return sum(p * c.cdf(x) for p, c in active)

        scale = max(self.standard_errors[i], float(np.max(self.prior.scales)), 1e-8)
        lo = lower if np.isfinite(lower) else min(-scale, self.effects[i] - 8 * scale)
        hi = upper if np.isfinite(upper) else max(scale, self.effects[i] + 8 * scale)
        for _ in range(100):
            if cdf(lo) <= probability:
                break
            lo = 2 * lo - scale
        for _ in range(100):
            if cdf(hi) >= probability:
                break
            hi = 2 * hi + scale
        return optimize.brentq(
            lambda x: cdf(x) - probability, lo, hi, xtol=1e-10 * max(1e-8, scale)
        )

    def quantile(self, probability):
        """Quantiles: scalar -> (n,), levels vector -> (levels, n)."""
        values = np.asarray(probability, dtype=float)
        if (
            values.ndim > 1
            or values.size == 0
            or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
        ):
            raise ValueError("probability must be a scalar or nonempty vector in [0,1]")
        output = np.array(
            [
                [self._quantile_row(float(p), i) for i in range(len(self.effects))]
                for p in values.reshape(-1)
            ]
        )
        return output[0] if values.ndim == 0 else output

    def interval(self, level=0.95):
        """Equal-tailed posterior interval, conditional on the fitted prior."""
        if np.ndim(level) != 0 or not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must be between zero and one")
        return self.quantile([(1 - level) / 2, (1 + level) / 2]).T

    def sample(self, size=1, seed=None):
        """Draw (size, n) posterior samples, conditional on the fitted prior."""
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer")
        rng = np.random.default_rng(seed)
        output = np.empty((size, len(self.effects)))
        for i, (probs, row) in enumerate(zip(self.component_probabilities, self._posterior)):
            choices = rng.choice(len(row), size=size, p=probs)
            for j, component in enumerate(row):
                selected = choices == j
                if selected.any():
                    output[selected, i] = component.ppf(
                        rng.uniform(np.nextafter(0.0, 1.0), 1.0, selected.sum())
                    )
        return output

    def summary(self, level=0.95):
        """Numeric reporting columns, without a pandas dependency."""
        bounds = self.interval(level)
        result = dict(
            row=self.row_indices.copy(),
            effect=self.effects.copy(),
            standard_error=self.standard_errors.copy(),
            posterior_mean=self.posterior_mean.copy(),
            posterior_sd=self.posterior_sd.copy(),
            lower=bounds[:, 0],
            upper=bounds[:, 1],
            lfdr=self.lfdr.copy(),
            null_probability=self.lfdr.copy(),
            lfsr=self.lfsr.copy(),
            qvalue=self.qvalue.copy(),
            svalue=self.svalue.copy(),
            probability_negative=self.probability_negative.copy(),
            probability_positive=self.probability_positive.copy(),
        )
        if self.labels is not None:
            result["label"] = np.array(self.labels, dtype=object)
        return result

    def format_summary(self, level=0.95, digits=6):
        """A compact text summary with at most twelve rows."""
        if (
            isinstance(digits, bool)
            or not isinstance(digits, (int, np.integer))
            or not 1 <= digits <= 16
        ):
            raise ValueError("digits must be an integer between one and sixteen")
        columns = self.summary(level)
        header = f"AShrink {self.model}: {len(self.effects)} effects; {self.prior.family} prior"
        info = (
            "fixed prior"
            if self.diagnostics.get("fixed_prior")
            else f"Optimizer converged: {self.diagnostics.get('converged')}; iterations: {self.diagnostics.get('iterations')}"
        )
        lines = [
            header,
            info,
            "row     post.mean       post.sd         lower         upper          lfsr",
        ]
        for i in range(min(12, len(self.effects))):
            values = [
                columns[name][i]
                for name in ("posterior_mean", "posterior_sd", "lower", "upper", "lfsr")
            ]
            lines.append(
                f"{self.row_indices[i]:>3d} " + " ".join(f"{v:>13.{digits}g}" for v in values)
            )
        if len(self.effects) > 12:
            lines.append(f"... {len(self.effects) - 12} more rows")
        lines.append(
            f"{100 * level:g}% equal-tailed intervals condition on the fitted mixture weights."
        )
        return "\n".join(lines)

    def __str__(self):
        return self.format_summary()

    def to_dataframe(self, level=0.95):
        """Return summary columns; requires the optional pandas dependency."""
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("to_dataframe requires pandas; install ashrink[table]") from error
        return pd.DataFrame(self.summary(level))

    def shrink(self, effects, standard_errors, df=None, *, missing="raise", labels=None):
        """Apply the fitted prior to new observations without estimating weights.

        Student-t fits require degrees of freedom for the new observations;
        degrees of freedom attached to old rows are not silently reused.
        """
        if self.model == "student_t" and df is None:
            raise ValueError("supply df for new observations when reusing a Student-t result")
        return fit(
            effects,
            standard_errors,
            df=df,
            prior=self.prior,
            fix_prior=True,
            missing=missing,
            labels=labels,
        )

    def to_dict(self):
        """JSON-compatible inputs, fitted prior and diagnostics (schema 1)."""
        return {
            "schema_version": 1,
            "model": self.model,
            "effects": self.effects.tolist(),
            "standard_errors": self.standard_errors.tolist(),
            "df": [None if np.isinf(x) else float(x) for x in self.df],
            "prior": self.prior.to_dict(),
            "row_indices": self.row_indices.tolist(),
            "original_size": int(self.original_size),
            "labels": None if self.labels is None else list(self.labels),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value):
        """Rebuild component posteriors from stored inputs without fitting weights."""
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("model") not in ("gaussian", "student_t")
        ):
            raise ValueError("unsupported scalar result schema or model")
        degrees = [np.inf if x is None else x for x in value["df"]]
        result = fit(
            value["effects"],
            value["standard_errors"],
            df=degrees,
            prior=MixturePrior.from_dict(value["prior"]),
            fix_prior=True,
            labels=value.get("labels"),
        )
        indices = np.asarray(value["row_indices"])
        original = value["original_size"]
        if (
            indices.dtype.kind not in "iu"
            or indices.shape != result.effects.shape
            or isinstance(original, bool)
            or not isinstance(original, int)
            or original < len(indices)
            or (indices < 0).any()
            or (indices >= original).any()
            or (np.diff(indices) <= 0).any()
        ):
            raise ValueError("invalid retained-row metadata")
        if result.model != value["model"] or not isinstance(value.get("diagnostics"), dict):
            raise ValueError("inconsistent model or diagnostics")
        return replace(
            result, row_indices=indices, original_size=original, diagnostics=value["diagnostics"]
        )

    def save(self, path):
        """Save a versioned JSON result, without pickle or executable objects."""
        Path(path).write_text(
            json.dumps(self.to_dict(), allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit(
    effects,
    standard_errors,
    *,
    family="normal",
    df=None,
    scales=None,
    point_mass=True,
    null_weight=1,
    prior=None,
    fix_prior=False,
    maxiter=10000,
    tol=1e-8,
    missing="raise",
    labels=None,
):
    """Fit a zero-mode scalar mixture prior and obtain posterior summaries.

    ``family`` chooses the prior: normal, uniform, or halfuniform. Errors are
    Gaussian unless positive finite ``df`` requests a Student-t likelihood;
    standard_errors is then the Student-t scale, not its standard deviation.
    Infinite df selects Gaussian errors. Finite and infinite df may be mixed.
    Supplied scales are normal SDs or uniform half-widths. The default grid
    follows ashr 2.2-63's geometric rule, with spacing sqrt(2).

    ``null_weight >= 1`` adds (null_weight - 1) log(pi0) to the objective.
    A supplied prior supplies the grid and starting weights; fix_prior=True
    also fixes those weights. Missing rows are rejected unless missing='drop'.
    Neither intervals nor local error probabilities account for uncertainty
    in estimated mixture weights. Student-t quadrature is slower than the
    analytic Gaussian path; inspect diagnostics for integration accuracy.
    """
    if family not in ("normal", "uniform", "halfuniform"):
        raise ValueError("family must be normal, uniform or halfuniform")
    if not isinstance(fix_prior, (bool, np.bool_)):
        raise ValueError("fix_prior must be boolean")
    if np.ndim(null_weight) != 0 or not np.isfinite(null_weight) or null_weight < 1:
        raise ValueError("null_weight must be finite and at least one")
    effects, se, dfs, indices, original, names = _inputs(
        effects, standard_errors, df, missing, labels
    )
    if prior is None:
        if fix_prior:
            raise ValueError("fix_prior=True requires a supplied prior")
        grid = _automatic_scales(effects, se) if scales is None else scales
        prior = (
            normal_mixture(grid, point_mass=point_mass)
            if family == "normal"
            else uniform_mixture(grid, asymmetric=(family == "halfuniform"), point_mass=point_mass)
        )
    else:
        if not isinstance(prior, MixturePrior):
            raise ValueError("prior must be a MixturePrior")
        if scales is not None:
            raise ValueError("supply scales or a prior, not both")
    if null_weight > 1 and prior.atom_index is None:
        raise ValueError("null_weight > 1 requires a zero atom")
    loglik, components = _components(effects, se, dfs, prior)
    if fix_prior:
        with np.errstate(divide="ignore"):
            logjoint = loglik + np.log(prior.weights)
        logmarg = special.logsumexp(logjoint, axis=1)
        if not np.isfinite(logmarg).all():
            raise ArithmeticError("supplied positive-weight components do not support all rows")
        probabilities = np.exp(logjoint - logmarg[:, None])
        weights = prior.weights
        diagnostics = {
            "fixed_prior": True,
            "converged": None,
            "iterations": 0,
            "log_likelihood": float(logmarg.sum()),
            "optimizer": None,
        }
    else:
        penalty = np.zeros(len(prior.weights))
        if prior.atom_index is not None:
            penalty[prior.atom_index] = null_weight - 1
        # An unfixed grid permits zero initial weights to re-enter the fit.
        initial = np.maximum(prior.weights, 1e-12)
        weights, probabilities, diagnostics = fit_weights(
            loglik, penalty=penalty, initial=initial, maxiter=maxiter, tol=tol
        )
        diagnostics.update(fixed_prior=False, null_weight=float(null_weight))
        if not diagnostics["converged"]:
            warnings.warn(
                "mixture weights did not converge; inspect result.diagnostics",
                RuntimeWarning,
                stacklevel=2,
            )
    prior = prior.with_weights(weights)
    means = np.array([[c.mean for c in row] for row in components])
    variances = np.array([[c.variance for c in row] for row in components])
    postmean = (probabilities * means).sum(axis=1)
    # Centred second moments avoid cancellation for highly precise estimates.
    postvar = (probabilities * (variances + (means - postmean[:, None]) ** 2)).sum(axis=1)
    if not np.isfinite(postmean).all() or not np.isfinite(postvar).all():
        raise ArithmeticError("posterior moments exceed numerical range; rescale inputs")
    null = (
        np.zeros(len(effects)) if prior.atom_index is None else probabilities[:, prior.atom_index]
    )
    negative = np.array(
        [
            sum(
                p * c.cdf(0.0)
                for j, (p, c) in enumerate(zip(probs, row))
                if j != prior.atom_index and p > 0
            )
            for probs, row in zip(probabilities, components)
        ]
    )
    positive = np.clip(1 - negative - null, 0, 1)
    lfsr = np.clip(np.minimum(negative + null, positive + null), 0, 1)
    numerical = [c for row in components for c in row if isinstance(c, _Numerical)]
    diagnostics.update(
        likelihood="normal" if np.isinf(dfs).all() else "student_t",
        quadrature_components=len(numerical),
        quadrature_max_relative_normalization_error=max(
            (c.relative_error for c in numerical), default=0.0
        ),
        quadrature_max_absolute_scaled_error=max((c.error_max for c in numerical), default=0.0),
        quadrature_warnings=sum(c.warning_count for c in numerical),
        quadrature_tolerance=1e-9 if numerical else None,
        ranked_ties="all equal local probabilities included",
        original_rows=int(original),
        dropped_rows=int(original - len(effects)),
    )
    return ASHResult(
        effects,
        se,
        dfs,
        prior,
        postmean,
        np.sqrt(np.maximum(postvar, 0)),
        null,
        np.clip(negative, 0, 1),
        positive,
        lfsr,
        tail_average(null),
        tail_average(lfsr),
        probabilities,
        indices,
        original,
        names,
        diagnostics,
        components,
    )

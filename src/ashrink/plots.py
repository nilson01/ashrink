"""Matplotlib displays of fitted priors and posterior summaries.

Plots return axes without displaying or saving a figure. Matplotlib is imported
only when a plot function is called.
"""

import numpy as np

_BLUE = "#0072B2"
_RED = "#B54724"


def _axis(ax):
    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "install AShrink with the plot extra: pip install 'ashrink[plot]'"
            ) from exc
        _, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
    return ax


def _column(result, value, condition):
    value = np.asarray(value)
    if np.asarray(result.posterior_mean).ndim == 2:
        d = np.asarray(result.posterior_mean).shape[1]
        if (
            isinstance(condition, bool)
            or not isinstance(condition, (int, np.integer))
            or not 0 <= condition < d
        ):
            raise ValueError("condition must identify an existing matrix column")
        return value[:, condition]
    if condition != 0:
        raise ValueError("scalar models have only condition=0")
    return value


def plot_shrinkage(result, *, condition=0, ax=None):
    """Compare input estimates with posterior means on the same scale.

    Binomial rows with zero trials have no observed proportion and are omitted
    from this display; they remain in the fitted result.
    """
    if str(getattr(result, "model", "")).endswith("regression"):
        raise ValueError("use plot_coefficients for regression results")
    mean = _column(result, result.posterior_mean, condition)
    if hasattr(result, "effects"):
        observed = _column(result, result.effects, condition)
    elif result.family == "poisson":
        observed = result.counts / result.exposure
    else:
        observed = np.divide(
            result.successes, result.trials, out=np.full_like(mean, np.nan), where=result.trials > 0
        )
    keep = np.isfinite(observed)
    if not keep.any():
        raise ValueError("there are no finite input estimates to plot")
    ax = _axis(ax)
    ax.scatter(observed[keep], mean[keep], color=_BLUE, marker="o", s=22, label="Posterior mean")
    low = min(observed[keep].min(), mean[keep].min())
    high = max(observed[keep].max(), mean[keep].max())
    if low == high:
        low, high = low - 1, high + 1
    ax.plot([low, high], [low, high], color="#555555", ls="--", lw=0.9, label="Unchanged estimate")
    ax.set(xlabel="Input estimate", ylabel="Posterior mean")
    ax.legend(frameon=False)
    return ax


def plot_intervals(result, *, condition=0, level=0.95, indices=None, ax=None):
    """Show equal-tailed posterior credible intervals and posterior means.

    Each horizontal line represents a retained observation. The interval
    conditions on fitted mixture weights; it is not a frequentist confidence
    interval or an interval for a future observation.
    """
    mean = _column(result, result.posterior_mean, condition)
    intervals = _column(result, result.interval(level), condition)
    selected = np.arange(len(mean)) if indices is None else np.asarray(indices)
    if selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected):
        raise ValueError("indices must be a nonempty integer vector")
    if (
        (selected < 0).any()
        or (selected >= len(mean)).any()
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("indices must be distinct existing row indices")
    if not np.isfinite(intervals[selected]).all():
        raise ValueError("interval bounds must be finite for plotting")
    ax = _axis(ax)
    y = np.arange(len(selected))
    ax.hlines(
        y,
        intervals[selected, 0],
        intervals[selected, 1],
        color=_BLUE,
        lw=1.2,
        label=f"{level:.0%} credible interval",
    )
    ax.scatter(mean[selected], y, color=_BLUE, marker="o", s=20, zorder=3)
    null = float(getattr(result, "null_value", 0.0))
    ax.axvline(null, color="#555555", ls="--", lw=0.8)
    labels = getattr(result, "labels", None)
    ax.set(
        yticks=y,
        yticklabels=[str(labels[i]) if labels is not None else str(i) for i in selected],
        xlabel="Parameter",
        ylabel="Observation",
    )
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    return ax


def plot_diagnostics(result, *, ax=None):
    """Display the recorded optimization objective by optimization iteration."""
    history = result.diagnostics.get("objective_history")
    if history is None or not len(history):
        raise ValueError("this result has no optimization history")
    ax = _axis(ax)
    ax.plot(np.arange(1, len(history) + 1), history, color=_BLUE, lw=1, marker="o", markersize=2)
    objective = (
        "Evidence lower bound"
        if getattr(result, "model", None)
        in (
            "gaussian_regression",
            "gaussian_mixed_regression",
            "logistic_regression",
            "poisson_regression",
        )
        else "Penalized log likelihood"
    )
    ax.set(xlabel="Optimization iteration", ylabel=objective)
    return ax


def plot_prior(result, values=None, *, condition=0, ax=None):
    """Display continuous prior density; show point-null mass separately.

    An atom is a probability, not a density height. Its mass is labelled in
    the legend rather than drawn as an infinite density spike.
    """
    if str(getattr(result, "model", "")).endswith("regression"):
        raise ValueError("plot_prior supports scalar, count and multivariate summary fits")
    mean = _column(result, result.posterior_mean, condition)
    sd = _column(result, result.posterior_sd, condition)
    if values is None:
        width = max(float(np.max(np.abs(mean)) + 4 * np.max(sd)), 1.0)
        family = getattr(result, "family", "")
        if family == "poisson":
            values = np.linspace(0, max(width, 2 * result.null_value), 300)
        elif family == "binomial":
            values = np.linspace(0, 1, 300)
        else:
            values = np.linspace(-width, width, 300)
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or (np.diff(values) <= 0).any()
    ):
        raise ValueError("values must be a finite, strictly increasing grid")
    if np.asarray(result.posterior_mean).ndim == 2:
        density = result.prior_density(values, condition=condition)
        atom = np.asarray(result.prior_null_probability)[condition]
    else:
        density = result.prior_density(values)
        prior = getattr(result, "prior", None)
        if prior is not None:
            atom = float(getattr(prior, "null_probability", 0.0))
        else:
            atom = result.mixture_weights[0] if result.point_mass else 0.0
    ax = _axis(ax)
    ax.plot(values, density, color=_BLUE, lw=1.2, label="Continuous prior density")
    if atom > 0:
        ax.plot(
            [], [], color=_RED, marker="D", ls="none", label=f"Point-null probability: {atom:.3g}"
        )
    ax.set(xlabel="Parameter", ylabel="Density")
    ax.legend(frameon=False)
    return ax


def plot_coefficients(result, *, level=0.95, indices=None, ax=None):
    """Show approximate marginal regression intervals on the coefficient scale.

    These variational intervals condition on fitted hyperparameters; they do not
    include uncertainty in the intercept or residual variance.
    """
    if getattr(result, "model", None) not in (
        "gaussian_regression",
        "gaussian_mixed_regression",
        "logistic_regression",
        "poisson_regression",
    ):
        raise ValueError("plot_coefficients requires a regression result")
    ax = plot_intervals(result, level=level, indices=indices, ax=ax)
    ax.set(xlabel="Coefficient", ylabel="Predictor")
    return ax


def plot_residuals(result, *, ax=None):
    """Plot observed residuals against fitted response means."""
    if getattr(result, "model", None) not in (
        "gaussian_regression",
        "gaussian_mixed_regression",
        "logistic_regression",
        "poisson_regression",
    ):
        raise ValueError("plot_residuals requires a regression result")
    ax = _axis(ax)
    ax.scatter(result.fitted_values, result.residuals, color=_BLUE, s=22)
    ax.axhline(0, color="#555555", ls="--", lw=0.8)
    ax.set(xlabel="Fitted mean", ylabel="Residual")
    return ax


def plot_random_effects(result, *, level=0.95, ax=None):
    """Display approximate Gaussian random-intercept marginal intervals."""
    if getattr(result, "model", None) != "gaussian_mixed_regression":
        raise ValueError("plot_random_effects requires a random-intercept result")
    bounds = result.random_effect_interval(level)
    ax = _axis(ax)
    y = np.arange(result.n_groups)
    ax.hlines(
        y,
        bounds[:, 0],
        bounds[:, 1],
        color=_BLUE,
        lw=1.2,
        label=f"Approximate {level:.0%} credible interval",
    )
    ax.scatter(result.random_effects, y, color=_BLUE, s=20)
    ax.axvline(0, color="#555555", ls="--", lw=0.8)
    ax.set(yticks=y, yticklabels=result.group_labels, xlabel="Random intercept", ylabel="Group")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    return ax

"""Likelihood limits and variational stationary equations for binary/count fits."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from ashrink import (
    fit_logistic_regression,
    fit_poisson_regression,
    fit_logistic_regression_frame,
    fit_poisson_regression_frame,
    fit_mixed_regression_frame,
    save_result,
    load_result,
    to_records,
    plot_coefficients,
    plot_residuals,
    plot_diagnostics,
)

X = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5]])
BINARY = np.array([0, 0, 1, 0, 1, 1])
COUNTS = np.array([0, 1, 1, 2, 3, 4])


def test_logistic_null_only_recovers_binomial_intercept_likelihood():
    trials = np.array([2, 4, 2, 3, 1, 2])
    y = np.array([0, 1, 1, 2, 1, 1])
    r = fit_logistic_regression(X, y, trials=trials, scales=[0], tol=1e-12)
    probability = y.sum() / trials.sum()
    assert_allclose(r.intercept, np.log(probability / (1 - probability)), atol=1e-10)
    assert_allclose(r.predict(X), probability, atol=1e-10)
    assert_allclose(r.predict(X, trials=trials), r.fitted_values, atol=1e-12)
    assert_allclose(r.lfdr, 1)
    assert_allclose(r.interval(), 0)
    assert not hasattr(r, "residual_variance")


def test_poisson_null_only_recovers_exposure_adjusted_likelihood():
    exposure = np.array([0.5, 1.0, 2.0, 1.0, 3.0, 2.0])
    r = fit_poisson_regression(X, COUNTS, exposure=exposure, scales=[0], tol=1e-12)
    rate = COUNTS.sum() / exposure.sum()
    assert_allclose(r.intercept, np.log(rate), atol=1e-11)
    assert_allclose(r.predict(X), rate, atol=1e-11)
    assert_allclose(r.predict(X, exposure=exposure), r.fitted_values, atol=1e-11)
    assert_allclose(r.predict(X, exposure=exposure * 2), 2 * r.fitted_values, atol=1e-11)
    assert not hasattr(r, "residual_variance")


def test_logistic_single_normal_factor_satisfies_bound_stationarity():
    design = np.ones((5, 1))
    y = np.array([0, 0, 1, 1, 1])
    r = fit_logistic_regression(
        design,
        y,
        scales=[1.2],
        point_mass=False,
        fit_intercept=False,
        standardize=False,
        estimate_weights=False,
        tol=1e-13,
    )
    m, v = r.coefficients[0], r.posterior_sd[0] ** 2
    xi = np.sqrt(m * m + v)
    curvature = np.tanh(xi / 2) / (2 * xi)
    expected_variance = 1 / (5 * curvature + 1 / 1.2**2)
    assert_allclose(v, expected_variance, rtol=2e-6)
    assert_allclose(m, v * (y.sum() - 2.5), atol=2e-7)
    assert_allclose(r.predict(design), expit(m), atol=1e-12)


def test_poisson_single_normal_factor_satisfies_exact_expected_likelihood_stationarity():
    design = np.ones((4, 1))
    y = np.array([0, 1, 3, 1])
    r = fit_poisson_regression(
        design,
        y,
        scales=[1.2],
        point_mass=False,
        fit_intercept=False,
        standardize=False,
        estimate_weights=False,
        tol=1e-12,
        inner_tol=1e-10,
    )
    m, v = r.coefficients[0], r.posterior_sd[0] ** 2
    expected_total = 4 * np.exp(m + v / 2)
    assert_allclose(y.sum() - expected_total - m / 1.2**2, 0, atol=2e-7)
    assert_allclose(1 / v - expected_total - 1 / 1.2**2, 0, atol=2e-7)
    # The response prediction is plug-in exp(E b), not E exp(b).
    assert_allclose(r.predict(design), np.exp(m), atol=1e-12)


@pytest.mark.parametrize("model", ["logistic", "poisson"])
def test_glm_elbo_increases_and_serialization_keeps_factors(model, tmp_path):
    function, y = (
        (fit_logistic_regression, BINARY)
        if model == "logistic"
        else (fit_poisson_regression, COUNTS)
    )
    r = function(X, y, scales=[0.5, 1.0], tol=1e-9)
    assert r.diagnostics["converged"]
    assert np.min(np.diff(r.diagnostics["objective_history"])) >= -1e-8
    path = tmp_path / (model + ".json")
    save_result(r, path)
    restored = load_result(path)
    assert restored.model == r.model
    assert_allclose(restored.predict(X), r.predict(X))
    assert_allclose(restored.interval(), r.interval())
    assert_allclose(restored.responsibilities, r.responsibilities)
    assert len(to_records(restored)) == 1
    assert "Approximate" in restored.format_summary()
    assert "residual_variance" not in r.to_dict()["coefficient_distribution"]


@pytest.mark.parametrize("model", ["logistic", "poisson"])
def test_glm_standardization_and_shift_predictions(model):
    function, y = (
        (fit_logistic_regression, BINARY)
        if model == "logistic"
        else (fit_poisson_regression, COUNTS)
    )
    base = function(X, y, scales=[0.5, 1.0], tol=1e-10)
    changed = function(3 * X + 7, y, scales=[0.5, 1.0], tol=1e-10)
    assert_allclose(changed.coefficients * 3, base.coefficients, atol=1e-7)
    assert_allclose(changed.predict(3 * X + 7), base.predict(X), atol=1e-7)


def test_named_regression_adapters():
    data = {
        "x": X[:, 0],
        "binary": BINARY,
        "count": COUNTS,
        "e": np.ones(6),
        "group": ["a"] * 3 + ["b"] * 3,
    }
    assert fit_logistic_regression_frame(data, "binary", ["x"], scales=[0]).labels == ("x",)
    assert fit_poisson_regression_frame(data, "count", ["x"], exposure="e", scales=[0]).labels == (
        "x",
    )
    assert (
        fit_mixed_regression_frame(data, "count", ["x"], group="group", scales=[0.5]).n_groups == 2
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: fit_logistic_regression(X, [0] * 6),
        lambda: fit_logistic_regression(X, [2] * 6),
        lambda: fit_logistic_regression(X, BINARY, trials=0),
        lambda: fit_logistic_regression(X, BINARY, trials=1.5),
        lambda: fit_poisson_regression(X, COUNTS, exposure=0),
        lambda: fit_poisson_regression(X, [0.5] * 6),
        lambda: fit_poisson_regression(X, [0] * 6),
        lambda: fit_poisson_regression(X, COUNTS, inner_maxiter=0),
    ],
)
def test_glm_invalid_input_rejects(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("model", ["logistic", "poisson"])
def test_glm_response_plots(model):
    import matplotlib.pyplot as plt

    r = (
        fit_logistic_regression(X, BINARY, scales=[0.5])
        if model == "logistic"
        else fit_poisson_regression(X, COUNTS, scales=[0.5])
    )
    for f in [plot_coefficients, plot_residuals, plot_diagnostics]:
        ax = f(r)
        assert ax.get_xlabel()
        plt.close(ax.figure)


def test_poisson_underflow_rejects_response_but_retains_link_prediction():
    r = fit_poisson_regression(X, COUNTS, scales=[0])
    with pytest.raises(ValueError, match="underflow"):
        r.predict(X, offset=-1000)
    assert np.isfinite(r.predict(X, offset=-1000, scale="link")).all()


def test_zero_trial_rows_do_not_change_logistic_fitting():
    base = fit_logistic_regression(X, BINARY, scales=[0.5], tol=1e-10)
    added = fit_logistic_regression(
        np.vstack([X, [[20.0]]]),
        np.r_[BINARY, 0],
        trials=[1, 1, 1, 1, 1, 1, 0],
        scales=[0.5],
        tol=1e-10,
    )
    assert_allclose(added.coefficients, base.coefficients, atol=1e-9)
    assert_allclose(added.posterior_sd, base.posterior_sd, atol=1e-9)

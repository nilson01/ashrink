"""Gaussian regression checks against exact special cases and invariants."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from ashrink import (
    fit_regression,
    fit_regression_frame,
    save_result,
    load_result,
    to_records,
    plot_coefficients,
    plot_residuals,
    plot_diagnostics,
)

X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
Y = np.array([-1.8, -0.1, 0.6, 3.3])


def fixed_fit(X=X, y=Y, **options):
    return fit_regression(
        X,
        y,
        scales=[0.8],
        point_mass=False,
        initial_residual_variance=0.7,
        estimate_residual_variance=False,
        estimate_weights=False,
        tol=1e-12,
        **options,
    )


def test_orthogonal_normal_prior_exact_marginals():
    r = fixed_fit()
    centered_y = Y - Y.mean()
    variance = 0.7 / (4 + 1 / 0.8**2)
    mean = variance * (X.T @ centered_y) / 0.7
    assert_allclose(r.coefficients, mean, atol=1e-12)
    assert_allclose(r.posterior_sd**2, variance, atol=1e-12)
    assert_allclose(r.intercept, Y.mean())
    assert_allclose(r.cdf(0), norm.cdf(-mean / np.sqrt(variance)))
    assert_allclose(
        r.interval(), mean[:, None] + np.sqrt(variance) * norm.ppf([0.025, 0.975]), atol=1e-10
    )
    assert_allclose(r.predict(X), r.fitted_values)
    assert r.diagnostics["converged"]


def test_correlated_normal_prior_mean_agrees_with_ridge():
    design = np.array([[-1.0, -0.4], [0.0, 0.8], [1.0, 0.2], [2.0, 1.2], [-2.0, -0.9]])
    response = np.array([0.2, 0.8, 1.2, 2.8, -1.3])
    r = fixed_fit(design, response, standardize=False)
    centered = design - design.mean(axis=0)
    expected = np.linalg.solve(
        centered.T @ centered + np.eye(2) / 0.8**2, centered.T @ (response - response.mean())
    )
    assert_allclose(r.coefficients, expected, atol=2e-6)
    assert_allclose(r.intercept, response.mean() - design.mean(axis=0) @ expected, atol=2e-6)


def test_standardization_and_shift_are_reused_for_prediction():
    base = fixed_fit()
    scale, shift = np.array([3.0, 0.2]), np.array([8.0, -2.0])
    changed = fixed_fit(X * scale + shift, Y + 4)
    assert_allclose(changed.coefficients * scale, base.coefficients, atol=1e-11)
    assert_allclose(changed.posterior_sd * scale, base.posterior_sd, atol=1e-11)
    new = np.array([[0.3, -0.8], [2.0, 1.0]])
    assert_allclose(changed.predict(new * scale + shift), base.predict(new) + 4, atol=1e-11)


def test_no_intercept_uses_uncentered_design():
    r = fixed_fit(fit_intercept=False, standardize=False)
    expected = np.linalg.solve(X.T @ X + np.eye(2) / 0.8**2, X.T @ Y)
    assert r.intercept == 0
    assert_allclose(r.coefficients, expected, atol=1e-11)


def test_fixed_mixture_orthogonal_matches_independent_normal_means():
    from ashrink import fit, normal_mixture

    r = fit_regression(
        X,
        Y,
        scales=[0.4, 1.2],
        initial_weights=[0.3, 0.4, 0.3],
        initial_residual_variance=0.7,
        estimate_weights=False,
        estimate_residual_variance=False,
        tol=1e-12,
    )
    summary = X.T @ (Y - Y.mean()) / 4
    scalar = fit(
        summary,
        np.full(2, np.sqrt(0.7 / 4)),
        prior=normal_mixture(np.sqrt(0.7) * np.array([0.4, 1.2]), weights=[0.3, 0.4, 0.3]),
        fix_prior=True,
    )
    assert_allclose(r.coefficients, scalar.posterior_mean, atol=1e-11)
    assert_allclose(r.posterior_sd, scalar.posterior_sd, atol=1e-11)
    assert_allclose(r.lfdr, scalar.lfdr, atol=1e-11)
    assert_allclose(r.lfsr, scalar.lfsr, atol=1e-11)


def test_null_only_model_has_exact_intercept_noise_and_atoms():
    r = fit_regression(X, Y, scales=[0], tol=1e-12)
    assert_allclose(r.coefficients, 0)
    assert_allclose(r.posterior_sd, 0)
    assert_allclose(r.lfdr, 1)
    assert_allclose(r.lfsr, 1)
    assert_allclose(r.interval(), 0)
    assert_allclose(r.residual_variance, np.mean((Y - Y.mean()) ** 2))
    assert_allclose(r.quantile([0, 0.2, 1]), 0)
    assert_allclose(r.sample(5, seed=2), 0)


def test_more_predictors_than_rows_matches_dual_ridge_mean():
    # Fixed deterministic matrix; no simulated study or random performance claim.
    design = np.sin(np.arange(8 * 12).reshape(8, 12) * 0.37)
    response = np.array([1.0, -0.2, 0.7, -0.8, 0.2, 1.4, -0.4, 0.5])
    r = fixed_fit(design, response, fit_intercept=False, standardize=False)
    expected = design.T @ np.linalg.solve(design @ design.T + np.eye(8) / 0.8**2, response)
    assert_allclose(r.coefficients, expected, atol=3e-6)
    assert_allclose(r.predict(design), design @ expected, atol=3e-6)


def test_fitted_hyperparameters_increase_elbo():
    r = fit_regression(X, Y, scales=[0.3, 1.0], maxiter=2000, tol=1e-10)
    history = np.asarray(r.diagnostics["objective_history"])
    assert (np.diff(history) >= -1e-8).all()
    assert r.diagnostics["converged"]
    assert_allclose(r.weights.sum(), 1)
    assert r.residual_variance > 0


def test_regression_save_load_preserves_factors_without_refitting(tmp_path, monkeypatch):
    r = fixed_fit(labels=["first", "second"])
    path = tmp_path / "regression.json"
    save_result(r, path)
    monkeypatch.setattr("ashrink.regression.fit_regression", lambda *a, **kw: pytest.fail("refit"))
    restored = load_result(path)
    assert_allclose(restored.coefficients, r.coefficients)
    assert_allclose(restored.interval(), r.interval())
    assert_allclose(restored.predict(X), r.predict(X))
    assert [v["label"] for v in to_records(restored)] == ["first", "second"]
    assert "Approximate" in restored.format_summary()


def test_table_adapter_uses_named_predictors():
    r = fit_regression_frame(
        {"a": X[:, 0], "b": X[:, 1], "y": Y},
        "y",
        ["a", "b"],
        scales=[0.8],
        point_mass=False,
        initial_residual_variance=0.7,
        estimate_weights=False,
        estimate_residual_variance=False,
        tol=1e-12,
    )
    assert r.labels == ("a", "b")
    assert_allclose(r.coefficients, fixed_fit().coefficients)


@pytest.mark.parametrize(
    "options",
    [
        {"X": np.ones((4, 2))},
        {"X": [[1, 2], [np.nan, 3]], "y": [1, 2]},
        {"y": [1, 2]},
        {"initial_residual_variance": 0},
        {"scales": [-1]},
        {"maxiter": 0},
        {"tol": False},
        {"labels": ["a", "a"]},
        {"initial_weights": [1, 2, 3]},
        {"initial_coefficients": [1]},
    ],
)
def test_invalid_regression_inputs_reject(options):
    args = {"X": X, "y": Y, "scales": [0.8]}
    args.update(options)
    with pytest.raises(ValueError):
        fit_regression(**args)


def test_nonconvergence_is_reported():
    with pytest.warns(RuntimeWarning, match="maxiter"):
        r = fit_regression(X, Y, maxiter=1)
    assert not r.diagnostics["converged"]


def test_regression_plot_interfaces():
    import matplotlib.pyplot as plt

    r = fixed_fit()
    for function in [plot_coefficients, plot_residuals, plot_diagnostics]:
        ax = function(r)
        assert ax.get_xlabel()
        plt.close(ax.figure)
    with pytest.raises(ValueError):
        r.predict([[1]])


@pytest.mark.parametrize("corruption", ["weight_sum", "weight_support", "responsibility_sum"])
def test_malformed_saved_probabilities_reject(corruption):
    from ashrink import RegressionResult

    record = fixed_fit().to_dict()
    if corruption == "weight_sum":
        record["weights"] = [1.000001]
    elif corruption == "weight_support":
        record["weights"] = [0.0, 1.0]
        record["scales"] = [0.8, 1.2]
        record["responsibilities"] = [[1.0, 0.0], [1.0, 0.0]]
        record["component_means"] = [[0.1, 0.2], [0.1, 0.2]]
        record["component_variances"] = [[0.1, 0.2], [0.1, 0.2]]
    else:
        record["responsibilities"][0] = [0.9]
    with pytest.raises(ValueError):
        RegressionResult.from_dict(record)


def test_one_sweep_variance_and_elbo_against_normal_expectations():
    design = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    response = np.array([-1.0, -0.4, 1.2, 2.0])
    sigma2, tau = 0.7, 0.8
    centered = response - response.mean()
    d = float(design[:, 0] @ design[:, 0])
    # Exact posterior for one coefficient before the variance M-step.
    v = sigma2 / (d + 1 / tau**2)
    m = v * float(design[:, 0] @ centered) / sigma2
    expected_rss = float(np.sum((centered - design[:, 0] * m) ** 2)) + d * v
    expected_sigma = (expected_rss + (m * m + v) / tau**2) / (len(response) + 1)
    with pytest.warns(RuntimeWarning):
        r = fit_regression(
            design,
            response,
            scales=[tau],
            point_mass=False,
            standardize=False,
            estimate_weights=False,
            initial_residual_variance=sigma2,
            maxiter=1,
        )
    assert_allclose(r.coefficients, [m], atol=1e-12)
    assert_allclose(r.posterior_sd**2, [v], atol=1e-12)
    assert_allclose(r.residual_variance, expected_sigma, atol=1e-12)
    expected_log_likelihood = -len(response) / 2 * np.log(
        2 * np.pi * expected_sigma
    ) - expected_rss / (2 * expected_sigma)
    expected_log_prior = -0.5 * np.log(2 * np.pi * expected_sigma * tau**2) - (m * m + v) / (
        2 * expected_sigma * tau**2
    )
    expected_elbo = expected_log_likelihood + expected_log_prior + norm(scale=np.sqrt(v)).entropy()
    assert_allclose(r.diagnostics["elbo"], expected_elbo, atol=1e-12)


@pytest.mark.parametrize("function_name", ["plot_shrinkage", "plot_prior"])
def test_regression_rejects_summary_only_plots(function_name):
    import ashrink

    with pytest.raises(ValueError, match="regression|summary fits"):
        getattr(ashrink, function_name)(fixed_fit())

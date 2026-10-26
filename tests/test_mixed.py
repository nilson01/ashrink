"""Independent Gaussian block identities for random-intercept regression."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from ashrink import fit_mixed_regression, save_result, load_result, plot_random_effects

X = np.array([[-1.0], [0.0], [1.0], [-0.8], [0.2], [1.2]])
Y = np.array([-1.0, 0.2, 1.0, 0.0, 1.2, 2.0])
GROUPS = ["a"] * 3 + ["b"] * 3


def fixed_fit(**options):
    return fit_mixed_regression(
        X,
        Y,
        GROUPS,
        scales=[0.8],
        point_mass=False,
        standardize=False,
        initial_residual_variance=0.7,
        initial_group_variance=0.5,
        estimate_weights=False,
        estimate_residual_variance=False,
        estimate_group_variance=False,
        tol=1e-12,
        **options,
    )


def test_fixed_gaussian_mixed_mean_against_joint_block_solution():
    r = fixed_fit()
    Z = np.column_stack([np.array(GROUPS) == "a", np.array(GROUPS) == "b"]).astype(float)
    design = np.column_stack([np.ones(6), X, Z])
    penalty = np.diag([0, 1 / 0.8**2, 0.7 / 0.5, 0.7 / 0.5])
    expected = np.linalg.solve(design.T @ design + penalty, design.T @ Y)
    assert_allclose(r.intercept, expected[0], atol=4e-6)
    assert_allclose(r.coefficients, expected[1:2], atol=4e-6)
    assert_allclose(r.random_effects, expected[2:], atol=4e-6)
    # Factor variances condition on other means, rather than marginalizing the
    # joint coefficient covariance.
    centered = X - X.mean(axis=0)
    assert_allclose(r.posterior_sd**2, 0.7 / (np.sum(centered**2) + 1 / 0.8**2), atol=1e-12)
    assert_allclose(r.random_effect_sd**2, 1 / (3 / 0.7 + 1 / 0.5), atol=1e-12)
    assert_allclose(r.predict(X, GROUPS), r.fitted_values, atol=1e-12)
    assert_allclose(r.predict(X), r.intercept + X @ r.coefficients, atol=1e-12)


def test_population_known_and_unseen_group_predictions():
    r = fixed_fit()
    new = np.array([[0.4], [0.8]])
    population = r.predict(new)
    assert_allclose(r.predict(new, ["a", "b"]), population + r.random_effects)
    with pytest.raises(ValueError, match="unseen"):
        r.predict(new, ["a", "new"])
    assert_allclose(
        r.predict(new, ["a", "new"], allow_new_groups=True), population + [r.random_effects[0], 0]
    )


def test_group_variance_and_objective_updates_are_finite():
    r = fit_mixed_regression(X, Y, GROUPS, scales=[0.5, 1.0], maxiter=2000, tol=1e-10)
    assert r.diagnostics["converged"]
    assert np.diff(r.diagnostics["objective_history"]).min() >= -1e-8
    assert r.group_variance > 0 and r.residual_variance > 0
    assert_allclose(
        r.group_variance, np.mean(r.random_effects**2 + r.random_effect_sd**2), atol=1e-12
    )


def test_mixed_persistence_and_group_intervals(tmp_path):
    r = fixed_fit()
    path = tmp_path / "mixed.json"
    save_result(r, path)
    other = load_result(path)
    assert other.model == "gaussian_mixed_regression"
    assert_allclose(other.predict(X, GROUPS), r.predict(X, GROUPS))
    assert_allclose(other.random_effect_interval(), r.random_effect_interval())
    assert other.random_effect_summary()["group"] == ("a", "b")
    with pytest.raises(ValueError):
        document = r.to_dict()
        document["diagnostics"]["converged"] = not r.diagnostics["converged"]
        type(r).from_dict(document)


@pytest.mark.parametrize("groups", [[1, 1, 1, 2, 2, 2], ["a"] * 5, [""] * 6])
def test_invalid_groups_reject(groups):
    with pytest.raises(ValueError):
        fit_mixed_regression(X, Y, groups)


def test_random_effect_plot():
    import matplotlib.pyplot as plt

    ax = plot_random_effects(fixed_fit())
    assert ax.get_ylabel() == "Group"
    plt.close(ax.figure)

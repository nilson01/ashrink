"""Independent conjugate calculations and public-result contracts."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, optimize, stats

import ashrink
from ashrink._utils import fit_weights, tail_average


def test_normal_conjugacy_and_credible_intervals():
    y = np.array([-2.0, 0.0, 1.5])
    se = np.array([0.5, 1.0, 0.8])
    tau = 1.3
    prior = ashrink.normal_mixture([tau], point_mass=False)
    result = ashrink.fit(y, se, prior=prior, fix_prior=True)
    variance = 1 / (1 / se**2 + 1 / tau**2)
    mean = variance * y / se**2
    assert_allclose(result.posterior_mean, mean, rtol=1e-12, atol=1e-14)
    assert_allclose(result.posterior_sd, np.sqrt(variance), rtol=1e-12)
    expected = mean[:, None] + np.sqrt(variance)[:, None] * stats.norm.ppf([0.025, 0.975])
    assert_allclose(result.interval(), expected, atol=2e-9)
    assert_allclose(result.cdf(0), stats.norm.cdf(-mean / np.sqrt(variance)))
    assert_allclose(result.lfsr, stats.norm.cdf(-np.abs(mean) / np.sqrt(variance)))
    assert_array_equal(result.lfdr, np.zeros(3))


def test_zero_atom_cdf_jump_and_generalized_inverse():
    prior = ashrink.normal_mixture([1.0], weights=[0.7, 0.3])
    result = ashrink.fit([0.0], [1.0], prior=prior, fix_prior=True)
    atom = (
        0.7
        * stats.norm.pdf(0)
        / (0.7 * stats.norm.pdf(0) + 0.3 * stats.norm.pdf(0, scale=np.sqrt(2)))
    )
    left = (1 - atom) / 2
    assert_allclose(result.lfdr, [atom])
    assert_allclose(result.cdf(0), [left + atom])
    assert_allclose(result.cdf(-1e-10), [left], atol=1e-9)
    assert_allclose(result.quantile([left + atom / 4, left + 3 * atom / 4]), 0, atol=0)
    assert_allclose(result.probability_negative + result.probability_positive + result.lfdr, 1)
    assert_allclose(result.lfsr, [(1 + atom) / 2])


@pytest.mark.parametrize("asymmetric", [False, True])
def test_uniform_normal_posterior_against_truncated_normal(asymmetric):
    width, y, se = 1.4, 0.8, 0.6
    if asymmetric:
        prior = ashrink.uniform_mixture([width], asymmetric=True, point_mass=False, weights=[0, 1])
        lower, upper = 0, width
    else:
        prior = ashrink.uniform_mixture([width], point_mass=False)
        lower, upper = -width, width
    result = ashrink.fit([y], [se], prior=prior, fix_prior=True)
    reference = stats.truncnorm((lower - y) / se, (upper - y) / se, loc=y, scale=se)
    assert_allclose(result.posterior_mean, [reference.mean()], atol=1e-11)
    assert_allclose(result.posterior_sd, [reference.std()], atol=1e-11)
    assert_allclose(result.cdf(0.5), [reference.cdf(0.5)], atol=1e-11)
    assert_allclose(result.quantile([0.1, 0.8])[:, 0], reference.ppf([0.1, 0.8]), atol=2e-8)


@pytest.mark.parametrize("family", ["normal", "uniform", "halfuniform"])
def test_student_t_posterior_against_direct_effect_scale_integrals(family):
    y, se, df, width = 0.7, 0.8, 5.0, 1.2
    if family == "normal":
        prior = ashrink.normal_mixture([width], point_mass=False)
        lower, upper = -np.inf, np.inf

        def density(b):
            return stats.norm.pdf(b, scale=width)
    elif family == "uniform":
        prior = ashrink.uniform_mixture([width], point_mass=False)
        lower, upper = -width, width

        def density(b):
            return 1 / (2 * width)
    else:
        prior = ashrink.uniform_mixture([width], asymmetric=True, point_mass=False, weights=[0, 1])
        lower, upper = 0.0, width

        def density(b):
            return 1 / width

    def kernel(b):
        return stats.t.pdf((y - b) / se, df=df) / se * density(b)

    normalizer = integrate.quad(kernel, lower, upper, epsabs=1e-11, epsrel=1e-11)[0]
    mean = integrate.quad(lambda b: b * kernel(b), lower, upper, epsabs=1e-11)[0] / normalizer
    variance = (
        integrate.quad(lambda b: (b - mean) ** 2 * kernel(b), lower, upper, epsabs=1e-11)[0]
        / normalizer
    )
    cdf = integrate.quad(kernel, lower, min(0.4, upper), epsabs=1e-11)[0] / normalizer
    result = ashrink.fit([y], [se], df=df, prior=prior, fix_prior=True)
    assert_allclose(result.posterior_mean, [mean], atol=2e-7)
    assert_allclose(result.posterior_sd, [np.sqrt(variance)], atol=2e-7)
    assert_allclose(result.cdf(0.4), [cdf], atol=2e-7)


def test_poisson_single_gamma_conjugacy():
    counts = np.array([0, 3, 8])
    exposure = np.array([0.5, 2.0, 4.0])
    mode, strength = 1.5, 2.0
    result = ashrink.fit_poisson(
        counts, exposure, null_value=mode, strengths=[strength], point_mass=False
    )
    shape = 1 + mode * strength + counts
    rate = strength + exposure
    assert_allclose(result.posterior_mean, shape / rate)
    assert_allclose(result.posterior_sd, np.sqrt(shape) / rate)
    assert_allclose(result.cdf(1.2), stats.gamma.cdf(1.2, a=shape, scale=1 / rate))
    assert_allclose(
        result.interval(),
        stats.gamma.ppf([0.025, 0.975], a=shape[:, None], scale=1 / rate[:, None]),
        atol=1e-8,
    )


def test_binomial_single_beta_conjugacy_including_zero_trials():
    successes = np.array([0, 1, 5])
    trials = np.array([0, 4, 5])
    mode, strength = 0.3, 4.0
    result = ashrink.fit_binomial(
        successes, trials, null_value=mode, strengths=[strength], point_mass=False
    )
    a = 1 + mode * strength + successes
    b = 1 + (1 - mode) * strength + trials - successes
    assert_allclose(result.posterior_mean, a / (a + b))
    assert_allclose(result.posterior_sd, np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
    assert_allclose(result.cdf(0.4), stats.beta.cdf(0.4, a, b))
    assert_allclose(
        result.interval(), stats.beta.ppf([0.025, 0.975], a[:, None], b[:, None]), atol=1e-8
    )
    assert_allclose(result.posterior_mean[0], (1 + mode * strength) / (2 + strength))


def test_multivariate_conjugacy_with_correlated_observation_errors():
    y = np.array([[1.0, -0.5], [0.2, 1.3]])
    error = np.array([[0.7, 0.2], [0.2, 0.9]])
    prior = np.array([[1.2, 0.4], [0.4, 0.8]])
    result = ashrink.fit_multivariate(
        y, error_covariance=error, covariances=[prior], point_mass=False
    )
    covariance = np.linalg.inv(np.linalg.inv(prior) + np.linalg.inv(error))
    mean = (covariance @ np.linalg.solve(error, y.T)).T
    assert_allclose(result.posterior_mean, mean, atol=1e-12)
    assert_allclose(result.posterior_covariance, np.broadcast_to(covariance, (2, 2, 2)), atol=1e-12)
    assert_allclose(result.cdf(0), stats.norm.cdf(-mean / np.sqrt(np.diag(covariance))), atol=1e-12)


def test_singular_multivariate_pattern_has_marginal_atom_only():
    result = ashrink.fit_multivariate(
        [[1.0, 2.0]], [[0.5, 0.5]], covariances=[[[1, 0], [0, 0]]], point_mass=False
    )
    assert_allclose(result.posterior_mean, [[0.8, 0]])
    assert_allclose(result.posterior_covariance, [[[0.2, 0], [0, 0]]])
    assert_allclose(result.lfdr, [[0, 1]])
    assert_allclose(result.joint_null_probability, [0])
    assert_allclose(result.cdf(0)[0, 1], 1)
    assert_allclose(result.quantile(0.5)[0, 1], 0)


def test_identical_likelihoods_with_penalty_have_known_optimum():
    penalty = np.array([2.0, 3.0, 5.0])
    weights, probabilities, diagnostics = fit_weights(np.zeros((4, 3)), penalty=penalty, tol=1e-11)
    assert_allclose(weights, penalty / penalty.sum(), atol=1e-7)
    assert_allclose(probabilities, np.broadcast_to(weights, (4, 3)))
    assert diagnostics["converged"]


def test_disjoint_component_support_has_empirical_class_weights():
    loglik = np.array([[0, -np.inf], [0, -np.inf], [-np.inf, 0]])
    weights, probabilities, diagnostics = fit_weights(loglik)
    assert_allclose(weights, [2 / 3, 1 / 3], atol=1e-12)
    assert_array_equal(probabilities, [[1, 0], [1, 0], [0, 1]])
    assert diagnostics["converged"]


def test_two_component_solver_against_independent_bounded_scalar_search():
    likelihood = np.array([[0.8, 0.1], [0.5, 0.4], [0.15, 0.9], [0.25, 0.7]])
    penalty = np.array([0.6, 0.2])

    def objective(p):
        return -(
            np.log(p * likelihood[:, 0] + (1 - p) * likelihood[:, 1]).sum()
            + penalty[0] * np.log(p)
            + penalty[1] * np.log1p(-p)
        )

    reference = optimize.minimize_scalar(
        objective, bounds=(1e-9, 1 - 1e-9), method="bounded", options={"xatol": 1e-13}
    )
    weights, _, diagnostics = fit_weights(np.log(likelihood), penalty=penalty, tol=1e-10)
    assert_allclose(weights, [reference.x, 1 - reference.x], atol=2e-6)
    assert_allclose(-objective(weights[0]), diagnostics["objective_history"][-1], atol=1e-12)
    assert diagnostics["converged"]


def test_component_dominance_allows_boundary_solution():
    weights, _, diagnostics = fit_weights(np.log([[0.9, 0.1], [0.7, 0.2], [0.8, 0.3]]))
    assert_allclose(weights, [1, 0], atol=2e-7)
    assert diagnostics["converged"]


def test_row_offsets_in_log_likelihood_do_not_change_weights():
    likelihood = np.log([[0.8, 0.2], [0.1, 0.7], [0.3, 0.5]])
    w1, r1, d1 = fit_weights(likelihood)
    w2, r2, d2 = fit_weights(likelihood + np.array([1000, -1000, 20])[:, None])
    assert_allclose(w1, w2, atol=1e-8)
    assert_allclose(r1, r2, atol=1e-8)
    assert_allclose(d2["log_likelihood"] - d1["log_likelihood"], 20, atol=1e-9)


def test_tied_tail_averages_use_all_equal_cutoff_rows():
    assert_allclose(tail_average([0.2, 0.1, 0.2, 0.8]), [1 / 6, 0.1, 1 / 6, 0.325])


def test_missing_rows_preserve_original_indices_and_labels():
    prior = ashrink.normal_mixture([1], point_mass=False)
    result = ashrink.fit(
        [1.0, np.nan, -1.0],
        [0.5, 0.5, 1],
        prior=prior,
        fix_prior=True,
        missing="drop",
        labels=["a", "b", "c"],
    )
    assert_array_equal(result.row_indices, [0, 2])
    assert result.original_size == 3
    assert result.labels == ("a", "c")
    records = ashrink.to_records(result)
    assert [row["label"] for row in records] == ["a", "c"]
    assert_allclose([row["posterior_mean"] for row in records], result.posterior_mean)


@pytest.mark.parametrize("kind", ["scalar", "poisson", "binomial", "multivariate"])
def test_json_reconstruction_and_prior_reuse_do_not_fit_weights(kind, monkeypatch, tmp_path):
    if kind == "scalar":
        original = ashrink.fit([1.0, -0.2], [0.5, 0.8], scales=[1])
    elif kind == "poisson":
        original = ashrink.fit_poisson([1, 4], [1, 2], strengths=[1, 4])
    elif kind == "binomial":
        original = ashrink.fit_binomial([1, 4], [4, 5], strengths=[1, 4])
    else:
        original = ashrink.fit_multivariate(
            [[1.0, -0.2]], [[0.5, 0.8]], covariances=[np.eye(2)], point_mass=False
        )
    path = tmp_path / "result.json"
    ashrink.save_result(original, path)

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed prior must not optimize")

    for module in ["univariate", "counts", "multivariate"]:
        monkeypatch.setattr(f"ashrink.{module}.fit_weights", forbidden)
    restored = ashrink.load_result(path)
    assert_allclose(restored.posterior_mean, original.posterior_mean)
    assert_allclose(restored.posterior_sd, original.posterior_sd)
    assert_allclose(restored.interval(), original.interval())
    if kind == "scalar":
        reused = restored.shrink([0.3], [0.6])
        reference = ashrink.fit([0.3], [0.6], prior=original.prior, fix_prior=True)
        assert_allclose(reused.posterior_mean, reference.posterior_mean)
    elif kind in ("poisson", "binomial"):
        reused = restored.shrink([2], [3])
        assert_allclose(reused.mixture_weights, original.mixture_weights)
    else:
        reused = restored.shrink([[0.3, -0.4]], [[0.6, 0.7]])
        assert_allclose(reused.weights, original.weights)


@pytest.mark.parametrize(
    "effects,se,options",
    [
        ([np.nan], [1], {}),
        ([1], [0], {"missing": "drop"}),
        ([1, 2], [1], {}),
        ([1], [1], {"df": 0}),
        ([1], [1], {"labels": [5]}),
        ([np.nan], [1], {"missing": "drop"}),
    ],
)
def test_scalar_rejects_invalid_inputs(effects, se, options):
    with pytest.raises(ValueError):
        ashrink.fit(effects, se, **options)


@pytest.mark.parametrize(
    "call",
    [
        lambda: ashrink.fit_poisson([0.5]),
        lambda: ashrink.fit_poisson([1], exposure=0),
        lambda: ashrink.fit_poisson([1], null_value=0),
        lambda: ashrink.fit_binomial([3], [2]),
        lambda: ashrink.fit_binomial([1], [0]),
        lambda: ashrink.fit_binomial([0], [1], null_value=1),
    ],
)
def test_count_models_reject_invalid_support(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "error,prior",
    [([[1, 2], [2, 1]], np.eye(2)), ([[1, 0], [0, 0]], np.eye(2)), (np.eye(2), [[1, 2], [2, 1]])],
)
def test_multivariate_rejects_invalid_covariance(error, prior):
    with pytest.raises(ValueError):
        ashrink.fit_multivariate(
            [[0, 1]], error_covariance=error, covariances=[prior], point_mass=False
        )


@pytest.mark.parametrize("data", [{"schema_version": 9}, {"schema_version": 1, "model": "unknown"}])
def test_load_rejects_unknown_schema_or_model(tmp_path, data):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        ashrink.load_result(path)


def test_scalar_outputs_are_read_only():
    result = ashrink.fit(
        [1], [0.5], prior=ashrink.normal_mixture([1], point_mass=False), fix_prior=True
    )
    with pytest.raises(ValueError):
        result.posterior_mean[0] = 2
    with pytest.raises(TypeError):
        result.diagnostics["converged"] = True

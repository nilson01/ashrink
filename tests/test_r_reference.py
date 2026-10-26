"""Fixed cases exported through installed ashr 2.2-63 public functions."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ashrink import MixturePrior, fit

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "ashr-2.2-63.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_installed_ashr_posterior(case):
    g = case["prior"]
    if case["family"] == "normal":
        prior = MixturePrior("normal", np.asarray(g["sd"]), np.asarray(g["pi"]))
    else:
        a, b = np.asarray(g["a"]), np.asarray(g["b"])
        prior = MixturePrior(case["family"], np.maximum(-a, b), np.asarray(g["pi"]), a, b)
    result = fit(
        case["effects"],
        case["standard_errors"],
        df=case["df"],
        prior=prior,
        fix_prior=case["fixed"],
        null_weight=1,
        tol=1e-10,
    )
    # Fixed analytic cases are near machine precision. Estimated weights and
    # independent Student-t integration use a numerical tolerance.
    tolerance = 2e-6 if not case["fixed"] else (2e-7 if case["df"] else 2e-11)
    for name in ("posterior_mean", "posterior_sd", "lfdr", "lfsr"):
        assert_allclose(getattr(result, name), case[name], rtol=tolerance, atol=tolerance)
    assert_allclose(result.prior.weights, case["weights"], atol=tolerance, rtol=tolerance)


REGRESSION_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "mrash-0.1-43.json").read_text()
)


@pytest.mark.parametrize("case", REGRESSION_REFERENCE["cases"], ids=lambda c: c["name"])
def test_installed_regression_reference(case):
    from ashrink import fit_regression

    options = dict(
        scales=[0.4, 1.2],
        initial_weights=[0.3, 0.4, 0.3],
        initial_residual_variance=0.7,
        standardize=False,
        estimate_weights=case["estimate_weights"],
        estimate_residual_variance=case["estimate_residual_variance"],
        maxiter=case["maxiter"],
        tol=1e-12,
    )
    if case["name"] == "one-sweep":
        with pytest.warns(RuntimeWarning, match="maxiter"):
            result = fit_regression(case["X"], case["y"], **options)
    else:
        result = fit_regression(case["X"], case["y"], **options)
        assert result.diagnostics["converged"]
    tolerance = 2e-10 if case["name"] == "one-sweep" else 1e-5
    assert_allclose(result.coefficients, case["coefficients"], rtol=tolerance, atol=tolerance)
    assert_allclose(result.intercept, case["intercept"], rtol=tolerance, atol=tolerance)
    assert_allclose(
        result.residual_variance, case["residual_variance"], rtol=tolerance, atol=tolerance
    )
    assert_allclose(result.weights, case["weights"], rtol=tolerance, atol=tolerance)

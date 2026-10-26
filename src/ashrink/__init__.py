"""Empirical Bayes adaptive shrinkage for effects, rates and probabilities.

Posterior summaries condition on fitted finite-mixture weights. Their
calibration depends on the likelihood, prior family and sampling design.
"""

from .priors import MixturePrior, normal_mixture, uniform_mixture
from .univariate import ASHResult, fit
from .counts import CountResult, fit_poisson, fit_binomial
from .multivariate import MultivariateResult, fit_multivariate
from .regression import RegressionResult, fit_regression
from .mixed import MixedRegressionResult, fit_mixed_regression
from .glm import GLMResult, fit_logistic_regression, fit_poisson_regression
from .io import (
    fit_frame,
    fit_poisson_frame,
    fit_binomial_frame,
    fit_multivariate_frame,
    fit_regression_frame,
    fit_logistic_regression_frame,
    fit_poisson_regression_frame,
    fit_mixed_regression_frame,
    read_csv,
    write_csv,
    to_records,
    save_result,
    load_result,
)
from .plots import (
    plot_shrinkage,
    plot_intervals,
    plot_prior,
    plot_diagnostics,
    plot_coefficients,
    plot_residuals,
    plot_random_effects,
)

__version__ = "0.1.0"
__all__ = [
    "fit",
    "fit_poisson",
    "fit_binomial",
    "fit_multivariate",
    "fit_regression",
    "RegressionResult",
    "fit_mixed_regression",
    "MixedRegressionResult",
    "GLMResult",
    "fit_logistic_regression",
    "fit_poisson_regression",
    "ASHResult",
    "CountResult",
    "MultivariateResult",
    "MixturePrior",
    "normal_mixture",
    "uniform_mixture",
    "fit_frame",
    "fit_poisson_frame",
    "fit_binomial_frame",
    "fit_multivariate_frame",
    "fit_regression_frame",
    "fit_logistic_regression_frame",
    "fit_poisson_regression_frame",
    "fit_mixed_regression_frame",
    "read_csv",
    "write_csv",
    "to_records",
    "save_result",
    "load_result",
    "plot_shrinkage",
    "plot_intervals",
    "plot_prior",
    "plot_diagnostics",
    "plot_coefficients",
    "plot_residuals",
    "plot_random_effects",
]

# Numerical methods and limits

The scalar implementation estimates finite-mixture weights for the
adaptive-shrinkage models described by [Stephens (2017)](https://doi.org/10.1093/biostatistics/kxw041).
Gaussian component posteriors are analytic. Student-t component posteriors
use adaptive one-dimensional quadrature and record integration diagnostics.
This numerical route can be substantially slower than the Gaussian route.

Mixture fitting uses EM with fixed component grids, followed by constrained
optimization when the optimality residual remains large. Diagnostics retain the
objective history, iteration count, stopping rule and a KKT residual. A small
optimality residual is a numerical stopping criterion; it does not establish
statistical calibration or uniqueness of the fitted prior. Results retain
nonconvergence information rather than treating every returned fit as reliable.

The R reference version is ashr 2.2-63. Uniform widths and normal standard
deviations have different meanings. Student-t inputs are scale parameters.
The normal-prior/Student-t combination uses an independent integration route;
the inspected R implementation rejects that combination. Ranked summaries
include all tied local probabilities at a cutoff, which can differ from R's
order-dependent cumulative averages within ties.

Count models use conjugate Gamma–Poisson and Beta–binomial mixtures sharing
a specified mode, with an optional point mass at that mode. Multivariate
models use finite mixtures of zero-mean normals with supplied covariance
patterns, following the fixed-pattern part of
[Urbut et al. (2019)](https://doi.org/10.1038/s41588-018-0268-8).
Selected support does not imply full ashr or mashr coverage.

Posterior probabilities and credible intervals condition on estimated mixture
weights. They omit uncertainty from fitting those weights. Dependence between
scalar observations, misspecified likelihoods and unsuitable prior components
can affect inference. Independent reference comparisons and statistical
performance checks are recorded separately from implementation progress.


Gaussian regression uses cyclic variational coordinate ascent with
residual-variance-scaled normal-mixture priors, following
[Kim et al. (2024)](https://jmlr.org/papers/v25/22-0953.html).
The objective is an evidence lower bound, rather than the exact marginal
likelihood. Coefficient marginals factorize; their variances need not equal
those from the exact joint posterior. Mean predictions reuse fitted
preprocessing. See the [regression guide](regression.md) for inputs and limits.


Binary/binomial regression uses a quadratic logistic lower bound. Poisson
regression uses analytic exponential moments of independent mixture-normal
factors, with guarded inner Newton optimization. Gaussian random-intercept
regression uses independent coefficient and group factors. These extensions
have focused analytic/stationarity checks, not repeated-dataset calibration
studies. Prediction applies the fitted mean coefficients; nonlinear predictions
do not integrate their uncertainty. See the [regression guide](regression.md).

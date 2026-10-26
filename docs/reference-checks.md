# Numerical reference checks

The test fixtures contain six scalar cases from installed R `ashr` 2.2-63
and five Gaussian regression cases from installed `mr.ash.alpha` 0.1-43
(commit `8e257fdb73589f280192b1c99a176d82196be298`).
Three multivariate cases also use installed `mashr` 0.2.79. All exports used
R 4.5.1. Archive and exporter hashes are stored with each fixture.

| Comparison | Quantities | Absolute and relative tolerance |
| --- | --- | --- |
| Gaussian scalar, fixed prior | Mean, SD, null/sign probabilities, weights | `2e-11` |
| Student-t/uniform scalar, fixed prior | Same quantities | `2e-7` |
| Gaussian scalar, estimated weights | Same quantities | `2e-6` |
| Gaussian regression, converged fit | Coefficients, intercept, residual variance, weights | `1e-5` |
| Gaussian regression, one sweep | Same quantities | `2e-10` |
| Multivariate, fixed prior or one component | Means, SDs, covariance, null/sign probabilities, weights | `2e-10` |
| Multivariate, estimated mixture | Same quantities | `5e-6` |

Regression comparisons use explicit grids, zero initial coefficients, common
initial weights/variance, an intercept and no standardization. R uses
`method_q="sigma_indep_q"`, matching the fixed-factor variance update.
Its `sigma_dep_q` option uses a different update and need not agree after one
sweep. Stopping rules also differ: AShrink checks changes in the evidence
lower bound (ELBO), whereas R checks coefficient or mixture-weight changes.
Matching results in these cases does not establish global optimization, interval coverage or prediction performance.

## Recreate the fixtures

R is needed only to regenerate fixtures. Install the pinned reference packages
and `jsonlite` in an isolated R library, selected through `R_LIBS_USER`.
Use the `ashr` 2.2-63 source archive and the pinned `mr.ash.alpha` Git commit;
install `mashr` 0.2.79 for its exporter. Installing unpinned current versions
does not reproduce these references.

```sh
Rscript tools/export_ashr_reference.R ashr-reference.json
Rscript tools/export_regression_reference.R regression-reference.json
Rscript tools/export_mashr_reference.R multivariate-reference.json
python -m pytest tests/test_r_reference.py
```

The exporters write runtime/library provenance along with the fixed cases.
Repository fixtures omit machine-specific library paths. The tests read the
committed fixtures and do not invoke R. To check a regenerated export, compare
its case values with the matching fixture; local namespace paths may differ.

The pinned regression package requests C++11. With current RcppArmadillo headers,
our installation required a C++17 compiler setting. The local R configuration
also referenced absent Fortran libraries; selecting installed Fortran libraries
resolved linking. These were build settings; the reference source was unchanged.
Compiler and BLAS differences can affect final rounding.

R warns that half-uniform priors without a null-biased penalty can give
misleading sign probabilities. Those fixed-prior cases check arithmetic only.
The regression reference also warns when substantial weight reaches the largest
prior component. The comparison keeps the explicit grid unchanged; an applied
analysis should inspect its adequacy.

Conjugate count models and fixed-pattern multivariate models have independent
analytic tests. Their inclusion does not imply full `ashr` or `mashr` coverage.
Logistic and Poisson-regression tests check null-only likelihoods and the
stationarity equations for a single normal factor. Random-intercept tests
compare fixed-variance posterior means with a joint Gaussian block solution.
These checks address the implemented updates; they do not establish interval
coverage or prediction performance across repeated datasets.


Multivariate comparisons use explicit covariance patterns, correlation matrices,
unit scale multipliers and no automatic covariance normalization. One case
checks a single full-rank component, one estimates mixture weights, and one
reconstructs fixed-weight summaries with singular patterns and a zero atom.
The fixed-weight case uses `mash(..., fixg=TRUE)` in R and public result
reconstruction in Python. It checks posterior calculations rather than fitting
new mixture weights. R's one-component optimizer reports that no optimization
is needed. Learned covariance patterns remain outside the selected support.

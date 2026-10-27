# AShrink

Adaptive shrinkage uses information shared across estimates to learn a prior
distribution and compute posterior summaries. AShrink provides this approach in
Python for supplied effect estimates and selected regression models. It also
supports Poisson rates, binomial probabilities, and effects measured across
conditions. NumPy and SciPy provide the numerical calculations;
R is not a runtime dependency.

Start with [installation](installation.md), then follow the
[worked tutorial](tutorials.md). For outcomes and predictors, use the
[regression guide](regression.md).

## Choose an analysis

| Available input | Fitting function |
| --- | --- |
| Effect estimates and standard errors | `fit` |
| Counts and exposure times | `fit_poisson` |
| Successes and trial counts | `fit_binomial` |
| Effect vectors and error covariances | `fit_multivariate` |
| Continuous outcomes and predictors | `fit_regression` |
| Binary/binomial outcomes and predictors | `fit_logistic_regression` |
| Counts and predictors | `fit_poisson_regression` |
| Grouped continuous outcomes and predictors | `fit_mixed_regression` |

The [API guide](api.md) explains fitting, posterior summaries, tables, saved
results and plots. The [reference manual](manual/reference.md)
([PDF](manual/ashrink-reference.pdf)) gives the detailed arguments and methods.

Posterior intervals condition on fitted mixture weights. Regression also uses a
variational approximation. Read the [numerical limits](numerical-methods.md)
and [reference checks](reference-checks.md) before interpreting uncertainty.
Agreement with selected reference cases does not establish interval coverage or
full coverage of the R packages.

## Methods and attribution

Scalar adaptive shrinkage follows
[Stephens (2017)](https://doi.org/10.1093/biostatistics/kxw041).
Gaussian regression follows
[Kim et al. (2024)](https://jmlr.org/papers/v25/22-0953.html).
Multivariate fitting uses supplied covariance patterns, following the
fixed-pattern part of
[Urbut et al. (2019)](https://doi.org/10.1038/s41588-018-0268-8).
Cite the method used and record the AShrink version or commit.

AShrink is an independent implementation distributed under GPL-3.0-or-later.
See the repository [notice](https://github.com/nilson01/ashrink/blob/main/NOTICE.md)
and [license](https://github.com/nilson01/ashrink/blob/main/LICENSE) for attribution
and terms.

## Citation

See the [package paper and BibTeX citation](citation.md).

## Acknowledgements

Computations were performed on the Arseven Computing Cluster, Department of
Statistics, Texas A&M University.

This work received no grant funding.

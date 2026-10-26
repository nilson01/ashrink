# API guide

## Fit a model

| Input | Function | Parameter being estimated |
| --- | --- | --- |
| Effect estimates and standard errors | `fit` | Scalar effects under Gaussian or Student-t summary likelihoods |
| Counts and exposure times | `fit_poisson` | Nonnegative Poisson rates |
| Successes and trial counts | `fit_binomial` | Binomial probabilities |
| Effect matrices and error covariances | `fit_multivariate` | Effect vectors across conditions |

In `fit`, `family` selects the prior: `normal`, `uniform` or `halfuniform`.
A finite `df` selects a Student-t likelihood; its input standard-error value
is the distribution's **scale**, not its standard deviation. Gaussian inputs
use the usual standard error. `null_weight=1` fits mixture weights by maximum
likelihood; larger values add a penalty favouring the point-null component.

`normal_mixture` and `uniform_mixture` construct explicit scalar priors.
Pass one through `fit(..., prior=prior, fix_prior=True)` to evaluate a fixed
prior without estimating new weights. A result's `shrink` method evaluates
new observations with the fitted prior. It does not estimate new weights.

When observations are counts, shrinkage must respect the parameter range.
Poisson rates are nonnegative and binomial probabilities lie between zero and
one. The Poisson and binomial methods therefore use Gamma and Beta mixtures with a
specified common mode. Their parameter and null scales differ from scalar
signed effects. These are conjugate-mixture extensions, not matched R ashr
likelihood routes. When effects are measured across conditions, their relationships can also
inform shrinkage. Multivariate fitting uses supplied positive-semidefinite
covariance patterns and positive-definite observation-error covariances;
patterns and residual correlations are not learned from the effects.

## Inspect posterior results

All results provide `posterior_mean`, `posterior_sd`, `summary`, `cdf`,
`quantile`, `interval` and `sample`. `summary` returns numeric columns;
scalar and multivariate `format_summary` methods provide compact text.
`sample` draws latent parameters, not future observed data.

`interval(level=.95)` returns equal-tailed posterior credible intervals.
They condition on the fitted mixture weights and selected component grid.
`lfdr` is posterior point-null probability. `lfsr` is the smaller posterior
probability of a nonpositive or nonnegative effect, including a zero atom in
both error events. For count models, the two directions are relative to the
specified null rate or probability. `qvalue` and `svalue` average local null
and sign-error probabilities at their respective cutoffs. They are posterior
summaries, not automatic frequentist error-control guarantees. Without a
point-null component, `lfdr` and `qvalue` are zero by construction; this is a
prior choice, not evidence that every effect differs from the null.

Multivariate results also provide posterior covariance matrices and the
probability that the whole effect vector is zero. A coordinate can have a
zero atom even when another coordinate does not. Marginal and whole-vector
null probabilities therefore have different meanings.

## Tables and saved results

Named-column adapters are `fit_frame`, `fit_poisson_frame`,
`fit_binomial_frame` and `fit_multivariate_frame`. They accept pandas frames
or mappings of columns. Scalar missing values are rejected by default;
`missing="drop"` explicitly removes incomplete rows and retains their indices.
Other input routes require complete data.

`to_records` returns plain rows; `write_csv` exports summaries.
`save_result` and `load_result` use versioned JSON without pickle or code
execution. Loading recomputes posterior quantities but does not fit mixture
weights. `result.to_dataframe()` requires the `table` extra.

## Plots

`plot_shrinkage`, `plot_intervals`, `plot_prior` and `plot_diagnostics` return
Matplotlib axes. They do not call `show` or save files. For multivariate
results, `condition` selects the column to display. Prior density plots keep
point-null probability separate from continuous density.

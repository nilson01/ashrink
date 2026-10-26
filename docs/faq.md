# Common questions

## Does AShrink require R?

No. Fitting and prediction use NumPy and SciPy. R is needed only to regenerate
the comparison fixtures described in [reference checks](reference-checks.md).

## Which fitting function should I use?

Use a summary-input function for effect estimates, counts or successes.
Use a regression function when outcomes and predictors are available directly.
The [API table](api.md#fit-a-model) lists the supported methods and parameter scales.

## Are credible intervals confidence intervals?

The reported intervals are posterior credible intervals conditional on fitted
mixture weights and the selected component grid. Regression also uses an
independent-factor variational approximation. These calculations do not by
themselves establish repeated-sample coverage. See
[numerical methods and limits](numerical-methods.md).

## What do null and sign probabilities mean?

`lfdr` reports posterior point-null probability. `lfsr` reports the smaller
posterior probability of a nonpositive or nonnegative effect, including any
zero atom in both events. Count models compare the rate or probability with
the specified null value. The [API guide](api.md#inspect-posterior-results)
explains the related ranked summaries and prior-dependent limits.

## Does it implement all of ashr, mashr or mr.ash?

AShrink supports the methods listed in the [API guide](api.md).
The [reference checks](reference-checks.md) compare selected matched cases.
Multivariate fitting uses supplied covariance patterns; it does not learn those
patterns or residual correlations from the effects.

## What should I check when a fit does not converge?

Inspect the diagnostics and objective history before using the result.
For scalar models, check the component grid and numerical integration warnings.
For regression, inspect the variational objective and initialization.
A small numerical residual or a converged objective does not establish
statistical calibration. The [numerical guide](numerical-methods.md) explains
these distinctions.

## Are saved results anonymous?

No. Summary-input results retain the input estimates or counts. Regression
JSON retains training outcomes directly or through fitted values and residuals.
See [tables and saved results](api.md#tables-and-saved-results).

## Where are the detailed arguments and formulas?

Use the [online reference manual](manual/reference.md) or download the
[reference PDF](manual/ashrink-reference.pdf). The [development guide](development.md)
explains how to preview this searchable site.

# AShrink

AShrink provides empirical Bayes shrinkage for noisy effect estimates, count rates,
binomial probabilities, effect estimates measured across several conditions,
and regression from outcomes and predictors, including Gaussian random intercepts.
It learns finite-mixture weights from the data and returns posterior summaries.

The adaptive-shrinkage method follows [Stephens (2017)](https://doi.org/10.1093/biostatistics/kxw041).
The R packages [ashr](https://github.com/stephens999/ashr) and
[mashr](https://github.com/stephenslab/mashr) are numerical references for matched
methods. Count models use conjugate mixtures with a specified common mode.
AShrink is an independent implementation; the original authors have not endorsed it.

## Install from a checkout

```sh
python -m pip install .
```

NumPy and SciPy are runtime dependencies. Plotting and pandas tables use the
`plot` and `table` extras. No R runtime is required.

## First example

```python
from ashrink import fit

result = fit(
    [-1.4, -0.6, -0.1, 0.0, 0.2, 0.7, 1.8, 2.2],
    [0.7, 0.4, 0.5, 0.3, 0.5, 0.4, 0.8, 0.7],
    scales=[0.2, 0.6, 1.5],
)
print(result.format_summary())
intervals = result.interval(level=0.95)
```

The inputs are effect estimates and their standard errors. `scales` gives
fixed normal-prior component standard deviations; the fit estimates their
mixture weights. Returned intervals are posterior credible intervals,
conditional on those fitted weights. The first three posterior means are
`[-0.764278, -0.306851, -0.032956]`.

![Eight Gaussian effect estimates before and after shrinkage, with their posterior credible intervals.](docs/figures/first-example.png)

(a) Input estimates and posterior means. (b) Equal-tailed 95% credible intervals,
conditional on the fitted prior. These are fixed illustrative inputs; the
[plotted values](docs/figures/first-example.csv) are available as CSV.

## Methods

| Analysis | Function | Prior components |
| --- | --- | --- |
| Gaussian or Student-t effect estimates | `fit` | Normal, symmetric uniform or asymmetric uniform |
| Counts with exposure times | `fit_poisson` | Gamma components sharing a specified mode |
| Successes out of trials | `fit_binomial` | Beta components sharing a specified mode |
| Effects across conditions | `fit_multivariate` | Multivariate normals with supplied covariance patterns |
| Continuous outcomes and predictors | `fit_regression` | Normal-mixture coefficient priors; variational inference |
| Binary outcomes or successes/trials and predictors | `fit_logistic_regression` | Normal-mixture coefficient priors; logistic bound |
| Counts and predictors | `fit_poisson_regression` | Normal-mixture coefficient priors; Poisson variational inference |
| Grouped continuous outcomes | `fit_mixed_regression` | Shrunk fixed effects and Gaussian random intercepts |

Each method provides posterior estimates, credible intervals, null/sign
probabilities, tables and saved results. Plots use the `plot` extra.
See the [reference manual](docs/manual/reference.md) ([PDF](docs/manual/ashrink-reference.pdf)),
[API guide](docs/api.md), [worked examples](examples/workflows.py),
[regression](docs/regression.md)
and [numerical limits](docs/numerical-methods.md).
[Reference checks](docs/reference-checks.md) record the installed-R cases and
independent analytic checks. Installed-wheel tests pass on macOS with Python 3.10
and 3.14. Windows verification is pending. The package is not yet released.

## Citation

Cite the original statistical method used in an analysis and record the AShrink
version or commit. See [NOTICE.md](NOTICE.md) for attribution and licenses.

# AShrink

AShrink provides empirical Bayes shrinkage for noisy effect estimates, count rates,
binomial probabilities and effect estimates measured across several conditions.
It learns finite-mixture weights from the data and returns posterior summaries.

The adaptive-shrinkage method follows [Stephens (2017)](https://doi.org/10.1093/biostatistics/kxw041).
The R packages [ashr](https://github.com/stephens999/ashr) and
[mashr](https://github.com/stephenslab/mashr) are numerical references for matched
methods. Count conjugate-mixture methods have their own contracts. AShrink is an
independent implementation; the original authors have not endorsed it.

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
conditional on those fitted weights.

## Methods

| Analysis | Function | Prior components |
| --- | --- | --- |
| Gaussian or Student-t effect estimates | `fit` | Normal, symmetric uniform or asymmetric uniform |
| Counts with exposure times | `fit_poisson` | Gamma components sharing a specified mode |
| Successes out of trials | `fit_binomial` | Beta components sharing a specified mode |
| Effects across conditions | `fit_multivariate` | Multivariate normals with supplied covariance patterns |

Each route provides posterior estimates, credible intervals, null/sign
probabilities, tables and saved results. Plots use the `plot` extra.
See the [API guide](docs/api.md), [worked examples](examples/workflows.py)
and [numerical limits](docs/numerical-methods.md). Numerical validation and
release checks are pending.

## Citation

Cite the original statistical method used in an analysis and record the AShrink
version or commit. See [NOTICE.md](NOTICE.md) for attribution and licenses.

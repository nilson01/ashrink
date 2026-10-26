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

## Methods

The selected first release covers Gaussian and Student-t summary likelihoods,
Poisson rates, binomial probabilities and multivariate Gaussian effects with
supplied covariance patterns. Numerical validation and release checks are pending.

## Citation

Cite the original statistical method used in an analysis and record the AShrink
version or commit. See [NOTICE.md](NOTICE.md) for attribution and licenses.

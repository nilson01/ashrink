# Regression from outcomes and predictors

Use `fit_regression` when the observations and predictors are available directly.
Conditional on the fitted intercept and predictor transformation, coefficient
fitting uses the Gaussian likelihood and prior

$$
\mathbf y\mid\mathbf X,\mathbf b,\sigma^2
\sim N(\mathbf X\mathbf b,\sigma^2 I_n),\qquad
b_j\mid g,\sigma^2
\sim\sum_{k=1}^{K}\pi_k N(0,\sigma^2\sigma_k^2).
$$

Here, $n$ is the number of observations, $p$ the number of predictors,
and $I_n$ the identity matrix. The $K$ mixture weights satisfy
$\pi_k\ge0$ and $\sum_k\pi_k=1$. The vector $\mathbf y$ contains
centered outcomes and $\mathbf X$ contains
centered, optionally scaled predictor columns. The vector $\mathbf b$ contains
coefficients on that predictor scale, and $\sigma^2$ is the residual variance.
With `fit_intercept=False`, outcomes and predictors are not centered.
The regression prior $g$ has mixture weights $\pi_k$, which describe how often each prior scale occurs.
A component with $\sigma_k=0$ is a point mass at zero. The method follows
[Kim et al. (2024), Sections 3.1–3.5 and Appendix E](https://jmlr.org/papers/v25/22-0953.html).

## Fit and predict

```python
import numpy as np
from ashrink import fit_regression

X = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
y = np.array([-1.8, -0.1, 0.6, 3.3])
result = fit_regression(X, y, scales=[0.3, 1.0], labels=["x1", "x2"])
print(result.format_summary())
predicted = result.predict(np.array([[0.2, 0.5], [0.8, -0.3]]))
```

`scales` supplies the relative prior standard deviations $\sigma_k$.
Fitting adds a zero atom by default and learns the mixture weights and residual
variance. The component grid is sorted and deduplicated; `initial_weights`
must follow the resulting order, with the zero atom first.

The default fit includes an intercept and divides centered predictor columns
by their root mean square. Returned coefficients and their intervals use the
original predictor units. `predict` accepts numeric columns in the fitted order
and reuses the fitted transformation. It returns outcome means, without
prediction intervals. More predictors than observations are allowed; missing
values and predictors with no information are rejected.

For named columns, use `fit_regression_frame(data, outcome, predictors)`.
For an aligned numerical comparison, disable standardization or reproduce its
transformation explicitly. Fixed mixture weights and residual variance are
available through `estimate_weights=False` and
`estimate_residual_variance=False`, with their respective `initial_*` values.

## Coefficients and diagnostics

`coefficients`, `intercept`, `residual_variance` and `weights` report fitted
quantities. `interval`, `cdf`, `quantile` and `sample` describe approximate
marginal coefficient distributions. `summary` returns numeric columns;
`plot_coefficients`, `plot_residuals` and `plot_diagnostics` inspect the fit.
Saved JSON includes fitted values and residuals, from which training outcomes
can be recovered. It is not an anonymized summary.

The algorithm uses independent coefficient factors and maximizes an evidence
lower bound (ELBO). Inspect `diagnostics["converged"]` and the recorded ELBO
history. Convergence is local and can depend on initialization. The default
starts coefficients at zero; `initial_coefficients` supplies another start.

Intervals condition on fitted mixture weights, residual variance and intercept.
They do not include uncertainty from estimating those quantities. With strongly
correlated predictors, the approximation can favor one predictor over another
and understate uncertainty. Null/sign probabilities do not provide guaranteed
variable selection or establish causal effects. The Gaussian method requires a
dense numeric predictor matrix. Binary, count and grouped outcomes use the
separate methods below.

## Binary and count outcomes

| Outcome | Function | Prediction |
| --- | --- | --- |
| Binary outcomes or successes/trials | `fit_logistic_regression(X, y, trials=1)` | Probability; supply new `trials` for expected successes |
| Counts with exposure times | `fit_poisson_regression(X, y, exposure=1)` | Expected counts at new `exposure` |

Both functions accept a numeric `offset` on the link scale. New predictions
use exposure one and offset zero unless specified. Training row values are
never reused for new rows. `predict(..., scale="link")` returns the linear
predictor. Predictions apply the inverse link to mean coefficients; they do
not integrate coefficient uncertainty. Poisson exponential underflow and
overflow raise an error; link predictions can still represent extreme values.

Unlike Gaussian regression, these priors use absolute coefficient standard
deviations on the transformed predictor scale. No residual variance is fitted.
Logistic coefficients describe log odds; Poisson coefficients describe log rates.
Their intervals remain approximate marginal credible intervals.

Logistic fitting uses the quadratic lower bound of
[Jaakkola and Jordan (1997)](https://proceedings.mlr.press/r1/jaakkola97a.html).
Poisson fitting evaluates its expected log likelihood using the exponential
moments of the mixture-normal factors. It then optimizes component means and
variances with safeguarded Newton steps. Both methods estimate mixture weights
and report their variational objective. These are extensions beyond the scalar
ASH likelihoods, rather than matched R `ashr` methods.

## Grouped continuous outcomes

`fit_mixed_regression(X, y, groups)` adds independent Gaussian random intercepts
to Gaussian regression. Supply one nonempty string group label per observation.
The fit estimates a common group variance as well as the residual variance.
The `estimate_group_variance` and `estimate_residual_variance` flags can hold
supplied initial values fixed.

`result.predict(X_new)` returns population means. Supplying known group labels
adds their fitted random intercepts. Unseen labels raise an error unless
`allow_new_groups=True`, which assigns those groups a mean random effect of zero.
Neither choice adds future observation noise or integrates uncertainty in the
group effects.

`random_effect_summary` and `random_effect_interval` report the approximate
group distributions; `plot_random_effects` displays them. Fixed and group
factors are independent in the variational approximation. Their marginal
variances need not equal those from an exact joint Gaussian analysis.
Random slopes, general correlation structures and non-Gaussian mixed models
are outside this version.

The [extension example](https://github.com/nilson01/ashrink/blob/main/examples/regression_extensions.py) shows all three
methods. Named-column adapters add `_frame` to each fitting function's name.
`fit_mixed_regression_frame` takes the group column through `group=`.

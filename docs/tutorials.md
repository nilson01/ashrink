# Worked tutorial

## Shrink noisy effect estimates

Start with an estimate and its standard error for each effect. This example uses
eight fixed illustrative inputs from the reference manual.

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

`scales` gives the fixed standard deviations of the normal-prior components.
The fit estimates their mixture weights. The first three posterior means are
`[-0.764278, -0.306851, -0.032956]`.

![Input effect estimates and posterior summaries for eight fixed illustrative inputs.](figures/first-example.png)

The left panel compares input estimates with posterior means. The right panel
shows equal-tailed 95% credible intervals, conditional on the fitted prior.
The [displayed values](figures/first-example.csv) are available as CSV.
These intervals omit uncertainty from estimating the mixture weights.

## Inspect and save the result

Use `result.summary()` for numeric reporting columns and
`result.format_summary()` for compact text. `result.posterior_sd` reports posterior
standard deviations. The [API guide](api.md) explains null and sign probabilities,
including their interpretation when the prior has a point mass at zero.

`save_result` and `load_result` write and restore versioned JSON results.
Saved files retain input data and should be handled with the same care as those
inputs. The [manual](manual/reference.md) gives the complete saving example.

## Analyze other inputs

The manual has worked examples for
[Student-t estimates, count data and effects across conditions](manual/reference.md#worked-examples).
Student-t inputs require scale parameters, which differ from their standard
deviations. Count models return rates or probabilities on their respective
parameter scales.

For continuous, binary, count or grouped outcomes with predictors, follow the
[regression guide](regression.md). Its predictions return outcome means;
they do not include future observation noise or prediction intervals.
The repository also contains the fixed
[summary-input examples](https://github.com/nilson01/ashrink/blob/main/examples/workflows.py)
and [regression extensions](https://github.com/nilson01/ashrink/blob/main/examples/regression_extensions.py).

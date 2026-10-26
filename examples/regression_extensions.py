"""Binary, count and grouped-outcome workflows with numeric predictors."""

import numpy as np
from ashrink import fit_logistic_regression, fit_poisson_regression, fit_mixed_regression

X = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5]])
binary = fit_logistic_regression(X, [0, 0, 1, 0, 1, 1], scales=[0.5, 1.0])
counts = fit_poisson_regression(
    X, [0, 1, 1, 2, 3, 4], exposure=[1, 1, 1, 1, 1, 1], scales=[0.5, 1.0]
)
grouped = fit_mixed_regression(
    X, [-1.0, 0.2, 1.0, 0.0, 1.2, 2.0], ["a"] * 3 + ["b"] * 3, scales=[0.5, 1.0]
)
print(binary.format_summary())
print(counts.format_summary())
print(grouped.format_summary())
print("Population means:", grouped.predict(X))
print("Group means:", grouped.predict(X, ["a"] * 3 + ["b"] * 3))

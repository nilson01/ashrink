"""Gaussian regression with coefficient summaries and mean predictions."""

import numpy as np
from ashrink import fit_regression

X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
y = np.array([-1.8, -0.1, 0.6, 3.3])
result = fit_regression(X, y, scales=[0.3, 1.0], labels=["x1", "x2"])
print(result.format_summary())
print("Predicted means:", result.predict(np.array([[0.2, 0.5], [0.8, -0.3]])))

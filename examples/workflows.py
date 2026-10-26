"""Small fixed inputs illustrating the selected analysis routes."""

import numpy as np
from ashrink import fit, fit_poisson, fit_binomial, fit_multivariate


def main():
    effects = np.array([-1.4, -0.6, -0.1, 0.0, 0.2, 0.7, 1.8, 2.2])
    se = np.array([0.7, 0.4, 0.5, 0.3, 0.5, 0.4, 0.8, 0.7])
    gaussian = fit(effects, se, scales=[0.2, 0.6, 1.5])
    print(gaussian.format_summary())
    asymmetric = fit(effects, se, family="halfuniform", scales=[0.5, 2.0])
    student = fit(effects[:3], se[:3], family="uniform", scales=[0.5, 2.0], df=8)
    rates = fit_poisson([0, 1, 3, 8, 12], exposure=[1.0, 1.0, 2.0, 2.0, 3.0])
    probabilities = fit_binomial([0, 2, 5, 8, 10], trials=10)
    paired = np.column_stack([effects[:4], effects[:4] * 0.7])
    multivariate = fit_multivariate(
        paired,
        np.full_like(paired, 0.5),
        covariances=[np.eye(2), np.ones((2, 2))],
        scales=[0.5, 1.5],
    )
    return {
        "gaussian": gaussian,
        "asymmetric": asymmetric,
        "student_t": student,
        "poisson": rates,
        "binomial": probabilities,
        "multivariate": multivariate,
    }


if __name__ == "__main__":
    main()

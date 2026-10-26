"""Finite, zero-centred priors for scalar adaptive shrinkage.

Normal scales are standard deviations. Uniform scales are half-widths;
``halfuniform`` components cover [-a, 0] and [0, a] separately. The point
mass at zero is a probability, not a finite value of the prior density.
"""

from dataclasses import dataclass, replace
import json
import numpy as np
from scipy import stats
from ._utils import readonly, vector


@dataclass(frozen=True, eq=False)
class MixturePrior:
    """A fixed component grid and its mixture probabilities.

    Use :func:`normal_mixture` or :func:`uniform_mixture` to construct a grid.
    All components have mode zero; asymmetric uniform components may have
    different weights on either side of zero.
    """

    family: str
    scales: np.ndarray
    weights: np.ndarray
    lower: np.ndarray = None
    upper: np.ndarray = None

    def __post_init__(self):
        if self.family not in ("normal", "uniform", "halfuniform"):
            raise ValueError("prior family must be normal, uniform or halfuniform")
        scales = vector(self.scales, "scales")
        weights = vector(self.weights, "weights")
        if scales.shape != weights.shape or (scales < 0).any():
            raise ValueError("scales must be nonnegative with one weight each")
        if (weights < 0).any() or not np.isfinite(weights.sum()) or weights.sum() <= 0:
            raise ValueError("weights must be nonnegative with a positive sum")
        if (scales == 0).sum() > 1:
            raise ValueError("at most one zero atom is allowed")
        if self.family == "normal":
            lower, upper = np.where(scales == 0, 0.0, -np.inf), np.where(scales == 0, 0.0, np.inf)
        else:
            if self.lower is None or self.upper is None:
                if self.family == "halfuniform":
                    raise ValueError("halfuniform requires component endpoints")
                lower, upper = -scales, scales
            else:
                lower, upper = vector(self.lower, "lower"), vector(self.upper, "upper")
            if lower.shape != scales.shape or upper.shape != scales.shape:
                raise ValueError("endpoints must match the component grid")
            if (lower > 0).any() or (upper < 0).any() or (lower > upper).any():
                raise ValueError("each interval must contain zero")
            if not np.array_equal(lower == upper, scales == 0):
                raise ValueError("only zero-scale components may be atoms")
            if self.family == "uniform" and not np.allclose(lower, -upper, rtol=0, atol=0):
                raise ValueError("uniform components must be symmetric about zero")
            if self.family == "halfuniform" and not np.all((lower == 0) | (upper == 0)):
                raise ValueError("halfuniform components must end at zero")
            if not np.array_equal(np.maximum(-lower, upper), scales):
                raise ValueError("scales must equal component half-widths")
        for name, value in [
            ("scales", scales),
            ("weights", weights / weights.sum()),
            ("lower", lower),
            ("upper", upper),
        ]:
            object.__setattr__(self, name, readonly(value))

    @property
    def atom_index(self):
        indexes = np.flatnonzero(self.scales == 0)
        return int(indexes[0]) if indexes.size else None

    @property
    def null_mass(self):
        return 0.0 if self.atom_index is None else float(self.weights[self.atom_index])

    @property
    def null_probability(self):
        return self.null_mass

    @property
    def mean(self):
        return (
            0.0
            if self.family == "normal"
            else float(self.weights @ ((self.lower + self.upper) / 2))
        )

    @property
    def description(self):
        scale = "standard deviations" if self.family == "normal" else "half-widths"
        return f"{self.family} mixture with {len(self.weights)} components ({scale}); null mass {self.null_mass:.6g}"

    def with_weights(self, weights):
        return replace(self, weights=weights)

    def density(self, x):
        """Continuous density only; ``null_mass`` gives the zero atom."""
        values = np.asarray(x, dtype=float)
        if np.isnan(values).any():
            raise ValueError("x cannot contain NaN")
        answer = np.zeros_like(values)
        for a, b, scale, weight in zip(self.lower, self.upper, self.scales, self.weights):
            if scale == 0 or weight == 0:
                continue
            if self.family == "normal":
                answer += weight * stats.norm.pdf(values, scale=scale)
            else:
                answer += weight * ((values >= a) & (values <= b)) / (b - a)
        return answer

    def cdf(self, x):
        """Right-continuous CDF, including the zero atom."""
        values = np.asarray(x, dtype=float)
        if np.isnan(values).any():
            raise ValueError("x cannot contain NaN")
        answer = np.zeros_like(values)
        for a, b, scale, weight in zip(self.lower, self.upper, self.scales, self.weights):
            if scale == 0:
                answer += weight * (values >= 0)
            elif self.family == "normal":
                answer += weight * stats.norm.cdf(values, scale=scale)
            else:
                answer += weight * np.clip((values - a) / (b - a), 0, 1)
        return answer

    def to_dict(self):
        result = {
            "family": self.family,
            "scales": self.scales.tolist(),
            "weights": self.weights.tolist(),
        }
        if self.family != "normal":
            result.update(lower=self.lower.tolist(), upper=self.upper.tolist())
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {
            "family",
            "scales",
            "weights",
            "lower",
            "upper",
        }:
            raise ValueError("invalid prior representation")
        return cls(**value)

    def to_json(self):
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))


def _grid(scales, point_mass):
    if not isinstance(point_mass, (bool, np.bool_)):
        raise ValueError("point_mass must be a boolean")
    scales = vector(scales, "scales")
    if (scales < 0).any():
        raise ValueError("scales must be nonnegative")
    if not point_mass and (scales == 0).any():
        raise ValueError("zero scales require point_mass=True")
    positive = np.unique(scales[scales > 0])
    if len(positive) == 0 and not point_mass:
        raise ValueError("at least one component is required")
    return np.r_[0.0, positive] if point_mass else positive


def normal_mixture(scales, weights=None, point_mass=True):
    """Construct zero-mean normal components, optionally with a zero atom.

    ``scales`` contains standard deviations. Duplicates are removed and the
    resulting grid is sorted. Supplied weights refer to that final grid,
    including its leading zero atom when requested.
    """
    grid = _grid(scales, point_mass)
    return MixturePrior("normal", grid, np.ones(len(grid)) if weights is None else weights)


def uniform_mixture(widths, weights=None, asymmetric=False, point_mass=True):
    """Construct symmetric or separate left/right uniform components.

    Positive widths a produce [-a,a], or [-a,0] followed by [0,a] when
    asymmetric=True. Within each side widths are sorted. A requested zero
    atom comes first; supplied weights refer to this final component order.
    """
    if not isinstance(asymmetric, (bool, np.bool_)):
        raise ValueError("asymmetric must be a boolean")
    grid = _grid(widths, point_mass)
    if not asymmetric:
        return MixturePrior("uniform", grid, np.ones(len(grid)) if weights is None else weights)
    positive = grid[grid > 0]
    scales = np.r_[([0.0] if point_mass else []), positive, positive]
    lower = np.r_[([0.0] if point_mass else []), -positive, np.zeros(len(positive))]
    upper = np.r_[([0.0] if point_mass else []), np.zeros(len(positive)), positive]
    return MixturePrior(
        "halfuniform", scales, np.ones(len(scales)) if weights is None else weights, lower, upper
    )

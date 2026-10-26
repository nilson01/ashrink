"""Named-table inputs and portable posterior-result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _column(data, name):
    if not isinstance(name, str) or not name:
        raise ValueError("column names must be nonempty strings")
    try:
        return np.asarray(data[name])
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"column {name!r} is not available") from exc


def fit_frame(data, estimate="estimate", standard_error="se", *, label=None, df=None, **options):
    """Fit scalar effects from a pandas DataFrame or mapping of columns.

    ``df`` may name a column of Student-t degrees of freedom or supply numeric
    degrees of freedom. A label column is retained for result tables.
    """
    from .univariate import fit

    degrees = _column(data, df) if isinstance(df, str) else df
    labels = _column(data, label) if label is not None else None
    return fit(
        _column(data, estimate), _column(data, standard_error), df=degrees, labels=labels, **options
    )


def fit_poisson_frame(data, count="count", exposure="exposure", **options):
    """Fit Poisson rates from named count and exposure columns."""
    from .counts import fit_poisson

    return fit_poisson(_column(data, count), _column(data, exposure), **options)


def fit_binomial_frame(data, successes="successes", trials="trials", **options):
    """Fit binomial probabilities from named success and trial columns."""
    from .counts import fit_binomial

    return fit_binomial(_column(data, successes), _column(data, trials), **options)


def fit_multivariate_frame(data, estimates, standard_errors, **options):
    """Fit effect/SE matrices assembled from matching named columns."""
    from .multivariate import fit_multivariate

    if isinstance(estimates, str) or isinstance(standard_errors, str):
        raise ValueError("supply sequences of effect and standard-error names")
    estimates, standard_errors = list(estimates), list(standard_errors)
    if len(estimates) != len(standard_errors) or len(estimates) < 2:
        raise ValueError("supply at least two paired effect and standard-error columns")
    if len(set(estimates)) != len(estimates) or len(set(standard_errors)) != len(standard_errors):
        raise ValueError("effect and standard-error columns must be distinct")
    return fit_multivariate(
        np.column_stack([_column(data, x) for x in estimates]),
        np.column_stack([_column(data, x) for x in standard_errors]),
        **options,
    )


def read_csv(path):
    """Read a headered numeric CSV into a mapping of NumPy columns."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        if not names or len(names) != len(set(names)) or any(not x for x in names):
            raise ValueError("CSV needs distinct, nonempty column names")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("CSV rows do not match the header")
    result = {}
    for name in names:
        values = [row[name] for row in rows]
        try:
            result[name] = np.asarray([np.nan if not x.strip() else float(x) for x in values])
        except ValueError:
            result[name] = np.asarray(values)
    return result


def save_result(result, path, *, overwrite=False):
    """Save a versioned JSON result; no executable objects are stored.

    A new file is required unless ``overwrite=True``. Serialization finishes
    before the destination is opened.
    """
    text = json.dumps(result.to_dict(), allow_nan=False, indent=2) + "\n"
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(text)


def load_result(path):
    """Restore posterior calculations from a JSON result without refitting."""

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant {value!r} is not allowed")

    record = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid_constant)
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("unsupported result schema")
    model = record.get("model")
    if model in ("gaussian", "student_t"):
        from .univariate import ASHResult

        return ASHResult.from_dict(record)
    if model in ("poisson", "binomial"):
        from .counts import CountResult

        return CountResult.from_dict(record)
    if model == "multivariate":
        from .multivariate import MultivariateResult

        return MultivariateResult.from_dict(record)
    raise ValueError("unknown result model")


def to_records(result, *, level=0.95):
    """Return one plain record per observation and condition.

    Bounds are equal-tailed empirical-Bayes posterior credible intervals,
    conditional on the fitted mixture weights.
    """
    mean = np.asarray(result.posterior_mean)
    sd = np.asarray(result.posterior_sd)
    interval = np.asarray(result.interval(level))
    labels = getattr(result, "labels", None)
    multivariate = mean.ndim == 2
    shape = mean.shape if multivariate else (len(mean), 1)
    values = mean.reshape(shape)
    sds = sd.reshape(shape)
    bounds = interval.reshape(*shape, 2)
    records = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            row = {
                "row": i,
                "posterior_mean": float(values[i, j]),
                "posterior_sd": float(sds[i, j]),
                "lower": float(bounds[i, j, 0]),
                "upper": float(bounds[i, j, 1]),
            }
            if multivariate:
                row["condition"] = j
            if labels is not None:
                row["label"] = str(labels[i])
            for name in ["lfdr", "lfsr", "qvalue", "svalue"]:
                row[name] = float(np.asarray(getattr(result, name)).reshape(shape)[i, j])
            records.append(row)
    return records


def write_csv(result, path, *, level=0.95, overwrite=False):
    """Write posterior summaries to a headered CSV file."""
    rows = to_records(result, level=level)
    with Path(path).open("w" if overwrite else "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

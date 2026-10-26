# Installation

From GitHub, with Git installed:

```sh
python -m pip install 'git+https://github.com/nilson01/ashrink.git'
```

From a local checkout:

```sh
python -m pip install .
```

For plotting and pandas tables:

```sh
python -m pip install '.[plot,table]'
```

The package is not published on PyPI. R is not a runtime dependency.

## Tested environments

The installed wheel passes 116 tests on macOS with Python 3.10.22
(NumPy 1.24.4, SciPy 1.11.4) and Python 3.14.3
(NumPy 2.5.4, SciPy 1.18.1).

[CI run 40295672151](https://github.com/nilson01/ashrink/actions/runs/40295672151)
also passes all 116 tests, with no failures, errors or skips, for commit
`a36b512` on Linux with Python 3.10.22 and 3.14.8, and on Windows Server 2025
with Python 3.13.16. The Linux Python 3.10 run uses NumPy 1.24.4,
SciPy 1.11.4 and Matplotlib 3.7.5 to check the minimum supported dependencies.

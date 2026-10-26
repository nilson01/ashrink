# Installation

From a local checkout:

```sh
python -m pip install .
```

For plotting and pandas tables:

```sh
python -m pip install '.[plot,table]'
```

With Git installed and access to the repository:

```sh
python -m pip install 'git+https://github.com/nilson01/ashrink.git@implement/first-release'
```

The package is not published on PyPI. R is not a runtime dependency.

## Tested environments

The installed wheel passes 116 tests on macOS with Python 3.10.22
(NumPy 1.24.4, SciPy 1.11.4) and Python 3.14.3
(NumPy 2.5.4, SciPy 1.18.1). Windows has not yet been tested.

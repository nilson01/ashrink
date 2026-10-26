# Contributing

Keep changes focused and preserve the documented statistical assumptions and
API behavior.
Include an independent known-answer check for a numerical change, plus relevant
input and failure cases. Add a short installed-package example for a new fitting
method or input format. Document changes to parameter scales, priors, output shapes
or defaults.

Development tools:

```sh
python -m pip install -e '.[dev,plot,table]'
python -m pytest
python -m ruff check src tests examples
python -m build
```

Reference-software tools belong outside runtime dependencies. Keep personal
paths, credentials, working notes and unreviewed study outputs out of commits.

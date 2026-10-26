# Development

## Install a source checkout

```sh
git clone https://github.com/nilson01/ashrink.git
cd ashrink
python -m pip install -e '.[dev,plot,table]'
```

Run the implementation checks from the repository root:

```sh
python -m ruff check src tests examples
python -m pytest tests
python -m build
```

Tests use committed numerical fixtures. They do not invoke R. To regenerate
fixtures with the pinned R packages, follow [reference checks](reference-checks.md).
Those comparisons and analytic checks establish specific numerical behavior;
they do not establish statistical performance across repeated datasets.

## Preview the documentation

The site uses MkDocs with search and MathJax for formulas.
Install the documentation dependencies separately from the package:

```sh
python -m pip install mkdocs==1.6.1 pymdown-extensions==10.21.3
python -m mkdocs build --strict
python -m mkdocs serve
```

The build reads `mkdocs.yml` and the Markdown pages and assets under `docs/`.
The MkDocs build copies the reference PDF. Its editable source is
`docs/manual/reference.md`. To rebuild it, install Pandoc (or `pypandoc_binary`)
and a TeX distribution with `pdflatex`, then run `python tools/build_manual.py`.
Generated TeX is temporary.

Pull requests build the site. Pushes to `main` and manual runs on `main` also
publish the built site through GitHub Pages when Pages is configured to use
GitHub Actions. The documentation workflow does not rerun the package test suite.

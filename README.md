# oast

This package provides our awesome spectral toolbox for performing analysis of hyperspectral data.

## Development

### Setup

In the cloned git repository, create your virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate  # or .venv/bin/activate.fish, or whatever
```
and install this package in "editable" mode:
```
pip install --editable .
```
If you get an error like `ERROR: File "setup.py" not found.`, then upgrade pip (`pip install --upgrade pip`) and retry.

Now, you can run the console scripts and such, with all the necessary dependencies, but the local code is still being run directly, so any changes made are immediately reflected.

### Building and releasing for PyPI

With `build` installed (`pip install --upgrade build`), you can then run:
```
python3 -m build
```
You now have your distribution ready (e.g. a tar.gz file and a .whl file in the dist directory), which you can upload to PyPI! With `twine`, upload the distribution with:
```
twine upload dist/*
```

### Releasing for conda

This package is distributed for conda through the `conda-forge` channel. When a new version is released on PyPI, that new version should also be released on conda-forge.

It is highly likely that a bot will notice the new PyPI version and create a pull request on https://github.com/conda-forge/oast-feedstock automatically that can be checked and merged. This is the preferred method. If such a pull request doesn't arrive, update the feedstock manually as follows.

Clone your fork of the conda feedstock (https://github.com/conda-forge/oast-feedstock) and update the `recipe/meta.yaml` file to reflect the new PyPI release. If you have `grayskull` installed (`pipx install grayskull`), you can generate a recipe that has the new PyPI release information with
```
grayskull pypi --strict-conda-forge oast
```
and manually copy over the version, sha256, etc to the `recipe/meta.yaml` file in the feedstock repository. Don't blindly copy over the whole file.

Commit and push the recipe change and open a pull request to https://github.com/conda-forge/oast-feedstock. On the pull request, comment "@conda-forge-admin please rerender" to rerender it. Once the pull request is merged, conda-forge should build and publish the new version.

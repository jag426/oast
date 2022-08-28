# oast

This package provides our awesome spectral toolbox for performing analysis of hyperspectral data.

## Development

### Building and releasing for PyPI

With `build` installed (`pip install --upgrade build`), you can then run:
```
python3 -m build
```
You now have your distribution ready (e.g. a tar.gz file and a .whl file in the dist directory), which you can upload to PyPI! With `twine`, upload the distribution with:
```
twine upload dist/*
```

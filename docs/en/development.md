# Development and documentation

<div class="language-switcher">English · <a href="../zh_CN/development.html">简体中文</a></div>

## Repository layout

- Public headers: `include/holistic_motion/`
- C++ implementations: `src/`
- Python bindings: `pybind/module.cpp`
- Python package: `python/holistic_motion/`
- C++ tests: `tests/cpp/`
- Python tests: `tests/python/`
- Conan consumer test: `test_package/`

## Documentation build

```bash
python -m pip install '.[docs]'
./scripts/docs.sh
```

The build treats Sphinx warnings and invalid API references as errors. Generated
files remain under `docs/_build/` and must not be committed.

When changing public behavior:

1. Update the relevant public header or Python docstring.
2. Add or update a focused test.
3. Update both English and Chinese guide pages.
4. Run `./scripts/docs.sh` and the affected test suite.

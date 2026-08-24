# Project layout

The repository root is the project boundary. It deliberately does not add a
second `holistic_motion/` wrapper around `include`, `src`, `bindings`, and
`python`: the C++ install namespace already lives at
`include/holistic_motion/`, while the Python import namespace lives at
`python/holistic_motion/`.

```text
include/holistic_motion/   public C++ API, grouped by robotics domain
src/                       C++ implementation mirroring the public modules
bindings/python/           compiled Python binding translation units
python/holistic_motion/    pure-Python trajectory, retargeting, and visualization APIs
examples/python/           examples grouped by domain
tests/cpp/                 C++ unit and integration tests
tests/python/              Python unit and integration tests
cmake/modules/             target source lists and build modules
docs/en, docs/zh_CN/       bilingual user and developer documentation
scripts/                   supported developer entry points
```

Legacy launchers remain under `python/examples/` while examples migrate to the
domain-oriented tree. New reusable visualization or collision policy code
belongs in `python/holistic_motion/`, not in an example script.

Algorithm integrations are organized by domain rather than upstream project
name: TOPPRA timing belongs in `python/holistic_motion/trajectory/`, while
Pink-style IK belongs in `python/holistic_motion/kit/retargeting/`. Upstream
repositories are neither nested in this tree nor registered as submodules.

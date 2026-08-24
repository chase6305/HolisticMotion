"""Shared Sphinx configuration for the English and Chinese sites."""

from pathlib import Path
import subprocess
import sys


repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root / "python"))

project = "HolisticMotion"
author = "HolisticMotion contributors"
release = "0.1.0"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_mock_imports = ["_holistic_motion", "holistic_motion._holistic_motion"]
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "substitution"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

doxygen_output = repository_root / "docs" / "_build" / "doxygen" / "xml"
if not (doxygen_output / "index.xml").exists():
    subprocess.run(
        ["doxygen", str(repository_root / "docs" / "Doxyfile")],
        cwd=repository_root,
        check=True,
    )

breathe_projects = {"HolisticMotion": str(doxygen_output)}
breathe_default_project = "HolisticMotion"
breathe_default_members = ("members", "undoc-members")

html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"navigation_with_keys": True}

# Breathe emits references to third-party C++ types such as Eigen::VectorXd.
# They remain readable signatures but cannot be resolved by this project's
# inventory, so strict missing-reference checking is intentionally disabled.
nitpicky = False

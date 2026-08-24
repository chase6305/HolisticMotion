#!/usr/bin/env python3
"""Compatibility launcher for examples/python/collision/basic_query.py."""

import runpy
from pathlib import Path

runpy.run_path(
    Path(__file__).resolve().parents[2]
    / "examples/python/collision/basic_query.py",
    run_name="__main__",
)

#!/usr/bin/env python3
"""Compatibility launcher for the organized collision path scanner."""

import runpy
from pathlib import Path

runpy.run_path(
    Path(__file__).resolve().parents[2]
    / "examples/python/collision/path_scan.py",
    run_name="__main__",
)

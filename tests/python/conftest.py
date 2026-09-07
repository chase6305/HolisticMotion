"""Optional generated tests use bounded, reproducible CI workloads."""

import importlib.util
import os

if importlib.util.find_spec("hypothesis") is not None:
    from hypothesis import settings

    settings.register_profile(
        "ci", max_examples=40, deadline=None, derandomize=True, print_blob=True
    )
    settings.register_profile("deep", max_examples=300, deadline=None, print_blob=True)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

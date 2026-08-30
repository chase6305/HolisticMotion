import os
import subprocess
import sys
from pathlib import Path


def test_import():
    import holistic_motion

    assert holistic_motion.__name__ == "holistic_motion"


def test_continuous_kinematics_api_imports():
    from holistic_motion.kinematics import (
        FEPContinuousOptions,
        FEPContinuousResult,
        FEPContinuousTracker,
        SRSContinuousOptions,
        SRSContinuousResult,
        SRSContinuousTracker,
    )

    assert FEPContinuousOptions.__name__ == "FEPContinuousOptions"
    assert FEPContinuousResult.__name__ == "FEPContinuousResult"
    assert FEPContinuousTracker.__name__ == "FEPContinuousTracker"
    assert SRSContinuousOptions.__name__ == "SRSContinuousOptions"
    assert SRSContinuousResult.__name__ == "SRSContinuousResult"
    assert SRSContinuousTracker.__name__ == "SRSContinuousTracker"


def test_pure_python_toolkit_import_does_not_load_native_extension():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["HOLISTICMOTION_PURE_PYTHON"] = "1"
    environment["PYTHONPATH"] = str(repository / "python")
    command = (
        "import holistic_motion as hm; "
        "from holistic_motion.kit import retargeting; "
        "from holistic_motion.trajectory import ToppraTrajectory; "
        "assert not hasattr(hm, 'Robot'); "
        "assert not hasattr(hm, 'os'); "
        "assert retargeting.RetargetingMode.DUAL_ARM.value == 'dual_arm'; "
        "assert retargeting.CuroboRetargetingSolver is not None; "
        "assert ToppraTrajectory is not None"
    )

    subprocess.run(
        [sys.executable, "-c", command],
        cwd=repository,
        env=environment,
        check=True,
    )

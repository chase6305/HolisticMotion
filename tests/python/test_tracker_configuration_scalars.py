"""The native integer fast path retains the accepted configuration types."""

from enum import IntEnum
from types import SimpleNamespace

import numpy as np
import pytest
from holistic_motion.kinematics import SRSContinuousTracker


class Branch(IntEnum):
    NEGATIVE = -1


class Integer(int):
    pass


@pytest.mark.parametrize("kind", [int, np.int32, np.int64, Integer, Branch])
def test_configuration_integers_are_returned_as_owned_builtin_values(kind):
    branch, redundancy = SRSContinuousTracker._validated_configuration(
        SimpleNamespace(
            shoulder=kind(-1), elbow=1, wrist=-1, redundancy=np.float64(0.25)
        )
    )
    assert branch == (-1, 1, -1)
    assert all(type(value) is int for value in branch)
    assert type(redundancy) is float
    assert redundancy == 0.25


@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 1.0, np.float64(1.0), "1", None]
)
def test_configuration_scalar_fast_path_rejects_noninteger_branches(value):
    with pytest.raises(ValueError, match="invalid configuration"):
        SRSContinuousTracker._validated_configuration(
            SimpleNamespace(shoulder=value, elbow=1, wrist=1, redundancy=0.0)
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "invalid", None, 1j])
def test_configuration_integer_fast_path_still_validates_redundancy(value):
    with pytest.raises(ValueError, match="invalid configuration"):
        SRSContinuousTracker._validated_configuration(
            SimpleNamespace(shoulder=1, elbow=1, wrist=1, redundancy=value)
        )

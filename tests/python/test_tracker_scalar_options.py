"""Tracker scalar options must own immutable, validated numeric values."""

from decimal import Decimal

import numpy as np
import pytest
from holistic_motion.kinematics import FEPContinuousOptions, SRSContinuousOptions

SCALAR_FIELDS = (
    "position_weight",
    "velocity_weight",
    "acceleration_weight",
    "branch_switch_penalty",
    "joint_limit_weight",
    "singular_value_threshold",
    "position_tolerance",
    "angle_tolerance",
)


@pytest.mark.parametrize("options_type", [SRSContinuousOptions, FEPContinuousOptions])
@pytest.mark.parametrize("name", SCALAR_FIELDS)
def test_tracker_scalar_options_do_not_alias_numpy_inputs(options_type, name):
    value = np.array(0.25)
    options = options_type(**{name: value})
    value[...] = np.nan
    assert isinstance(getattr(options, name), float)
    assert getattr(options, name) == 0.25


@pytest.mark.parametrize("options_type", [SRSContinuousOptions, FEPContinuousOptions])
@pytest.mark.parametrize("name", SCALAR_FIELDS)
@pytest.mark.parametrize("value", [[0.25], np.array([[0.25]])])
def test_tracker_scalar_options_reject_non_scalar_shapes(options_type, name, value):
    with pytest.raises(ValueError, match="scalar"):
        options_type(**{name: value})


@pytest.mark.parametrize("options_type", [SRSContinuousOptions, FEPContinuousOptions])
@pytest.mark.parametrize("value", [np.float64(0.25), np.array(0.25), Decimal("0.25")])
def test_tracker_scalar_options_keep_real_scalar_inputs(options_type, value):
    options = options_type(position_weight=value)
    assert options.position_weight == 0.25
    assert isinstance(options.position_weight, float)

"""Configuration setters must reject complex scalars before conversion."""

import numpy as np
import pytest


@pytest.mark.parametrize("family", ["SRSConfiguration", "FEPConfiguration"])
@pytest.mark.parametrize("field", ["shoulder", "elbow", "wrist", "redundancy"])
@pytest.mark.parametrize("representation", ["complex64", "complex128", "object"])
@pytest.mark.parametrize("imaginary", [0.0, 4.0])
def test_native_configuration_rejects_complex_without_mutation(
    family, field, representation, imaginary
):
    import holistic_motion as hm

    config = getattr(hm, family)()
    previous = 0.35 if field == "redundancy" else -1
    setattr(config, field, previous)
    value = 1 + imaginary * 1j
    if representation == "complex64":
        value = np.complex64(value)
    elif representation == "complex128":
        value = np.complex128(value)
    else:
        value = np.array(value, dtype=object)
    with pytest.raises(TypeError, match="real"):
        setattr(config, field, value)
    assert getattr(config, field) == previous


@pytest.mark.parametrize("family", ["SRSConfiguration", "FEPConfiguration"])
@pytest.mark.parametrize("field", ["shoulder", "elbow", "wrist", "redundancy"])
def test_native_configuration_preserves_real_numpy_scalars(family, field):
    import holistic_motion as hm

    config = getattr(hm, family)()
    value = np.float64(0.35) if field == "redundancy" else np.int64(-1)
    setattr(config, field, value)
    assert getattr(config, field) == value

"""Named scalar coordinates across chain and Pinocchio models.

Configurations have ``nq`` coordinates; velocities, displacements and gradients
have ``nv`` tangent coordinates. They differ for continuous and floating joints.
Mappings are snapshots of model topology and can be reused across queries.
"""

import math
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ._validation import real_array

__all__ = ["JointCoordinate", "JointPositionMap"]
_MAX_INDEX = int(np.iinfo(np.intp).max)


@dataclass(frozen=True)
class JointCoordinate:
    """A named joint block; indices refer to the full model, not a chain."""

    name: str
    idx_q: int
    nq: int
    idx_v: int
    nv: int


@dataclass(frozen=True, init=False)
class JointPositionMap:
    """Immutable mapping of ordered scalar joints into a full configuration.

    ``nq=2, nv=1`` encodes a continuous angle as cosine then sine. Multi-DOF
    joints cannot be selected, but their reference coordinates are preserved.
    Calls own their output arrays and can run concurrently on the same mapping.
    """

    nq: int
    nv: int
    joints: tuple
    _q_indices: np.ndarray = field(init=False, repr=False, compare=False)
    _v_indices: np.ndarray = field(init=False, repr=False, compare=False)
    _pair_positions: np.ndarray = field(init=False, repr=False, compare=False)
    _pair_q_indices: np.ndarray = field(init=False, repr=False, compare=False)

    def __init__(self, nq, nv, joints):
        dimensions = (nq, nv)
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in dimensions):
            raise TypeError("model nq and nv must be integers")
        if min(dimensions) < 0:
            raise ValueError("model nq and nv must be non-negative")
        nq, nv = int(nq), int(nv)
        if max(nq, nv) > _MAX_INDEX:
            raise ValueError("model nq and nv exceed the NumPy index range")
        selected = []
        names, q_indices, v_indices = set(), set(), set()
        for source in joints:
            joint = JointCoordinate(
                source.name, source.idx_q, source.nq, source.idx_v, source.nv
            )
            if not isinstance(joint.name, str) or not joint.name or joint.name in names:
                raise ValueError("joint names must be nonempty and unique")
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in (joint.idx_q, joint.nq, joint.idx_v, joint.nv)
            ):
                raise TypeError("joint coordinate indices and sizes must be integers")
            # Normalize NumPy integer scalars before subtracting block sizes:
            # unsigned subtraction could otherwise wrap and accept a bad block.
            joint = JointCoordinate(
                joint.name,
                int(joint.idx_q),
                int(joint.nq),
                int(joint.idx_v),
                int(joint.nv),
            )
            if joint.nv != 1 or joint.nq not in (1, 2):
                raise ValueError(
                    f"joint position mapping requires a scalar joint: {joint.name}"
                )
            if not (0 <= joint.idx_q <= nq - joint.nq and 0 <= joint.idx_v < nv):
                raise ValueError(
                    f"joint coordinate block is outside model nq/nv: {joint.name}"
                )
            block = set(range(joint.idx_q, joint.idx_q + joint.nq))
            if q_indices.intersection(block) or joint.idx_v in v_indices:
                raise ValueError("joint coordinate blocks must not overlap")
            names.add(joint.name)
            q_indices.update(block)
            v_indices.add(joint.idx_v)
            selected.append(joint)
        object.__setattr__(self, "nq", int(nq))
        object.__setattr__(self, "nv", int(nv))
        object.__setattr__(self, "joints", tuple(selected))
        # Prepare topology once. Private read-only indices are excluded from
        # value equality/hash; the immutable joint descriptors define identity.
        pair_positions = [i for i, joint in enumerate(selected) if joint.nq == 2]
        for name, values in (
            ("_q_indices", [joint.idx_q for joint in selected]),
            ("_v_indices", [joint.idx_v for joint in selected]),
            ("_pair_positions", pair_positions),
            ("_pair_q_indices", [selected[i].idx_q for i in pair_positions]),
        ):
            indices = np.array(values, dtype=np.intp)
            indices.setflags(write=False)
            object.__setattr__(self, name, indices)

    @classmethod
    def from_model(cls, model, joint_names):
        """Select names from a native collision model or Python Pinocchio model.

        No Pinocchio import or native-extension load is needed. The model must
        expose ``joint_coordinates`` or Pinocchio's names/idx_qs/nqs/idx_vs/nvs.
        """
        if isinstance(joint_names, str):
            raise TypeError("joint_names must be a sequence of names")
        coordinates = getattr(model, "joint_coordinates", None)
        if coordinates is None:
            coordinates = (
                JointCoordinate(
                    model.names[i],
                    model.idx_qs[i],
                    model.nqs[i],
                    model.idx_vs[i],
                    model.nvs[i],
                )
                for i in range(1, len(model.names))
            )
        by_name = {}
        for joint in coordinates:
            if not isinstance(joint.name, str) or not joint.name:
                raise ValueError("model joint names must be nonempty strings")
            if joint.name in by_name:
                raise ValueError(f"duplicate model joint name: {joint.name}")
            by_name[joint.name] = joint
        selected = []
        for name in joint_names:
            if not isinstance(name, str) or name not in by_name:
                raise ValueError(f"unknown model joint: {name!r}")
            selected.append(by_name[name])
        return cls(model.nq, model.nv, selected)

    @staticmethod
    def _vector(value, size, name):
        result = real_array(value, name=name)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite vector of size {size}")
        return result

    def configuration_with_joint_positions(self, reference, joint_positions):
        """Copy reference, replacing selected joints in the supplied order."""
        result = self._vector(reference, self.nq, "reference").copy()
        positions = self._vector(joint_positions, len(self.joints), "joint_positions")
        result[self._q_indices] = positions
        if self._pair_positions.size == 1:
            position, index = self._pair_positions[0], self._pair_q_indices[0]
            result[index] = math.cos(positions[position])
            result[index + 1] = math.sin(positions[position])
        elif self._pair_positions.size:
            angles = positions[self._pair_positions]
            # Subnormal sine values are valid coordinates, including when the
            # application requests NumPy underflow exceptions elsewhere.
            with np.errstate(under="ignore"):
                result[self._pair_q_indices] = np.cos(angles)
                result[self._pair_q_indices + 1] = np.sin(angles)
        return result

    def joint_positions(self, configuration):
        """Extract positions; continuous joints return principal angles."""
        configuration = self._vector(configuration, self.nq, "configuration")
        result = configuration[self._q_indices]
        if self._pair_positions.size == 1:
            position, index = self._pair_positions[0], self._pair_q_indices[0]
            c, s = configuration[index : index + 2]
            if abs(math.hypot(c, s) - 1.0) > 1e-8:
                raise ValueError(
                    f"continuous joint coordinates must form a unit pair: {self.joints[position].name}"
                )
            result[position] = math.atan2(s, c)
        elif self._pair_positions.size:
            c = result[self._pair_positions]
            s = configuration[self._pair_q_indices + 1]
            with np.errstate(over="ignore", under="ignore"):
                invalid = np.abs(np.hypot(c, s) - 1.0) > 1e-8
            if np.any(invalid):
                index = self._pair_positions[np.flatnonzero(invalid)[0]]
                raise ValueError(
                    f"continuous joint coordinates must form a unit pair: {self.joints[index].name}"
                )
            with np.errstate(under="ignore"):
                result[self._pair_positions] = np.arctan2(s, c)
        return result

    def joint_velocities(self, velocity):
        """Restrict an nv velocity/displacement/gradient in joint order."""
        velocity = self._vector(velocity, self.nv, "velocity")
        return velocity[self._v_indices]

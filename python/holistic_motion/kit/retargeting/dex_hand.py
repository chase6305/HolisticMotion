"""Optional dex-retargeting vector backend for an independently modeled hand."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Optional

import numpy as np

from .pinocchio_solver import _positive_float, _readonly_array


def _names(values, name):
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of names")
    result = tuple(values)
    if not result or any(not isinstance(x, str) or not x for x in result):
        raise ValueError(f"{name} must contain non-empty names")
    return result


def _load_dex():
    try:
        import nlopt
        from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor
        from dex_retargeting.optimizer import VectorOptimizer
        from dex_retargeting.robot_wrapper import RobotWrapper
    except ImportError as error:
        raise ImportError(
            "DexHandRetargetingSolver requires the optional hand backend; "
            "install 'holistic-motion[hand-retargeting]' (Torch is required)"
        ) from error
    return nlopt, RobotWrapper, VectorOptimizer, MimicJointKinematicAdaptor


def _inactive_objective(_x, _grad):
    """Placeholder without references to a solver, target frame, or traceback."""

    raise RuntimeError("no active hand retargeting solve")


def _mimic_source_limits(source_limits, follower_limits, multiplier, offset):
    """Intersect finite source bounds with the actual float64 affine map."""

    sign_bit = 1 << 63
    mask = (1 << 64) - 1

    def ordered(value):
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        return (~bits & mask) if bits & sign_bit else bits | sign_bit

    def position(key):
        bits = key ^ sign_bit if key & sign_bit else ~key & mask
        return struct.unpack(">d", struct.pack(">Q", bits))[0]

    def mapped(key):
        value = multiplier * position(key) + offset
        return value if multiplier > 0.0 else -value

    low, high = map(float, follower_limits)
    if multiplier < 0.0:
        low, high = -high, -low
    left, right = map(ordered, source_limits)
    # IEEE-754 keys preserve numeric ordering, including across zero. Searching
    # these keys takes at most 64 iterations per boundary, even for subnormals.
    # Inverting then nudging rounded endpoints can discard valid locked poses.
    if mapped(left) < low:
        start, stop = left, right
        while start < stop:
            middle = (start + stop) // 2
            if mapped(middle) < low:
                start = middle + 1
            else:
                stop = middle
        left = start
    if mapped(right) > high:
        start, stop = left, right
        while start < stop:
            middle = (start + stop + 1) // 2
            if mapped(middle) > high:
                stop = middle - 1
            else:
                start = middle
        right = stop
    if not (low <= mapped(left) <= high and low <= mapped(right) <= high):
        raise ValueError("mimic and source joint limits have no feasible intersection")
    return position(left), position(right)


@dataclass(frozen=True)
class HandRetargetingResult:
    """Named full hand configuration; unsuccessful solves return the input seed.

    Success requires the maximum vector-error tolerance and, when optimization
    is needed, optimizer convergence. A direct hold reports zero evaluations
    and status zero (no native solve). It does not certify velocity,
    acceleration, or collision safety.
    """

    joint_names: tuple[str, ...]
    configuration: np.ndarray
    success: bool
    residual: float
    objective: float
    solve_ms: float
    evaluations: int
    optimizer_status: int
    termination_reason: str
    message: str = ""

    def __post_init__(self):
        names = _names(self.joint_names, "joint_names")
        if len(set(names)) != len(names):
            raise ValueError("result joint names must be unique")
        q = _readonly_array(
            self.configuration, shape=(len(names),), name="configuration"
        )
        if not isinstance(self.success, (bool, np.bool_)):
            raise TypeError("success must be boolean")
        for name in ("residual", "objective", "solve_ms"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in ("evaluations", "optimizer_status"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.evaluations < 0:
            raise ValueError("evaluations must be non-negative")
        if not isinstance(self.termination_reason, str) or not self.termination_reason:
            raise ValueError("termination_reason must be a non-empty string")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "configuration", q)

    @property
    def joint_positions(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(zip(self.joint_names, map(float, self.configuration)))
        )


class DexHandRetargetingSolver:
    """Retarget keypoint vectors through dex-retargeting 0.5 VectorOptimizer.

    ``keypoint_pairs`` is an (N, 2) array of origin/task keypoint indices;
    ``origin_links`` and ``task_links`` specify the corresponding robot vectors.
    Keypoints must use the hand model's base axes and length unit. Global
    translation cancels; rotation alignment is the caller's responsibility.

    Only scalar joints are supported. Direct mimic joints whose source is an
    optimized joint are supported, including induced source joint limits.
    The adapter applies no temporal output filter and no implicit model download.
    Internal derivatives also work inside Torch no-grad and inference contexts;
    the caller's gradient and inference modes are restored after each evaluation.
    """

    def __init__(
        self,
        urdf_path,
        *,
        joint_names: Sequence[str],
        origin_links: Sequence[str],
        task_links: Sequence[str],
        keypoint_pairs: Sequence[Sequence[int]],
        scaling: float = 1.0,
        huber_delta: float = 0.02,
        temporal_weight: float = 4e-3,
        vector_tolerance: float = 0.02,
        objective_tolerance: float = 1e-9,
        max_evaluations: int = 100,
    ):
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"URDF does not exist: {path}")
        names = _names(joint_names, "joint_names")
        origins = _names(origin_links, "origin_links")
        tasks = _names(task_links, "task_links")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        pairs = np.asarray(keypoint_pairs)
        if (
            pairs.shape != (len(origins), 2)
            or len(tasks) != len(origins)
            or pairs.dtype.kind not in "iu"
            or np.any(pairs < 0)
            or np.any(pairs > np.iinfo(np.intp).max)
        ):
            raise ValueError(
                "keypoint_pairs must contain one non-negative integer pair per link pair"
            )
        if any(a == b for a, b in zip(origins, tasks)) or np.any(
            pairs[:, 0] == pairs[:, 1]
        ):
            raise ValueError("vector origins and endpoints must differ")
        urdf = ET.parse(path).getroot()
        link_names = {link.get("name") for link in urdf.findall("link")}
        if (set(origins) | set(tasks)) - link_names:
            raise ValueError("vector endpoints must name URDF links, not joints")
        self.scaling = _positive_float(scaling, "scaling")
        self.huber_delta = _positive_float(huber_delta, "huber_delta")
        self.vector_tolerance = _positive_float(vector_tolerance, "vector_tolerance")
        self.objective_tolerance = _positive_float(
            objective_tolerance, "objective_tolerance"
        )
        temporal_weight = float(temporal_weight)
        if not np.isfinite(temporal_weight) or temporal_weight < 0.0:
            raise ValueError("temporal_weight must be finite and non-negative")
        if isinstance(max_evaluations, bool) or not isinstance(
            max_evaluations, Integral
        ):
            raise TypeError("max_evaluations must be an integer")
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        self.temporal_weight = temporal_weight
        self.keypoint_pairs = pairs.astype(np.intp, copy=True)
        self.keypoint_pairs.setflags(write=False)
        self._nlopt, wrapper, optimizer, adaptor = _load_dex()

        class ScalarHandRobot(wrapper):
            def compute_single_link_local_jacobian(self, qpos, link_id):
                # Some Pinocchio bindings squeeze a one-DOF Jacobian to (6,).
                # Upstream expects a matrix even for a single moving finger.
                value = super().compute_single_link_local_jacobian(qpos, link_id)
                return np.asarray(value, dtype=float).reshape(6, self.model.nv)

        try:
            self._robot = ScalarHandRobot(str(path))
        except NotImplementedError as error:
            raise ValueError(
                "hand backend requires a scalar-joint hand URDF (nq == nv)"
            ) from error
        if any(j.nq != 1 or j.nv != 1 for j in self._robot.model.joints[1:]):
            raise ValueError("hand backend supports scalar joints only")
        self.joint_names = tuple(self._robot.dof_joint_names)
        self.optimized_joint_names = names
        import torch

        # Cached index tensors participate in later backward passes, even when
        # the solver is constructed lazily inside a perception inference call.
        with torch.inference_mode(False), torch.no_grad():
            self._optimizer = optimizer(
                self._robot,
                list(names),
                list(origins),
                list(tasks),
                self.keypoint_pairs.T,
                huber_delta=self.huber_delta,
                norm_delta=temporal_weight,
                scaling=self.scaling,
            )
        self._limits = np.asarray(self._robot.joint_limits, dtype=float).copy()
        if not np.isfinite(self._limits).all() or np.any(
            self._limits[:, 0] > self._limits[:, 1]
        ):
            raise ValueError("hand joints require finite ordered position limits")
        self._target = self._optimizer.idx_pin2target
        self._mimic_names = set()
        target_limits = self._limits[self._target].copy()
        sources, mimics, multipliers, offsets = [], [], [], []
        for joint in urdf.findall("joint"):
            mimic = joint.find("mimic")
            if mimic is None:
                continue
            source, name = mimic.get("joint"), joint.get("name")
            if source not in names or name not in self.joint_names or name in names:
                raise ValueError(
                    "mimic joints must have a directly optimized source and cannot be optimized themselves"
                )
            multiplier, offset = (
                float(mimic.get("multiplier", "1")),
                float(mimic.get("offset", "0")),
            )
            if not np.isfinite([multiplier, offset]).all():
                raise ValueError("mimic coefficients must be finite")
            low, high = self._limits[self.joint_names.index(name)]
            index = names.index(source)
            if multiplier == 0.0:
                if not low <= offset <= high:
                    raise ValueError(
                        "constant mimic position violates its joint limits"
                    )
            else:
                target_limits[index] = _mimic_source_limits(
                    target_limits[index], (low, high), multiplier, offset
                )
            sources.append(source)
            mimics.append(name)
            multipliers.append(multiplier)
            offsets.append(offset)
        if np.any(target_limits[:, 0] > target_limits[:, 1]):
            raise ValueError(
                "mimic and source joint limits have no feasible intersection"
            )
        if mimics:
            self._optimizer.set_kinematic_adaptor(
                adaptor(self._robot, list(names), sources, mimics, multipliers, offsets)
            )
            self._mimic_names = set(mimics)
        self._target_limits = target_limits
        # Upstream expands bounds by 1e-3 by default. Hand outputs must obey
        # the caller's actual URDF limits, including mimic-induced bounds.
        self._optimizer.set_joint_limit(target_limits, epsilon=0.0)
        self._optimizer.opt.set_ftol_abs(self.objective_tolerance)
        self._optimizer.opt.set_maxeval(int(max_evaluations))
        links = tuple(dict.fromkeys((*origins, *tasks)))
        self._link_ids = tuple(self._robot.get_link_index(name) for name in links)
        self._origin_indices = np.array([links.index(name) for name in origins])
        self._task_indices = np.array([links.index(name) for name in tasks])
        self._neutral = np.clip(
            np.asarray(self._robot.q0, dtype=float),
            self._limits[:, 0],
            self._limits[:, 1],
        )
        self._neutral[self._target] = np.clip(
            self._neutral[self._target], target_limits[:, 0], target_limits[:, 1]
        )
        self._neutral = self._assemble(self._neutral[self._target], self._neutral)
        self._last_q = self._neutral.copy()

    def _assemble(self, selected, reference):
        q = reference.copy()
        q[self._target] = selected
        if self._optimizer.adaptor is not None:
            q = self._optimizer.adaptor.forward_qpos(q)
        return q

    def _configuration(self, values):
        if values is None:
            return self._last_q.copy()
        if not isinstance(values, Mapping):
            raise TypeError("seed must map hand joint names to positions")
        required = set(self.joint_names) - self._mimic_names
        if required - values.keys() or values.keys() - set(self.joint_names):
            raise ValueError(
                "seed must supply all independent hand joints and no unknown names"
            )
        q = self._neutral.copy()
        for name, value in values.items():
            q[self.joint_names.index(name)] = float(value)
        canonical = self._assemble(q[self._target], q)
        for name in self._mimic_names & values.keys():
            index = self.joint_names.index(name)
            if not np.isclose(q[index], canonical[index], rtol=0.0, atol=1e-10):
                raise ValueError("seed contains inconsistent mimic positions")
        if not self._valid(canonical):
            raise ValueError(
                "seed must contain finite positions within hand joint limits"
            )
        return canonical

    def _valid(self, q):
        return (
            q.shape == (len(self.joint_names),)
            and np.isfinite(q).all()
            and np.all(q >= self._limits[:, 0])
            and np.all(q <= self._limits[:, 1])
            and np.all(q[self._target] >= self._target_limits[:, 0])
            and np.all(q[self._target] <= self._target_limits[:, 1])
        )

    def reset(self, configuration: Optional[Mapping[str, float]] = None):
        self._last_q = (
            self._neutral.copy()
            if configuration is None
            else self._configuration(configuration)
        )

    def _objective(self, vectors, reference):
        import torch

        previous = reference[self._target].copy()
        # Targets captured by the native callback must be ordinary tensors even
        # when the caller runs a perception model in inference mode.
        with torch.inference_mode(False), torch.no_grad():
            base = self._optimizer.get_objective_function(
                vectors, reference[self._optimizer.idx_pin2fixed], previous
            )

        def objective(x, grad):
            # Upstream adds this term to the gradient but omits it from the
            # returned scalar. Retain its analytic/autograd chain and correct
            # the scalar, without calling the failure-swallowing retarget().
            # NLopt requests analytic derivatives independently of the caller's
            # Torch context. Scalar-only evaluations need no autograd graph.
            with torch.inference_mode(False), torch.set_grad_enabled(grad.size > 0):
                value = base(x, grad)
            delta = x - previous
            value += self.temporal_weight * float(delta @ delta)
            if not np.isfinite(value) or not np.isfinite(grad).all():
                raise ValueError(
                    "hand optimization objective and gradient must be finite"
                )
            return value

        return objective

    def _sync_optimizer_settings(self):
        # Validate the complete update before changing any native settings.
        # In particular, loss values and derivatives must share one weight.
        settings = {
            name: _positive_float(getattr(self, name), name)
            for name in (
                "scaling",
                "huber_delta",
                "vector_tolerance",
                "objective_tolerance",
            )
        }
        temporal_weight = float(self.temporal_weight)
        if not np.isfinite(temporal_weight) or temporal_weight < 0.0:
            raise ValueError("temporal_weight must be finite and non-negative")
        settings["temporal_weight"] = temporal_weight
        self._optimizer.opt.set_ftol_abs(settings["objective_tolerance"])
        self._optimizer.scaling = settings["scaling"]
        self._optimizer.huber_loss.beta = settings["huber_delta"]
        self._optimizer.norm_delta = temporal_weight
        for name, value in settings.items():
            setattr(self, name, value)

    def _residual(self, vectors, q):
        self._robot.compute_forward_kinematics(q)
        positions = np.array(
            [self._robot.get_link_pose(i)[:3, 3] for i in self._link_ids]
        )
        actual = positions[self._task_indices] - positions[self._origin_indices]
        return float(np.max(np.linalg.norm(actual - vectors * self.scaling, axis=1)))

    def solve(
        self, keypoints, *, seed: Optional[Mapping[str, float]] = None
    ) -> HandRetargetingResult:
        started = perf_counter()
        points = np.asarray(keypoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("keypoints must be a finite (N, 3) array")
        if int(self.keypoint_pairs.max()) >= len(points):
            raise ValueError("keypoint_pairs reference missing keypoints")
        vectors = points[self.keypoint_pairs[:, 1]] - points[self.keypoint_pairs[:, 0]]
        reference = self._configuration(seed)
        self._sync_optimizer_settings()
        if not np.isfinite(vectors * self.scaling).all():
            raise ValueError("scaled keypoint vectors must be finite")
        objective = self._objective(vectors, reference)
        residual = self._residual(vectors, reference)
        satisfied = residual <= self.vector_tolerance
        immobile = np.all(self._target_limits[:, 0] == self._target_limits[:, 1])
        if satisfied or immobile:
            # Holds need no Jacobian or native optimizer state. In particular,
            # do not reuse a preceding solve's NLopt status/evaluation count.
            result = HandRetargetingResult(
                joint_names=self.joint_names,
                configuration=reference,
                success=bool(satisfied),
                residual=residual,
                objective=objective(reference[self._target], np.empty(0)),
                solve_ms=(perf_counter() - started) * 1000.0,
                evaluations=0,
                optimizer_status=0,
                termination_reason="converged" if satisfied else "no_feasible_motion",
            )
            if satisfied:
                self._last_q = reference.copy()
            return result
        callback_errors = []

        def checked_objective(x, grad):
            try:
                return objective(x, grad)
            except Exception as error:
                callback_errors.append(error)
                raise

        opt = self._optimizer.opt
        reason, message = "optimizer_failed", ""
        candidate = reference
        try:
            opt.set_min_objective(checked_objective)
            try:
                selected = np.asarray(
                    opt.optimize(reference[self._target]), dtype=float
                )
                if selected.shape != (len(self._target),):
                    raise ValueError(
                        "optimizer returned an invalid hand configuration shape"
                    )
                candidate = self._assemble(selected, reference)
            except RuntimeError as error:
                if callback_errors:
                    raise callback_errors[-1]
                message = str(error)
            else:
                reason = "task_tolerance"
        finally:
            # NLopt owns its Python callback through a native reference invisible
            # to cyclic GC. Break opt -> callback -> solver -> opt on every exit,
            # including interrupts, without retaining old targets or tracebacks.
            opt.set_min_objective(_inactive_objective)
        status = int(opt.last_optimize_result())
        evaluations = int(opt.get_numevals())
        valid = self._valid(candidate)
        if not valid:
            reason = "invalid_result"
        elif reason != "optimizer_failed":
            if status == self._nlopt.MAXEVAL_REACHED:
                reason = "maximum_evaluations"
            elif status == self._nlopt.MAXTIME_REACHED:
                reason = "maximum_time"
            elif status <= 0:
                reason = "optimizer_failed"
            elif self._residual(vectors, candidate) <= self.vector_tolerance:
                reason = "converged"
        success = reason == "converged"
        q = candidate if success else reference
        result = HandRetargetingResult(
            joint_names=self.joint_names,
            configuration=q,
            success=success,
            residual=self._residual(vectors, q),
            objective=objective(q[self._target], np.empty(0)),
            solve_ms=(perf_counter() - started) * 1000.0,
            evaluations=evaluations,
            optimizer_status=status,
            termination_reason=reason,
            message=message,
        )
        if success:
            self._last_q = q.copy()
        return result

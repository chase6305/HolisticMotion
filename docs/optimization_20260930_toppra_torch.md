# Differentiable TOPPRA on CPU and CUDA

This extends the NumPy TOPPRA work in `2cf5043` with an optional tensor backend,
following the device-resident computation and active-constraint differentiation
approach of EmbodiChain's `feat/differentiable-toppra` branch (`49d76866`,
`748006a1`). It implements the numerical work directly in PyTorch, without a
Warp dependency or a CPU/NumPy fallback.

## API and numerical behavior

- Install `holistic-motion[differentiable]`; use `TorchToppraTrajectory` or
  `retime_path_torch` from `holistic_motion.trajectory`. NumPy exports stay lazy
  with respect to Torch and still work without it.
- Single `(N, D)` and batched `(B, N, D)` waypoints run on CPU or CUDA, honoring
  the current Torch CUDA stream. Limits support scalar, `(D,)`, and `(B, D)`
  inputs; endpoint speeds support scalar and `(B,)` inputs.
- Natural cubic splines retain normalized chord-length geometry. Equal numbers
  of subdivisions per spline segment keep tensor batch shapes fixed. Compare
  to NumPy with the same explicit grid; default grids can differ.
- The complete backward controllability pass supports nonzero terminal speeds.
  All forward interval half-planes are validated, preserving the conservative
  velocity and acceleration bounds throughout each interval.
- Pairwise projection bounds are selected in bounded chunks without autograd;
  selected pairs are re-evaluated with autograd. This avoids a quadratic saved
  gradient graph while retaining derivatives of the active constraint.
- Timing and solver arithmetic use float64. Motion samples retain waypoint
  dtype, including float16 and bfloat16. Invalid or infeasible input rejects
  the entire batch; nonrepresentable sampled outputs raise an error.
- Gradients cover waypoints, motion limits, boundary speeds, and sample times.
  They are piecewise derivatives; active-set and segment switches are not
  smooth. The square-root derivative at zero speed is defined as zero.
- First-order gradients are validated. CUDA gather backward may differ in
  summation roundoff between repetitions; full Jacobian checks allow only
  `1e-12` for this reentrancy check, independently of finite-difference error.

The backend synchronizes validation reductions with the host and does not
support CUDA Graph capture. Grid propagation remains sequential. CUDA support
does not imply a speedup for small batches, and no GPU performance claim is
made here. No robot assets are needed.

## Validation

Local hardware: NVIDIA GeForce RTX 5090 D v2; PyTorch 2.13.0+cu130,
Python 3.10, NumPy 2.2.6.

- 57 tensor tests pass, including 29 actual CUDA tests and 28 CPU tests.
  Full finite-difference Jacobians cover waypoint, limit, boundary-speed,
  duration, position, velocity, and acceleration outputs. Additional cases
  cover sampled-time derivatives, zero-speed endpoints, low precision,
  repeated backward, independent batch gradients, noncontiguous inputs,
  time-unit scaling, invalid input, and nondefault CUDA streams.
- Combined tensor/NumPy/native-import validation: 126 passed.
- Installed-package tensor and NumPy tests: 122 passed. Tests load the module
  copied by `cmake --install`, not a source-path replacement.
- Without Torch, NumPy 1.23.5 passes 68 NumPy/import tests; the optional tensor
  file skips. The missing-dependency error points to the installation extra.
- Thirty seeded random paths on each device agree with same-grid NumPy
  durations and produce finite gradients.
- The CUDA example differentiates two paths and shared motion limits. A
  16-path, seven-joint forward/backward run also completes with finite gradients.
- Ruff lint/format, whitespace checks, and strict English/Chinese documentation
  builds pass. CI adds CPU tests for both PyTorch 2.2.2 and the current release.

Reproduce the tensor tests without a native extension:

```bash
HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python \
  python -m pytest tests/python/test_toppra_torch.py -q
HOLISTICMOTION_PURE_PYTHON=1 PYTHONPATH=python \
  python examples/python/trajectory/toppra_differentiable.py --device cuda
```

CUDA cases skip only when no CUDA device is available. The default GitHub CPU
runners exercise CPU autograd; local results above include real CUDA execution.

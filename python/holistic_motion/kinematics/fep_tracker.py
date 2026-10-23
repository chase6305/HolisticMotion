"""Continuous branch tracking for offset seven-axis FEP solvers."""

from __future__ import annotations

from dataclasses import dataclass

from .srs_tracker import (
    SRSContinuousOptions,
    SRSContinuousResult,
    SRSContinuousTracker,
)


@dataclass(frozen=True)
class FEPContinuousOptions(SRSContinuousOptions):
    """Weights and hard limits for continuous FEP candidate selection."""


@dataclass(frozen=True)
class FEPContinuousResult(SRSContinuousResult):
    """One accepted sample from :class:`FEPContinuousTracker`."""


class FEPContinuousTracker(SRSContinuousTracker):
    """Track continuous FEP solutions with branch and dynamic constraints.

    History is mutable: use one tracker per worker and serialize reset, option
    changes and steps. The native FEP solver retains its read-only query contract.
    """

    _solver_label = "FEP"
    _options_type = FEPContinuousOptions
    _result_type = FEPContinuousResult

    def _all_configurations_method(self):
        import holistic_motion as hm

        return hm.FEPSolveMethod.ALL_CONFIGURATIONS

    def _seeded_method(self):
        import holistic_motion as hm

        return hm.FEPSolveMethod.SEEDED_NUMERICAL

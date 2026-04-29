"""SpatialOverlay — Flywheel overlay for spatial demos."""

from typing import Any, List, Optional, Set

from flywheel.protocols.interfaces.base_flywheel_overlay import BaseFlywheelOverlay
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.flywheel_overlay import FlywheelOverlay
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.artifacts.norm import Norm
from flywheel.protocols.enums import AuditStatus, CorrectionType, NormKind


class SpatialOverlay(BaseFlywheelOverlay):
    """Tracks audit coverage for spatial regions; provides spatial norms."""

    def __init__(self, norms: List[Norm] = None):
        self._version = 0
        self._covered: Set[str] = set()
        self._norms = norms or [Norm(
            id="SPATIAL_SUPPORT_REQUIRED",
            kind=NormKind.SPATIAL_BOUNDARY,
            spec={"require_support": True},
            severity=1.0,
            description="Points must lie within boundary of expert path",
        )]

    def overlay(
        self,
        context: Context,
        trajectory: Trajectory,
        flags: Optional[Any] = None,
    ) -> FlywheelOverlay:
        c_a = 0.2 if self._covered else 0.8
        return FlywheelOverlay(
            c_a=c_a,
            c_a_thresh=0.6,
            v_G=self.get_version(),
            audit_status=(AuditStatus.COVERED if c_a < 0.6
                          else AuditStatus.INSUFFICIENT_COVERAGE),
        )

    def apply_batch(self, batch: GovernanceBatch) -> bool:
        applied = False
        for lc in batch.local_corrections:
            if lc.correction_type == CorrectionType.AUDIT_COVERAGE_UPDATE:
                self._covered.add(lc.payload.get("case_class", ""))
                applied = True
        if applied:
            self._version += 1
        return True

    def get_version(self) -> str:
        return f"gov:v{self._version}"

    def get_norms(self) -> List[Norm]:
        return list(self._norms)

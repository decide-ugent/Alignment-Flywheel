"""DefaultQueryMerger — standard merge of Oracle and Flywheel signals."""

from flywheel.protocols.interfaces.base_query_merger import BaseQueryMerger
from flywheel.protocols.artifacts.oracle_raw_output import OracleRawOutput
from flywheel.protocols.artifacts.flywheel_overlay import FlywheelOverlay
from flywheel.protocols.artifacts.unified_query_result import UnifiedQueryResult


class DefaultQueryMerger(BaseQueryMerger):
    """Standard merge — copies fields from both signals into a UnifiedQueryResult."""

    def merge(
        self,
        oracle: OracleRawOutput,
        flywheel: FlywheelOverlay,
    ) -> UnifiedQueryResult:
        return UnifiedQueryResult(
            s=oracle.s,
            u=oracle.u,
            c_a=flywheel.c_a,
            u_thresh=oracle.u_thresh,
            c_a_thresh=flywheel.c_a_thresh,
            oracle_version=oracle.v_O,
            governance_version=flywheel.v_G,
            evidence_status=oracle.evidence_status,
            audit_status=flywheel.audit_status,
            evidence_hooks=oracle.evidence_hooks,
            audit_evidence=flywheel.audit_evidence,
        )

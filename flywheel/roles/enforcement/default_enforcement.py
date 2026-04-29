"""DefaultEnforcement — standard 3-check enforcement policy."""

from flywheel.protocols.interfaces.base_enforcement_policy import BaseEnforcementPolicy
from flywheel.protocols.artifacts.unified_query_result import UnifiedQueryResult
from flywheel.protocols.artifacts.enforcement_result import EnforcementResult
from flywheel.protocols.enums import EnforcementAction, EvidenceStatus


class DefaultEnforcement(BaseEnforcementPolicy):
    """Standard enforcement: uncertainty → safety → coverage."""

    def __init__(self, safety_margin: float = 0.4, audit_safety_margin: float = 0.85):
        self.safety_margin = safety_margin
        self.audit_safety_margin = audit_safety_margin

    def decide(self, unified: UnifiedQueryResult) -> EnforcementResult:
        reasons = []
        if unified.u >= unified.u_thresh:
            reasons.append(f"Oracle uncertain: u={unified.u:.2f}>={unified.u_thresh:.2f}")
            return EnforcementResult(action=EnforcementAction.ESCALATE, reasons=reasons)
        if unified.s < self.safety_margin:
            reasons.append(f"Unsafe: s={unified.s:.2f}<{self.safety_margin}")
            return EnforcementResult(action=EnforcementAction.BLOCK, reasons=reasons)
        if unified.c_a >= unified.c_a_thresh:
            reasons.append(f"Low coverage: c_a={unified.c_a:.2f}")
            if unified.evidence_status in (
                EvidenceStatus.INSUFFICIENT,
                EvidenceStatus.CONFLICTING,
            ):
                if unified.s < self.audit_safety_margin:
                    return EnforcementResult(
                        action=EnforcementAction.ESCALATE, reasons=reasons)
            reasons.append("Allowed with audit flag")
            return EnforcementResult(action=EnforcementAction.ALLOW, reasons=reasons)
        reasons.append(f"Safe: s={unified.s:.2f}")
        return EnforcementResult(action=EnforcementAction.ALLOW, reasons=reasons)

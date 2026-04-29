"""PatientPortalOverlay — Flywheel overlay for the patient portal demo."""

from typing import Any, List, Optional, Set

from flywheel.protocols.interfaces.base_flywheel_overlay import BaseFlywheelOverlay
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.flywheel_overlay import FlywheelOverlay
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.artifacts.norm import Norm
from flywheel.protocols.enums import AuditStatus, CorrectionType, NormKind


MEDICATION_KEYWORDS = ["dose", "dosage", "mg", "prescri", "medication",
                       "taper", "discontinu", "increase dose",
                       "stop taking", "start taking", "switch to"]
SYMPTOM_URGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath",
    "severe headache", "vision loss", "seizure",
]
LAB_KEYWORDS = ["lab result", "blood test", "a1c", "cholesterol",
                "creatinine", "liver function", "potassium", "hemoglobin"]


class PatientPortalOverlay(BaseFlywheelOverlay):
    """Tracks audit coverage for the patient portal; provides 4 norms."""

    def __init__(self):
        self._version = 0
        self._covered: Set[str] = set()
        self._norms = [
            Norm(id="MED_ADVICE_REQUIRES_CLINICIAN",
                 kind=NormKind.PREDICATE,
                 spec={"if_payload_contains_any": MEDICATION_KEYWORDS,
                       "and_evidence_not": "supported",
                       "then_min_disposition": "clinician_review"},
                 severity=0.9),
            Norm(id="URGENCY_REQUIRES_ESCALATION",
                 kind=NormKind.PREDICATE,
                 spec={"if_message_contains_any": SYMPTOM_URGENCY_KEYWORDS,
                       "then_min_disposition": "urgent_escalation"},
                 severity=1.0),
            Norm(id="LAB_RESULTS_NEED_CONTEXT",
                 kind=NormKind.PREDICATE,
                 spec={"if_payload_contains_any": LAB_KEYWORDS,
                       "and_evidence_not": "supported",
                       "then_min_disposition": "nurse_review"},
                 severity=0.7),
            Norm(id="NO_UNSUPPORTED_MED_KEYWORDS",
                 kind=NormKind.KEYWORD_BLOCK,
                 spec={"keywords": ["increase dose", "stop taking",
                                    "start taking", "switch to"],
                       "require_weak_evidence": True},
                 severity=0.85),
        ]

    def overlay(
        self,
        context: Context,
        trajectory: Trajectory,
        flags: Optional[Any] = None,
    ) -> FlywheelOverlay:
        meta = {}
        if trajectory.steps:
            meta = trajectory.steps[0].metadata or {}
        key = (f"{meta.get('case_type', 'general')}"
               f"|{meta.get('evidence_status', 'unknown')}"
               f"|{meta.get('acuity', 'routine')}")
        c_a = 0.15 if key in self._covered else 0.85
        return FlywheelOverlay(
            c_a=c_a, c_a_thresh=0.6,
            v_G=self.get_version(),
            audit_status=(AuditStatus.COVERED if c_a < 0.6
                          else AuditStatus.INSUFFICIENT_COVERAGE),
        )

    def apply_batch(self, batch: GovernanceBatch) -> bool:
        applied = False
        for lc in batch.local_corrections:
            if lc.correction_type == CorrectionType.AUDIT_COVERAGE_UPDATE:
                cc = lc.payload.get("case_class", "")
                if cc:
                    self._covered.add(cc)
                    applied = True
        if applied:
            self._version += 1
        return True

    def get_version(self) -> str:
        return f"gov:v{self._version}"

    def get_norms(self) -> List[Norm]:
        return list(self._norms)

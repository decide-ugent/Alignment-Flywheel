"""ComplexMedicalOverlay — Flywheel overlay for complex medical demo."""

from typing import Any, List, Optional, Set

from flywheel.protocols.interfaces.base_flywheel_overlay import BaseFlywheelOverlay
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.flywheel_overlay import FlywheelOverlay
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.artifacts.norm import Norm
from flywheel.protocols.enums import AuditStatus, CorrectionType, NormKind


HIGH_RISK_MEDS = ["warfarin", "insulin", "oxycodone", "fentanyl",
                  "tramadol", "prednisone", "amiodarone", "digoxin", "lithium"]


class ComplexMedicalOverlay(BaseFlywheelOverlay):
    """4 norms across 3 norm kinds."""

    def __init__(self):
        self._version = 0
        self._covered: Set[str] = set()
        self._norms = [
            Norm(id="NO_HIGH_RISK_MED_WITHOUT_EVIDENCE",
                 kind=NormKind.KEYWORD_BLOCK,
                 spec={"keywords": HIGH_RISK_MEDS,
                       "require_weak_evidence": True},
                 severity=0.95),
            Norm(id="INTERACTION_SAFETY",
                 kind=NormKind.REGEX,
                 spec={"pairs": [("warfarin", "aspirin"),
                                  ("sertraline", "tramadol"),
                                  ("warfarin", "ibuprofen")]},
                 severity=1.0),
            Norm(id="VULNERABLE_PATIENT_PROTECTION",
                 kind=NormKind.THRESHOLD_RULE,
                 spec={"age_max": 75, "age_min": 18,
                       "risk_conditions": ["renal_impairment", "hepatic_impairment", "pregnancy"],
                       "require_evidence": "supported"},
                 severity=0.85),
            Norm(id="SEVERE_ACTION_EVIDENCE",
                 kind=NormKind.PREDICATE,
                 spec={"if_payload_contains_any": ["prescribe", "increase", "stop", "switch"],
                       "specialties": ["oncology", "psychiatry", "pain_management"],
                       "and_evidence_not": "supported",
                       "then_min_disposition": "clinician_review"},
                 severity=0.85),
            Norm(id="LAB_RESULTS_NEED_CONTEXT",
                 kind=NormKind.PREDICATE,
                 spec={"if_payload_contains_any": ["lab result", "cholesterol",
                                                    "creatinine", "hemoglobin",
                                                    "potassium", "blood sugar",
                                                    "liver function", "a1c"],
                       "and_evidence_not": "supported",
                       "then_min_disposition": "nurse_review"},
                 severity=0.7),
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

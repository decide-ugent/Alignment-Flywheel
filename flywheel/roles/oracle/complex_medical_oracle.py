"""ComplexMedicalOracle — 5-dimensional medical oracle with risk tables."""

from typing import Any, Dict, List, Optional

from flywheel.protocols.interfaces.base_oracle import BaseOracle
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.oracle_raw_output import OracleRawOutput
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.enums import EvidenceStatus, CorrectionType


MEDICATION_RISK = {
    "acetaminophen": 0.1, "ibuprofen": 0.2, "metformin": 0.3,
    "lisinopril": 0.4, "atorvastatin": 0.4, "metoprolol": 0.4,
    "amlodipine": 0.4, "sertraline": 0.5, "gabapentin": 0.5,
    "prednisone": 0.6, "tramadol": 0.7, "warfarin": 0.8,
    "insulin": 0.8, "oxycodone": 0.85, "fentanyl": 0.9,
}
INTERACTION_PAIRS = [
    ("warfarin", "aspirin", 0.8), ("warfarin", "ibuprofen", 0.7),
    ("sertraline", "tramadol", 0.8), ("metformin", "alcohol", 0.5),
    ("lisinopril", "potassium", 0.6),
]
ACTION_SEVERITY = {
    "monitor": 0.05, "follow_up": 0.1, "lifestyle": 0.15,
    "test": 0.2, "refer": 0.3, "prescribe": 0.6,
    "increase": 0.65, "stop": 0.7, "switch": 0.75,
    "emergency": 0.9,
}


class ComplexMedicalOracle(BaseOracle):
    """Multi-dimensional medical oracle with patchable hard-blocks and overrides."""

    def __init__(self):
        self._version = 0
        self._hard_blocks: List[str] = []
        self._threshold_overrides: Dict[str, float] = {}
        self._disposition_overrides: Dict[str, str] = {}

    def predict(
        self,
        context: Context,
        trajectory: Trajectory,
        flags: Optional[Any] = None,
    ) -> OracleRawOutput:
        payload, meta = {}, {}
        if trajectory.steps:
            payload = trajectory.steps[0].payload or {}
            meta = trajectory.steps[0].metadata or {}

        draft = (payload.get("draft_reply") or payload.get("text") or "").lower()
        disposition = payload.get("disposition", "reply_only")
        evidence = meta.get("evidence_status", "unknown")
        age = meta.get("patient_age", context.data.get("patient_age", 45))
        comorb = meta.get("comorbidities", context.data.get("comorbidities", []))
        case_type = meta.get("case_type", context.data.get("case_type", "general"))

        for kw in self._hard_blocks:
            if kw in draft:
                return OracleRawOutput(s=0.0, u=0.05, u_thresh=0.30,
                                       v_O=self.get_version(),
                                       evidence_hooks={"hard_blocked": kw},
                                       evidence_status=self._parse_ev(evidence))

        # Disposition override check
        rank = {"reply_only": 0, "nurse_review": 1,
                "clinician_review": 2, "urgent_escalation": 3}
        ovkey = f"{case_type}|{evidence}"
        if ovkey in self._disposition_overrides:
            min_disp = self._disposition_overrides[ovkey]
            if rank.get(disposition, 0) < rank.get(min_disp, 0):
                return OracleRawOutput(s=0.15, u=0.1, u_thresh=0.30,
                                       v_O=self.get_version(),
                                       evidence_hooks={"disposition_override": min_disp},
                                       evidence_status=self._parse_ev(evidence))

        med_risk = max((r for n, r in MEDICATION_RISK.items() if n in draft), default=0.0)

        interact_risk = 0.0
        for a, b, r in INTERACTION_PAIRS:
            if a in draft and b in draft:
                interact_risk = max(interact_risk, r)

        action_risk = max((r for kw, r in ACTION_SEVERITY.items() if kw in draft), default=0.0)

        patient_risk = 0.0
        if age > 75:
            patient_risk += 0.4
        elif age < 12:
            patient_risk += 0.5
        for comorb_item in comorb:
            if comorb_item in ("renal_impairment", "hepatic_impairment"):
                patient_risk += 0.3
            elif comorb_item in ("pregnancy", "elderly"):
                patient_risk += 0.2
        patient_risk = min(1.0, patient_risk)

        ev_pen = {"supported": 0.0, "insufficient": 0.5,
                  "conflicting": 0.7, "unknown": 0.4,
                  "retracted": 0.9}.get(evidence, 0.4)

        s = max(0.0, 1.0 - (
            0.30 * med_risk + 0.25 * interact_risk +
            0.20 * action_risk + 0.10 * patient_risk + 0.15 * ev_pen
        ))

        u = 0.15
        if evidence in ("insufficient", "unknown"):
            u += 0.3
        elif evidence == "conflicting":
            u += 0.4
        if med_risk > 0.5:
            u += 0.1
        if interact_risk > 0:
            u += 0.15

        u_thresh = self._threshold_overrides.get(meta.get("specialty", "general"), 0.5)

        return OracleRawOutput(s=s, u=min(1.0, u), u_thresh=u_thresh,
                               v_O=self.get_version(),
                               evidence_status=self._parse_ev(evidence))

    def apply_batch(self, batch: GovernanceBatch) -> bool:
        applied = False
        for lc in batch.local_corrections:
            if lc.correction_type == CorrectionType.MEDICAL_HARD_BLOCK:
                kw = lc.payload.get("keyword", "")
                if kw and kw not in self._hard_blocks:
                    self._hard_blocks.append(kw)
                    applied = True
            elif lc.correction_type == CorrectionType.THRESHOLD_ADJUSTMENT:
                # Specialty threshold or disposition override?
                if "specialty" in lc.payload and "threshold" in lc.payload:
                    self._threshold_overrides[lc.payload["specialty"]] = lc.payload["threshold"]
                    applied = True
                elif "key" in lc.payload and "min_disposition" in lc.payload:
                    self._disposition_overrides[lc.payload["key"]] = lc.payload["min_disposition"]
                    applied = True
        if applied:
            self._version += 1
        return True

    def get_version(self) -> str:
        return f"oracle:v{self._version}"

    @staticmethod
    def _parse_ev(s):
        try:
            return EvidenceStatus(s)
        except (ValueError, KeyError):
            return EvidenceStatus.UNKNOWN

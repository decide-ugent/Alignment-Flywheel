"""SimpleMedicalOracle — minimal heuristic oracle for the basic medical demo."""

from typing import Any, List, Optional

from flywheel.protocols.interfaces.base_oracle import BaseOracle
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.oracle_raw_output import OracleRawOutput
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.enums import EvidenceStatus, CorrectionType


MED_STEMS = ["dose", "dosage", "prescri", "discontinu", "taper",
             "tapering", "increase", "decrease"]


class SimpleMedicalOracle(BaseOracle):
    """Three-rule heuristic oracle. Demonstrates the protocol minimally."""

    def __init__(self):
        self._version = 0
        self._hard_blocks: List[str] = []
        self._disposition_overrides: dict = {}

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
        evidence = meta.get("evidence_status", context.data.get("evidence_status", "unknown"))
        case_type = meta.get("case_type", context.data.get("case_type", "general"))

        # Hard block check
        for kw in self._hard_blocks:
            if kw in draft:
                return OracleRawOutput(s=0.0, u=0.05, u_thresh=0.30,
                                       v_O=self.get_version(),
                                       evidence_hooks={"hard_blocked": kw},
                                       evidence_status=self._parse_ev(evidence))

        # Disposition override check (set by THRESHOLD_ADJUSTMENT corrections)
        rank = {"reply_only": 0, "nurse_review": 1,
                "clinician_review": 2, "urgent_escalation": 3}
        key = f"{case_type}|{evidence}"
        if key in self._disposition_overrides:
            min_disp = self._disposition_overrides[key]
            if rank.get(disposition, 0) < rank.get(min_disp, 0):
                return OracleRawOutput(s=0.15, u=0.1, u_thresh=0.30,
                                       v_O=self.get_version(),
                                       evidence_hooks={"disposition_override": min_disp},
                                       evidence_status=self._parse_ev(evidence))

        s = 1.0
        if any(stem in draft for stem in MED_STEMS):
            s -= 0.3
        if case_type == "medication":
            s -= 0.2
        if evidence in ("insufficient", "conflicting"):
            s -= 0.15
        s = max(0.0, s)

        u = 0.15
        if evidence in ("insufficient", "unknown"):
            u += 0.3
        elif evidence == "conflicting":
            u += 0.4
        if case_type == "medication":
            u += 0.1

        u_thresh = 0.30 if case_type == "medication" else 0.50

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
                key = lc.payload.get("key", "")
                disp = lc.payload.get("min_disposition", "")
                if key and disp:
                    self._disposition_overrides[key] = disp
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

"""MedicalViolationDecider — checks medical norms (keyword, predicate, threshold)."""

from typing import Any, Dict

from flywheel.protocols.ooda.decide_step import DecideStep
from flywheel.protocols.enums import VerificationOutcome, NormKind


DISPOSITION_RANK = {
    "reply_only": 0,
    "nurse_review": 1,
    "clinician_review": 2,
    "urgent_escalation": 3,
}


class MedicalViolationDecider(DecideStep):
    """Decide: keyword, predicate, threshold checks against medical norms."""

    def _disp_at_least(self, actual, minimum):
        return DISPOSITION_RANK.get(actual, 0) >= DISPOSITION_RANK.get(minimum, 0)

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        candidate = oriented["candidate"]

        for norm, ctx in oriented["matched"]:
            draft = ctx["draft"]
            evidence = ctx["evidence"]
            disposition = ctx["disposition"]
            payload = ctx.get("payload", {}) if isinstance(ctx.get("payload"), dict) else {}
            patient_msg = payload.get("patient_message", "").lower()

            if norm.kind == NormKind.KEYWORD_BLOCK:
                keywords = norm.spec.get("keywords", [])
                require_weak = norm.spec.get("require_weak_evidence", True)
                weak = evidence not in ("supported",)
                for kw in keywords:
                    if kw in draft and (not require_weak or weak):
                        return {
                            "candidate": candidate,
                            "outcome": VerificationOutcome.VIOLATION,
                            "norm_id": norm.id,
                            "evidence": f"keyword_{kw}_evidence_{evidence}",
                        }

            elif norm.kind == NormKind.PREDICATE:
                payload_kws = norm.spec.get("if_payload_contains_any", [])
                msg_kws = norm.spec.get("if_message_contains_any", [])
                evidence_not = norm.spec.get("and_evidence_not")
                min_disp = norm.spec.get("then_min_disposition")
                if not min_disp:
                    continue

                combined = draft + " " + patient_msg
                payload_match = any(kw in draft for kw in payload_kws) if payload_kws else False
                msg_match = any(kw in combined for kw in msg_kws) if msg_kws else False
                trigger = payload_match or msg_match

                if evidence_not:
                    trigger = trigger and evidence != evidence_not

                if trigger and not self._disp_at_least(disposition, min_disp):
                    return {
                        "candidate": candidate,
                        "outcome": VerificationOutcome.VIOLATION,
                        "norm_id": norm.id,
                        "evidence": f"disposition_{disposition}_below_{min_disp}",
                    }

        return {
            "candidate": candidate,
            "outcome": VerificationOutcome.NO_VIOLATION,
            "norm_id": None,
            "evidence": "",
        }

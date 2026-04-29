"""MedicalNormMatcher — matches medical candidates against Phi norms."""

from typing import Any, Dict

from flywheel.protocols.ooda.orient_step import OrientStep


class MedicalNormMatcher(OrientStep):
    """Orient: extract draft, evidence, disposition fields for verification."""

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        candidate = observation["candidate"]
        norms = observation["norms"]

        steps = candidate.trajectory.get("steps", [])
        payload = steps[0].get("payload", {}) if steps else {}
        meta = steps[0].get("metadata", {}) if steps else {}
        draft = (payload.get("draft_reply", "") or payload.get("text", "")).lower()
        evidence = candidate.context.get(
            "evidence_status",
            meta.get("evidence_status", "unknown"),
        )
        disposition = payload.get(
            "disposition",
            candidate.context.get("proposed_disposition", "reply_only"),
        )

        matched = []
        for norm in norms:
            matched.append((norm, {
                "draft": draft,
                "evidence": evidence,
                "disposition": disposition,
                "payload": payload,
                "meta": meta,
            }))

        return {"candidate": candidate, "matched": matched}

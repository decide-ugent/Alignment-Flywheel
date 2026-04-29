"""MedicalBatchDeployer — assembles medical GovernanceBatch."""

from typing import Any, Dict

from flywheel.protocols.ooda.act_step import ActStep
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class MedicalBatchDeployer(ActStep):
    """Act: build medical GovernanceBatch from accepted corrections."""

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        accepted = decision["accepted"]
        oracle_version = decision["oracle_version"]
        v = int(oracle_version.split("v")[-1]) if "v" in oracle_version else 0

        corrections = []
        for item in accepted:
            corrections.extend(item.get("corrections", []))

        batch = GovernanceBatch(
            from_oracle_version=oracle_version,
            to_oracle_version=f"oracle:v{v + 1}",
            local_corrections=corrections,
            signature="auto-approved",
        )

        return {
            "batch": batch,
            "patched_points": [],
            "patched_count": len(accepted),
            "rejected": decision.get("rejected", 0),
            "shrinks": 0,
        }

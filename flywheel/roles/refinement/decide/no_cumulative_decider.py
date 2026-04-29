"""NoCumulativeDecider — accepts all planned patches (baseline)."""

from typing import Any, Dict

from flywheel.protocols.ooda.decide_step import DecideStep


class NoCumulativeDecider(DecideStep):
    """Decide: accept all planned patches up to max_patches (baseline)."""

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        max_patches = oriented["max_patches"]
        accepted = [
            {**item, "final_bw": item["proposed_bw"]}
            for item in oriented["planned"][:max_patches]
        ]
        return {
            "accepted": accepted,
            "rejected": 0,
            "shrinks": 0,
            "oracle_version": oriented["oracle_version"],
        }

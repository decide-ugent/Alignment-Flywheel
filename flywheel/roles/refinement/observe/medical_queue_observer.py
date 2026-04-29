"""MedicalQueueObserver — reads verified medical flaws from triage."""

from typing import Any, Dict

from flywheel.protocols.ooda.observe_step import ObserveStep


class MedicalQueueObserver(ObserveStep):
    """Observe: read verified items for medical refinement."""

    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        obs = {
            "verified_items": context.get("verified_items", []),
            "oracle_version": context.get("oracle_version", "oracle:v0"),
            "max_patches": context.get("max_patches", 3),
        }
        if "category_filter" in context:
            obs["category_filter"] = context["category_filter"]
        return obs

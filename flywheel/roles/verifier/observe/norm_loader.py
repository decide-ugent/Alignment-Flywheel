"""NormLoader — loads norms and the candidate to verify."""

from typing import Any, Dict

from flywheel.protocols.ooda.observe_step import ObserveStep


class NormLoader(ObserveStep):
    """Observe: load Phi and the candidate being verified."""

    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "candidate": context["candidate"],
            "norms": context.get("norms", []),
            "expert_path": context.get("expert_path"),
            "boundary": context.get("boundary", 0.34),
        }

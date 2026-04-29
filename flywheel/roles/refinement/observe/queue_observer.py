"""QueueObserver — reads verified flaws from the triage queue (spatial)."""

from typing import Any, Dict

from flywheel.protocols.ooda.observe_step import ObserveStep


class QueueObserver(ObserveStep):
    """Observe: read verified items + basin points for regression testing."""

    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "verified_items": context.get("verified_items", []),
            "oracle_version": context.get("oracle_version", "oracle:v0"),
            "basin_points": context.get("basin_points"),
            "max_patches": context.get("max_patches", 200),
            "boundary": context.get("boundary", 0.34),
            "max_basin_loss": context.get("max_basin_loss", 0.10),
            "min_bw": context.get("min_bw", 0.03),
            "max_bw": context.get("max_bw", 0.30),
            "fixed_bw": context.get("fixed_bw", 0.05),
            "safety_floor": context.get("safety_floor", 0.005),
        }

"""FixedBandwidthOrienter — fixed narrow bandwidth (baseline, no collateral)."""

from typing import Any, Dict

from flywheel.protocols.ooda.orient_step import OrientStep


class FixedBandwidthOrienter(OrientStep):
    """Orient: fixed bandwidth for all flaws (baseline)."""

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        items = observation["verified_items"]
        fixed_bw = observation.get("fixed_bw", 0.05)

        planned = []
        for result, candidate in items:
            planned.append({
                "point": candidate.context.get("point", [0, 0, 0]),
                "dist": candidate.context.get("dist_to_path", 0),
                "proposed_bw": fixed_bw,
                "candidate": candidate,
                "result": result,
            })

        return {**observation, "planned": planned}

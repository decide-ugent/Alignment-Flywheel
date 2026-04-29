"""AdaptiveBandwidthOrienter — computes adaptive bandwidth per flaw."""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.orient_step import OrientStep


class AdaptiveBandwidthOrienter(OrientStep):
    """Orient: far flaws get wide kernels, near-boundary flaws get narrow ones."""

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        items = observation["verified_items"]
        boundary = observation["boundary"]
        min_bw = observation["min_bw"]
        max_bw = observation["max_bw"]

        planned = []
        for result, candidate in items:
            dist = candidate.context.get("dist_to_path", 0)
            point = candidate.context.get("point", [0, 0, 0])
            proposed_bw = float(np.clip(
                (dist - boundary) * 0.5, min_bw, max_bw))
            planned.append({
                "point": point,
                "dist": dist,
                "proposed_bw": proposed_bw,
                "candidate": candidate,
                "result": result,
            })

        return {**observation, "planned": planned}

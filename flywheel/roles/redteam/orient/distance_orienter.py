"""DistanceOrienter — computes distance to expert path for each queried point."""

from typing import Any, Dict

import numpy as np
from scipy.spatial.distance import cdist

from flywheel.protocols.ooda.orient_step import OrientStep


class DistanceOrienter(OrientStep):
    """Orient: compute distance to expert path, identify flaw candidates."""

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        points = observation["points"]
        values = observation["values"]
        expert = observation["expert_path"]
        boundary = observation["boundary"]
        safety_floor = observation.get("safety_floor", 0.01)

        dists = cdist(points, expert).min(axis=1)
        mask = (values > safety_floor) & (dists > boundary)

        flaw_indices = np.where(mask)[0]
        return {
            "points": points,
            "values": values,
            "dists": dists,
            "flaw_indices": flaw_indices,
            "oracle_version": observation["oracle_version"],
            "boundary": boundary,
        }

"""GridObserver — queries oracle on a regular 3D grid."""

import itertools
from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.observe_step import ObserveStep


class GridObserver(ObserveStep):
    """Observe: regular grid scan of the 3D input space."""

    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        oracle = context["oracle"]
        resolution = context.get("grid_resolution", 20)
        bounds = context.get("bounds", (-1.0, 1.0))

        axis = np.linspace(bounds[0], bounds[1], resolution)
        grid = np.array(
            list(itertools.product(axis, axis, axis)),
            dtype=np.float32,
        )

        result = oracle.query_points(grid.tolist())
        return {
            "points": grid,
            "values": np.array(result["values"]),
            "oracle_version": result.get("oracle_version", "?"),
            "expert_path": context.get("expert_path"),
            "boundary": context.get("boundary", 0.34),
            "safety_floor": context.get("safety_floor", 0.01),
        }

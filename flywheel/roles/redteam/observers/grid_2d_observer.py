"""Grid2DObserver — queries oracle on a regular 2D grid."""

import itertools
from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.observe_step import ObserveStep


class Grid2DObserver(ObserveStep):
    """Observe: regular grid scan of a 2D input space."""

    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        oracle = context["oracle"]
        resolution = context.get("grid_resolution", 80)
        bounds = context.get("bounds", (-4.0, 4.0))

        axis = np.linspace(bounds[0], bounds[1], resolution)
        grid = np.array(
            list(itertools.product(axis, axis)),
            dtype=np.float32,
        )

        result = oracle.query_points(grid.tolist())
        return {
            "points": grid,
            "values": np.array(result["values"]),
            "oracle_version": result.get("oracle_version", "?"),
            "expert_path": context.get("expert_path"),
            "boundary": context.get("boundary", 0.5),
            "safety_floor": context.get("safety_floor", 0.01),
        }

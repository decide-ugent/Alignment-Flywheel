"""FineSamplingObserver — coarse grid + adaptive fine sampling around prior flaws."""

import itertools
from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.observe_step import ObserveStep


class FineSamplingObserver(ObserveStep):
    """Observe: coarse grid plus dense sampling around previously discovered flaws.

    Intensity adapts as flaws shrink — more samples, tighter radius
    as we close in on the remaining boundary region.
    """

    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        oracle = context["oracle"]
        prior_flaws = context.get("prior_flaws", [])
        resolution = context.get("grid_resolution", 20)
        bounds = context.get("bounds", (-1.0, 1.0))
        iteration = context.get("iteration", 1)
        prev_remaining = context.get("prev_remaining", 9999)

        axis = np.linspace(bounds[0], bounds[1], resolution)
        coarse = np.array(
            list(itertools.product(axis, axis, axis)),
            dtype=np.float32,
        )

        if prior_flaws:
            n_fine = min(2000, max(500, 8000 // max(1, prev_remaining // 50)))
            fine_radius = max(0.12, min(0.35, prev_remaining / 5000))
            rng = np.random.RandomState(42 + iteration * 7)
            centers = prior_flaws[-200:]
            per = max(3, n_fine // len(centers))
            fine = np.vstack([
                np.clip(c + rng.uniform(-fine_radius, fine_radius, (per, 3)),
                        bounds[0], bounds[1])
                for c in centers
            ]).astype(np.float32)
            points = np.vstack([coarse, fine])
        else:
            points = coarse

        result = oracle.query_points(points.tolist())
        return {
            "points": points,
            "values": np.array(result["values"]),
            "oracle_version": result.get("oracle_version", "?"),
            "expert_path": context.get("expert_path"),
            "boundary": context.get("boundary", 0.34),
            "safety_floor": context.get("safety_floor", 0.01),
        }

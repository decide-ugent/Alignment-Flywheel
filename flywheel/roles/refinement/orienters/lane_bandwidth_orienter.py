"""LaneBandwidthOrienter — computes directional patch parameters.

For each verified lane-discipline flaw, computes:
  - spatial bandwidth (how wide the suppression Gaussian is)
  - suppression strength (soft, based on distance from ideal 0.75)
  - directional sigma (how selective the velocity gate is)

The result is a list of planned directional patches ready for the
regression decider.
"""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.orient_step import OrientStep


class LaneBandwidthOrienter(OrientStep):
    """Orient: compute directional patch parameters from lane position.

    Each flaw has a `lane_position` in [0, 1] and a `patch_strength`
    computed by the observer.  This orienter translates those into
    concrete patch parameters:

      - spatial bw:  controls how wide the Gaussian kernel is
      - strength:    how much to suppress (0 → no change, 1 → full kill)
                     Scaled by `strength_scale` to keep patches soft.
      - dir_sigma:   directional selectivity (lower = more selective)
    """

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        items = observation["verified_items"]
        spatial_bw = observation.get("spatial_bw", 0.12)
        strength_scale = observation.get("strength_scale", 0.85)
        dir_sigma = observation.get("dir_sigma", 0.35)

        planned = []
        for result, candidate in items:
            ctx = candidate.context
            point = ctx.get("point", [0, 0])
            velocity = ctx.get("velocity", [0, 0])
            travel_dir = ctx.get("travel_direction", [0, 0])
            lane_pos = ctx.get("lane_position", 0.5)
            raw_strength = ctx.get("patch_strength", 0.5)
            cell_center = ctx.get("cell_center")
            lane_normal = ctx.get("lane_normal")
            cell_radius = ctx.get("cell_radius", 0.5)

            # Scale the strength to keep patches soft
            # raw_strength is 0 at ideal (0.75), up to 1.0 at far edge
            final_strength = float(np.clip(
                raw_strength * strength_scale, 0.0, 1.0))

            planned.append({
                "point": point,
                "velocity": velocity,
                "travel_direction": travel_dir,
                "lane_position": lane_pos,
                "proposed_bw": spatial_bw,
                "proposed_strength": final_strength,
                "dir_sigma": dir_sigma,
                "cell_center": cell_center,
                "lane_normal": lane_normal,
                "cell_radius": cell_radius,
                "candidate": candidate,
                "result": result,
            })

        return {**observation, "planned": planned}

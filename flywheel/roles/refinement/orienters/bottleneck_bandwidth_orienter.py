"""BottleneckBandwidthOrienter — computes bandwidth in the MoE bottleneck space.

Instead of using Euclidean distance in full obs-space (curse of
dimensionality for 111D), we compute bandwidth based on the
distance to the nearest demo point in bottleneck space.

Wide bandwidth → suppress a larger region in latent space.
Narrow bandwidth → surgical correction of a specific flaw.
"""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.orient_step import OrientStep


class BottleneckBandwidthOrienter(OrientStep):
    """Orient: assign bottleneck-space bandwidth per flaw."""

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        items = observation["verified_items"]
        oracle = observation.get("oracle_ref")  # MoELocomotionOracle
        min_bw = observation.get("min_bw", 0.3)
        max_bw = observation.get("max_bw", 3.0)

        planned = []
        for result, candidate in items:
            point = candidate.context.get("point")
            oob_ratio = candidate.context.get("oob_ratio", 0)
            violated_group = candidate.context.get("violated_group")
            safety_val = candidate.s

            if point is None:
                continue

            t = min(oob_ratio, 1.0)
            if observation.get("adaptive", False):
                # Adaptive: severity-proportional
                #   marginal OOB → wide bw (catch boundary), moderate strength
                #   severe OOB   → narrow bw (surgical), high strength
                proposed_bw = float(np.clip(
                    max_bw - (max_bw - min_bw) * t,
                    min_bw, max_bw,
                ))
                strength = float(np.clip(0.5 + 0.45 * t, 0.5, 0.95))
            else:
                # Fixed: higher OOB → wider kernel, low OOB → narrow
                proposed_bw = float(np.clip(
                    min_bw + (max_bw - min_bw) * t,
                    min_bw, max_bw,
                ))
                # Boost strength for high-safety flaws (more dangerous)
                strength = float(np.clip(safety_val * 1.2, 0.5, 1.0))

            planned.append({
                "point": point,
                "oob_ratio": oob_ratio,
                "violated_group": violated_group,
                "safety_val": safety_val,
                "proposed_bw": proposed_bw,
                "strength": strength,
                "candidate": candidate,
                "result": result,
            })

        return {**observation, "planned": planned}

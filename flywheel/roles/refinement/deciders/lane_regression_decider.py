"""LaneRegressionDecider — ensures lane patches are soft, not destructive.

For each planned directional lane patch, verifies:
  1. The suppression strength is below `max_strength` — never fully kill
     reward on any part of the lane.
  2. **Correct-side regression check**: the patch's spatial Gaussian
     must not bleed into correct-side traffic above a budget.  Patches
     near the corridor centre are analytically weakened so their
     worst-case impact at the mirror correct-side point stays within
     ``max_correct_side_impact``.

Patches that exceed thresholds are scaled down, not rejected,
to preserve coverage while staying soft.
"""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.decide_step import DecideStep


class LaneRegressionDecider(DecideStep):
    """Decide: accept directional lane patches with regression guard.

    Two checks per patch:

    1. **Hard cap** — ``max_strength`` (default 0.95).
    2. **Correct-side regression** — for each patch the decider
       computes the analytical worst-case suppression at the mirror
       point on the correct side of the corridor.  The mirror point
       is the closest correct-side location to the patch centre,
       at perpendicular distance ``2 × |lane_pos − 0.5| × cell_width``.
       Because the oracle uses ``max`` aggregation over patches, the
       single-patch budget bounds the total correct-side impact.

       If the mirror impact exceeds ``max_correct_side_impact`` the
       patch strength is reduced until the budget is met.  Patches
       whose resulting strength falls below ``min_useful_strength``
       are dropped entirely.
    """

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        planned = oriented["planned"]
        max_strength = oriented.get("max_strength", 0.95)
        max_patches = oriented.get("max_patches", 100000)
        max_correct_impact = oriented.get("max_correct_side_impact", 0.05)
        cell_width = oriented.get("cell_width", 1.0)
        min_useful = oriented.get("min_useful_strength", 0.01)

        accepted = []
        capped = 0
        regression_capped = 0
        rejected_weak = 0

        for item in planned:
            if len(accepted) >= max_patches:
                break

            strength = item["proposed_strength"]
            bw = item["proposed_bw"]
            lane_pos = item.get("lane_position", 0.5)

            # 1. Hard cap on absolute strength
            if strength > max_strength:
                strength = max_strength
                capped += 1

            # 2. Correct-side regression check
            #    Distance from corridor centre in world units
            dist_from_centre = abs(lane_pos - 0.5) * cell_width
            #    Nearest correct-side point is at the centre line,
            #    i.e. dist_from_centre away from the patch
            nearest_correct_dist = dist_from_centre
            #    Spatial Gaussian contribution at that point
            mirror_spatial = np.exp(
                -nearest_correct_dist ** 2 / (2.0 * bw ** 2))
            #    Worst-case directional gate = 1.0 (same-direction traffic)
            mirror_impact = mirror_spatial * strength

            if mirror_impact > max_correct_impact:
                safe_strength = max_correct_impact / (mirror_spatial + 1e-12)
                strength = min(strength, safe_strength)
                regression_capped += 1

            # Drop patches that are too weak to matter
            if strength < min_useful:
                rejected_weak += 1
                continue

            accepted.append({**item, "final_strength": strength})

        return {
            "accepted": accepted,
            "total_planned": len(planned),
            "capped": capped,
            "regression_capped": regression_capped,
            "rejected_weak": rejected_weak,
            "oracle_version": oriented.get("oracle_version", "?"),
        }

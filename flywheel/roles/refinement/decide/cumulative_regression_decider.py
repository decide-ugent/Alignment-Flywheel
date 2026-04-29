"""CumulativeRegressionDecider — tests each patch against cumulative basin impact."""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.decide_step import DecideStep


class CumulativeRegressionDecider(DecideStep):
    """Decide: cumulative regression test.

    Tracks total suppression across all accepted patches in the
    batch. Shrinks bandwidth or rejects patches that would push
    cumulative basin impact above MAX_BASIN_LOSS.
    """

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        planned = oriented["planned"]
        basin_pts = oriented.get("basin_points")
        max_patches = oriented["max_patches"]
        max_loss = oriented["max_basin_loss"]
        min_bw = oriented["min_bw"]

        accepted = []
        rejected = 0
        shrinks = 0
        batch_kernels = []

        for item in planned:
            if len(accepted) >= max_patches:
                break

            point = np.array(item["point"], dtype=np.float32)
            bw = item["proposed_bw"]

            if basin_pts is not None and len(basin_pts) > 0:
                basin_dists = np.sqrt(np.sum(
                    (basin_pts - point) ** 2, axis=1))
                nearest = basin_dists.min()

                max_safe = nearest / np.sqrt(-2 * np.log(max_loss))
                if bw > max_safe * 0.9:
                    bw = max_safe * 0.9
                    shrinks += 1

                if bw < min_bw:
                    bw = min_bw
                    new_supp = np.exp(-basin_dists ** 2 / (2 * bw ** 2))
                    existing = self._cumulative(basin_pts, batch_kernels)
                    if (existing + new_supp).max() > 0.3:
                        rejected += 1
                        continue

                new_supp = np.exp(-basin_dists ** 2 / (2 * bw ** 2))
                existing = self._cumulative(basin_pts, batch_kernels)
                total = existing + new_supp

                if total.max() > max_loss:
                    lo, hi = min_bw, bw
                    for _ in range(10):
                        mid = (lo + hi) / 2
                        test = np.exp(-basin_dists ** 2 / (2 * mid ** 2))
                        if (existing + test).max() <= max_loss:
                            lo = mid
                        else:
                            hi = mid
                    bw = lo
                    shrinks += 1
                    if bw < min_bw:
                        rejected += 1
                        continue

            batch_kernels.append((point, bw))
            accepted.append({**item, "final_bw": bw})

        return {
            "accepted": accepted,
            "rejected": rejected,
            "shrinks": shrinks,
            "oracle_version": oriented["oracle_version"],
        }

    @staticmethod
    def _cumulative(basin_pts, kernels):
        if not kernels:
            return np.zeros(len(basin_pts), dtype=np.float32)
        supp = np.zeros(len(basin_pts), dtype=np.float32)
        for center, bw in kernels:
            d2 = np.sum((basin_pts - center) ** 2, axis=1)
            supp += np.exp(-d2 / (2 * bw ** 2))
        return supp

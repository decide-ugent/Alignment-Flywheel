"""BottleneckRegressionDecider — per-expert per-point regression testing.

Each MoE expert has its own bottleneck space.  Patches target a
specific expert, and regression testing is done in THAT expert's
latent space against the demo basin projected through that expert.

The suppression on each demo point is weighted by the gating weight
for that expert, matching how suppression is applied at query time.
"""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.decide_step import DecideStep

GAUSS_FLOOR = 1e-3  # clip suppression below this to zero


class BottleneckRegressionDecider(DecideStep):
    """Decide: per-expert regression test of patches against demo basin."""

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        planned = oriented["planned"]
        max_basin_loss = oriented.get("max_basin_loss", 0.05)
        min_bw = oriented.get("min_bw", 0.3)
        oracle = oriented.get("oracle_ref")

        # Per-expert basin data from oracle
        basin_z_per_expert = None
        basin_gating = None
        if oracle is not None:
            basin_z_per_expert = oracle.demo_z_per_expert  # list of (N,B)
            basin_gating = oracle.demo_gating              # (N, E)

        accepted = []
        rejected = 0
        shrinks = 0
        # Incremental cumulative suppression per expert — avoids
        # recomputing from scratch for each candidate (O(K*N) → O(N)).
        cum_supp = {}   # expert_idx → running (N,) array

        for item in planned:

            point = np.array(item["point"], dtype=np.float32)
            bw = item["proposed_bw"]
            strength = item["strength"]

            if oracle is None:
                accepted.append({**item, "final_bw": bw, "final_strength": strength})
                continue

            # Determine dominant expert for this flaw
            expert_idx = int(oracle.dominant_expert(point)[0])
            z = oracle.encode_expert(point, expert_idx)[0]  # (B,)

            if basin_z_per_expert is not None:
                basin_z_e = basin_z_per_expert[expert_idx]      # (N, B)
                gating_e = basin_gating[:, expert_idx]           # (N,)
                N = len(basin_z_e)

                # Gating-weighted distance in this expert's space
                basin_dists_sq = np.sum((basin_z_e - z) ** 2, axis=1)

                # Gating-weighted suppression on basin (clip negligible tails)
                new_supp = gating_e * strength * np.exp(
                    -basin_dists_sq / (2 * bw ** 2)
                )
                new_supp[new_supp < GAUSS_FLOOR] = 0.0

                # Cumulative suppression (incremental — O(1) lookup)
                existing = cum_supp.get(expert_idx)
                if existing is None:
                    existing = np.zeros(N, dtype=np.float32)
                total = existing + new_supp

                # If max basin impact too high, shrink bandwidth
                if total.max() > max_basin_loss:
                    lo_bw, hi_bw = min_bw, bw
                    for _ in range(15):
                        mid = (lo_bw + hi_bw) / 2
                        test = gating_e * strength * np.exp(
                            -basin_dists_sq / (2 * mid ** 2)
                        )
                        test[test < GAUSS_FLOOR] = 0.0
                        if (existing + test).max() <= max_basin_loss:
                            lo_bw = mid
                        else:
                            hi_bw = mid
                    bw = lo_bw
                    shrinks += 1

                    new_supp = gating_e * strength * np.exp(
                        -basin_dists_sq / (2 * bw ** 2)
                    )
                    new_supp[new_supp < GAUSS_FLOOR] = 0.0
                    if (existing + new_supp).max() > max_basin_loss:
                        # Reduce strength at the worst combined point
                        worst_idx = np.argmax(existing + new_supp)
                        gauss_val = gating_e[worst_idx] * np.exp(
                            -basin_dists_sq[worst_idx] / (2 * bw ** 2)
                        )
                        if gauss_val > 1e-9:
                            headroom = max(0.0, max_basin_loss - existing[worst_idx])
                            strength = float(headroom / gauss_val) if headroom > 0 else 0.0
                            strength = max(0.1, min(1.0, strength))
                        else:
                            rejected += 1
                            continue

                # Update incremental cumulative with accepted patch
                final_supp = gating_e * strength * np.exp(
                    -basin_dists_sq / (2 * bw ** 2)
                )
                final_supp[final_supp < GAUSS_FLOOR] = 0.0
                if expert_idx in cum_supp:
                    cum_supp[expert_idx] += final_supp
                else:
                    cum_supp[expert_idx] = final_supp.copy()

            accepted.append({
                **item,
                "final_bw": bw,
                "final_strength": strength,
                "expert_idx": expert_idx,
                "center_z": z.tolist(),
            })

        return {
            "accepted": accepted,
            "rejected": rejected,
            "shrinks": shrinks,
            "oracle_version": oriented.get("oracle_version", "oracle:v0"),
        }

"""LocomotionObserver — samples observation-space points around demo data.

Generates candidate observations via:
  1. Gaussian perturbation at multiple scales (0.25σ – 2.0σ)
  2. Boundary probing (push single dims just past demo range)
  3. Correlated group perturbation (scale entire position/velocity group)
  4. Re-sampling around prior flaws (adaptive refinement)
"""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.observe_step import ObserveStep


class LocomotionObserver(ObserveStep):
    """Observe: generate candidate observations for locomotion flaw discovery."""

    def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        oracle = context["oracle"]
        demo_obs = context["demo_obs"]               # (N, D)
        dim_stats = context["dim_stats"]
        dim_groups = context.get("dim_groups", {})
        iteration = context.get("iteration", 1)
        prior_flaws = context.get("prior_flaws", [])
        rng = np.random.RandomState(42 + iteration * 13)

        N, D = demo_obs.shape
        std = dim_stats["std"]
        lo = dim_stats["min"]
        hi = dim_stats["max"]

        samples = []

        # 1. Gaussian perturbation at multiple scales
        for sigma_mult in [0.25, 0.5, 0.75, 1.0, 1.5]:
            n = min(2000, N)
            idx = rng.choice(N, n, replace=True)
            base = demo_obs[idx].copy()
            noise = rng.randn(n, D).astype(np.float32) * std * sigma_mult
            samples.append(base + noise)

        # 2. Boundary probing — push one dim just past bounds
        n_bnd = 1000
        idx = rng.choice(N, n_bnd, replace=True)
        base_bnd = demo_obs[idx].copy()
        for i in range(n_bnd):
            d = rng.randint(0, D)
            rng_d = hi[d] - lo[d] + 1e-9
            eps = rng.uniform(0.01, 0.10) * rng_d
            if rng.rand() > 0.5:
                base_bnd[i, d] = hi[d] + eps
            else:
                base_bnd[i, d] = lo[d] - eps
        samples.append(base_bnd)

        # 3. Correlated group perturbation
        for group_name, dims in dim_groups.items():
            if hasattr(dims, 'numpy'):
                dims = dims.numpy()
            n_grp = 500
            idx = rng.choice(N, n_grp, replace=True)
            for scale in [1.1, 1.2, 1.5]:
                grp = demo_obs[idx].copy()
                grp[:, dims] *= scale
                samples.append(grp)

        # 4. Re-sample around prior flaws
        if prior_flaws:
            centers = np.array(prior_flaws[-500:], dtype=np.float32)
            n_refine = min(2000, len(centers) * 5)
            idx = rng.choice(len(centers), n_refine, replace=True)
            noise = rng.randn(n_refine, D).astype(np.float32) * std * 0.3
            samples.append(centers[idx] + noise)

        points = np.vstack(samples).astype(np.float32)

        # Query oracle
        result = oracle.query_points(points.tolist())
        values = np.array(result["values"], dtype=np.float32)

        return {
            "points": points,
            "values": values,
            "oracle_version": result.get("oracle_version", "?"),
            "demo_obs": demo_obs,
            "dim_stats": dim_stats,
            "dim_groups": dim_groups,
        }

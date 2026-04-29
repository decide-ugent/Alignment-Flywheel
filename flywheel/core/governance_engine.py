"""GovernanceEngine — orchestrates the full Flywheel loop.

Runs in the demo process. Calls oracle/proposer/flywheel/enforcement
through their abstract interfaces — concretely, those are HTTP
clients hitting the local API.
"""

import itertools
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

from flywheel.protocols.ooda.ooda_role import OODARole
from flywheel.protocols.enums import VerificationOutcome
from flywheel.core.knowledge_base.in_memory_knowledge_base import InMemoryKnowledgeBase


class GovernanceEngine:
    """Orchestrates the iterative governance loop."""

    def __init__(
        self,
        redteam: OODARole,
        verifier: OODARole,
        refinement: OODARole,
        oracle_adapter,        # BaseSpatialOracleAdapter (HTTP client) for spatial demos
        oracle,                # BaseOracle (HTTP client) for medical demos
        flywheel_overlay,      # BaseFlywheelOverlay (HTTP client)
        enforcement,           # BaseEnforcementPolicy (HTTP client)
        triage,
        blue_team=None,
        kb: Optional[InMemoryKnowledgeBase] = None,
        expert_path=None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.redteam = redteam
        self.verifier = verifier
        self.refinement = refinement
        self.oracle_adapter = oracle_adapter
        self.oracle = oracle
        self.flywheel = flywheel_overlay
        self.enforcement = enforcement
        self.triage = triage
        self.blue_team = blue_team
        self.kb = kb or InMemoryKnowledgeBase()
        self.expert_path = expert_path
        self.params = params or {}

        self.active_flaws: List = []

    def run_iteration(self, iteration: int, prev_remaining: int = 9999) -> Dict[str, Any]:
        t0 = time.perf_counter()
        boundary = self.params.get("boundary", 0.34)
        safety_floor = self.params.get("safety_floor", 0.01)

        # ── 1. RED TEAM ───────────────────────────────────────
        rt_context = {
            "oracle": self.oracle_adapter,
            "expert_path": self.expert_path,
            "boundary": boundary,
            "safety_floor": safety_floor,
            "prior_flaws": self.active_flaws,
            "iteration": iteration,
            "prev_remaining": prev_remaining,
            **self.redteam.params,
        }
        rt_result = self.redteam.execute(rt_context)
        candidates = rt_result.get("candidates", [])

        # ── 2. VERIFY ─────────────────────────────────────────
        norms = self.flywheel.get_norms()
        for cand in candidates:
            v_context = {
                "candidate": cand,
                "norms": norms,
                "expert_path": self.expert_path,
                "boundary": boundary,
            }
            v_result = self.verifier.execute(v_context)
            result = v_result["result"]
            self.kb.append_candidate(cand)
            if result.outcome == VerificationOutcome.VIOLATION:
                self.kb.append_verification(result)
                self.triage.submit(result, cand)

        # ── 3. REFINEMENT ─────────────────────────────────────
        verified_items = self.triage.pop_all()
        patched_points: List = []
        rejected = 0
        shrinks = 0
        predicted_coverage = 0

        if verified_items:
            basin_pts = self._get_basin_points(boundary, safety_floor) if self.expert_path is not None else None
            ref_context = {
                "verified_items": verified_items,
                "oracle_version": self.oracle_adapter.get_version() if self.oracle_adapter else self.oracle.get_version(),
                "basin_points": basin_pts,
                "safety_floor": safety_floor,
                **self.refinement.params,
            }
            ref_result = self.refinement.execute(ref_context)

            batch = ref_result.get("batch")
            patched_points = ref_result.get("patched_points", [])
            rejected = ref_result.get("rejected", 0)
            shrinks = ref_result.get("shrinks", 0)
            predicted_coverage = ref_result.get("predicted_coverage", 0)

            # ── 4. DEPLOY (two HTTP calls) ───────────────────
            if batch and batch.local_corrections:
                if self.oracle_adapter:
                    self.oracle_adapter.send_patch(batch)
                if self.oracle:
                    self.oracle.apply_batch(batch)
                self.flywheel.apply_batch(batch)
                self.kb.append_batch(batch)

        # ── 5. BLUE TEAM ──────────────────────────────────────
        collateral = 0
        if self.blue_team and candidates:
            patched_set = {tuple(p) for p in patched_points}
            unpatched_pts = [
                c.context["point"] for c in candidates
                if tuple(c.context.get("point", [])) not in patched_set
            ]
            self.active_flaws, collateral = self.blue_team.check_collateral(
                unpatched_pts, safety_floor)
        else:
            self.active_flaws = [
                c.context.get("point") for c in candidates
                if c.context.get("point")
            ]

        elapsed = time.perf_counter() - t0
        oracle_v = self.oracle_adapter.get_version() if self.oracle_adapter else self.oracle.get_version()
        gov_v = self.flywheel.get_version()
        return {
            "iteration": iteration,
            "found": len(candidates),
            "verified": len(verified_items),
            "patched": len(patched_points),
            "patched_points": patched_points,
            "collateral": collateral,
            "rejected": rejected,
            "shrinks": shrinks,
            "predicted_coverage": predicted_coverage,
            "oracle_version": oracle_v,
            "governance_version": gov_v,
            "time": round(elapsed, 3),
        }

    def _get_basin_points(self, boundary, safety_floor):
        if self.expert_path is None or self.oracle_adapter is None:
            return None
        grid_res = self.params.get("grid_resolution", 20)
        bounds = self.params.get("bounds", (-1.0, 1.0))
        axis = np.linspace(bounds[0], bounds[1], grid_res)
        grid = np.array(list(itertools.product(axis, axis, axis)),
                        dtype=np.float32)
        vals = np.array(self.oracle_adapter.query_points(grid.tolist())["values"])
        dists = cdist(grid, self.expert_path).min(axis=1)
        mask = (vals > safety_floor) & (dists <= boundary)
        return grid[mask]

    def count_flaws(self):
        boundary = self.params.get("boundary", 0.34)
        safety_floor = self.params.get("safety_floor", 0.01)
        grid_res = self.params.get("grid_resolution", 20)
        bounds = self.params.get("bounds", (-1.0, 1.0))
        axis = np.linspace(bounds[0], bounds[1], grid_res)
        grid = np.array(list(itertools.product(axis, axis, axis)),
                        dtype=np.float32)
        vals = np.array(self.oracle_adapter.query_points(grid.tolist())["values"])
        dists = cdist(grid, self.expert_path).min(axis=1)
        active = vals > safety_floor
        near = active & (dists <= boundary)
        far = active & (dists > boundary)
        return {"basin": int(near.sum()), "flaws": int(far.sum())}

"""Build PointMaze flywheel kernel and visualize constraint checking.

Steps:
  1. Load PointMaze_demos.pt and the trained MoE
  2. Run the flywheel governance loop to place suppression patches
  3. Export the kernel .pt file
  4. Visualize: trajectories colored by speed, with constraint boundaries
     and flywheel safety overlaid

Usage:
    cd moe-guide
    python build_pointmaze_kernel.py
"""
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root
sys.path.insert(0, ROOT)

from IIRL.models import MixtureOfExperts
from flywheel.factory.registry import FactoryRegistry
from flywheel.protocols.ooda.ooda_role import OODARole
from flywheel.protocols.artifacts.norm import Norm
from flywheel.protocols.enums import NormKind, VerificationOutcome
from flywheel.roles.oracle.moe_locomotion_oracle import MoELocomotionOracle
from flywheel.roles.flywheel_overlay.spatial_overlay import SpatialOverlay
from flywheel.roles.triage.fifo_triage import FIFOTriage
from flywheel.core.knowledge_base.in_memory_knowledge_base import InMemoryKnowledgeBase

# ── Config ──────────────────────────────────────────────────────
ENV_NAME = "PointMaze"
OBS_DIM = 4
BOTTLENECK = 2
NUM_EXPERTS = 2
MODEL_PATH = os.path.join(HERE, "models", "PointMaze_l2_e2.pth")
DATA_PATH = os.path.join(HERE, "data", "PointMaze_demos.pt")

# Estimator config — tuned for PointMaze (very low reconstruction error)
L_MIN = 0.00001
L_MAX = 0.0001
STEEPNESS = 50

DIM_GROUPS = {
    "position": [0, 1],    # x, y
    "velocity": [2, 3],    # vx, vy
}

STRICTNESS = {
    "safety_floor": 0.15,
    "max_basin_loss": 0.0,
    "min_bw": 0.2,
    "max_bw": 2.5,
}
STRICTNESS_LABEL = "tight"
MAX_PATCHES = 400
PATIENCE = 3

# ── Load data ───────────────────────────────────────────────────
print(f"Loading {DATA_PATH}")
demos = torch.load(DATA_PATH, weights_only=False)
obs = demos['observations'].numpy().astype(np.float32)
print(f"  {obs.shape[0]} observations, {obs.shape[1]}D")

# ── Load model ──────────────────────────────────────────────────
model = MixtureOfExperts(OBS_DIM, BOTTLENECK, NUM_EXPERTS)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print(f"  MoE: {OBS_DIM}D -> {BOTTLENECK}B x {NUM_EXPERTS}E")

# ── Dim stats ───────────────────────────────────────────────────
obs_mean = obs.mean(axis=0)
obs_min = obs.min(axis=0)
obs_max = obs.max(axis=0)
dim_stats = {
    "min": torch.from_numpy(obs_min),
    "max": torch.from_numpy(obs_max),
    "mean": torch.from_numpy(obs_mean),
    "std": torch.from_numpy(obs.std(axis=0)),
}
# Asymmetric stds
pos_std = np.zeros_like(obs_mean)
neg_std = np.zeros_like(obs_mean)
for d in range(obs.shape[1]):
    above = obs[obs[:, d] > obs_mean[d], d]
    below = obs[obs[:, d] <= obs_mean[d], d]
    pos_std[d] = above.std() if len(above) > 1 else obs[:, d].std()
    neg_std[d] = below.std() if len(below) > 1 else obs[:, d].std()
dim_stats["pos_std"] = torch.from_numpy(pos_std.astype(np.float32))
dim_stats["neg_std"] = torch.from_numpy(neg_std.astype(np.float32))

dim_groups = {
    name: torch.tensor(dims, dtype=torch.long)
    for name, dims in DIM_GROUPS.items()
}
estimator_config = {"l_min": L_MIN, "l_max": L_MAX, "steepness": STEEPNESS}

# ── Build oracle ────────────────────────────────────────────────
oracle = MoELocomotionOracle(
    model=model,
    estimator_config=estimator_config,
    dim_stats=dim_stats,
    dim_groups=dim_groups,
    demo_obs=obs,
    env_name=ENV_NAME,
    skip_dbscan=True,
)
print(f"  Oracle ready, {oracle.bottleneck_dim}B, {oracle.num_experts}E")

# ── Build OODA roles ────────────────────────────────────────────
factory = FactoryRegistry()
factory.auto_register()

redteam = OODARole(
    observe=factory.create("LocomotionObserver"),
    orient=factory.create("LocomotionOrienter"),
    decide=factory.create("HighestCombinedDecider"),
    act=factory.create("LocomotionCandidateSubmitter"),
    params={
        "demo_obs": obs,
        "dim_stats": {k: v.numpy() for k, v in dim_stats.items()},
        "dim_groups": dim_groups,
    },
)

oob_norm = Norm(
    id="OBS_DIM_BOUNDS",
    kind=NormKind.SPATIAL_BOUNDARY,
    spec={"require_within_demo_range": True},
    severity=1.0,
    description="Observations must stay within demo dim ranges",
)

verifier = OODARole(
    observe=factory.create("LocomotionNormLoader"),
    orient=factory.create("LocomotionNormMatcher"),
    decide=factory.create("LocomotionViolationDecider"),
    act=factory.create("VerificationEmitter"),
    params={
        "dim_stats": {k: v.numpy() for k, v in dim_stats.items()},
        "dim_groups": dim_groups,
    },
)

refinement = OODARole(
    observe=factory.create("LocomotionQueueObserver"),
    orient=factory.create("BottleneckBandwidthOrienter"),
    decide=factory.create("BottleneckRegressionDecider"),
    act=factory.create("LocomotionBatchDeployer"),
    params={
        "max_basin_loss": STRICTNESS["max_basin_loss"],
        "min_bw": STRICTNESS["min_bw"],
        "max_bw": STRICTNESS["max_bw"],
        "oracle_ref": oracle,
    },
)

overlay = SpatialOverlay(norms=[oob_norm])
triage = FIFOTriage()
kb = InMemoryKnowledgeBase()

# ── Run governance loop ─────────────────────────────────────────
print(f"\nRunning flywheel (patience={PATIENCE}, max_patches={MAX_PATCHES})")
print(f"  {'It':>3}  {'Found':>6}  {'Verified':>8}  {'Patched':>7}  "
      f"{'Rejected':>8}  {'Total':>6}")
print("  " + "-" * 55)

best_found = 0
stale_count = 0
active_flaws = []
t0 = time.perf_counter()

for it in range(1, 101):
    # RED TEAM
    rt_context = {
        "oracle": oracle,
        "demo_obs": obs,
        "dim_stats": {k: v.numpy() for k, v in dim_stats.items()},
        "dim_groups": dim_groups,
        "iteration": it,
        "prior_flaws": active_flaws,
        "safety_floor": STRICTNESS["safety_floor"],
    }
    rt_result = redteam.execute(rt_context)
    candidates = rt_result.get("candidates", [])

    # VERIFY
    norms = overlay.get_norms()
    for cand in candidates:
        v_context = {
            "candidate": cand,
            "norms": norms,
            "dim_stats": {k: v.numpy() for k, v in dim_stats.items()},
            "dim_groups": dim_groups,
        }
        v_result = verifier.execute(v_context)
        result = v_result["result"]
        kb.append_candidate(cand)
        if result.outcome == VerificationOutcome.VIOLATION:
            kb.append_verification(result)
            triage.submit(result, cand)

    # REFINEMENT
    verified_items = triage.pop_all()
    patched_points = []
    rejected = 0
    if verified_items:
        ref_context = {
            "verified_items": verified_items,
            "oracle_version": oracle.get_version(),
            "oracle_ref": oracle,
            "max_basin_loss": STRICTNESS["max_basin_loss"],
            "min_bw": STRICTNESS["min_bw"],
            "max_bw": STRICTNESS["max_bw"],
        }
        ref_result = refinement.execute(ref_context)
        batch = ref_result.get("batch")
        patched_points = ref_result.get("patched_points", [])
        rejected = ref_result.get("rejected", 0)

        if batch and batch.local_corrections:
            # Enforce patch cap
            if MAX_PATCHES:
                budget = MAX_PATCHES - oracle.patch_count
                if budget <= 0:
                    batch.local_corrections = []
                elif len(batch.local_corrections) > budget:
                    batch.local_corrections = batch.local_corrections[:budget]
                    patched_points = patched_points[:budget]
            oracle.send_patch(batch)
            overlay.apply_batch(batch)
            kb.append_batch(batch)

    print(f"  {it:>3}  {len(candidates):>6}  {len(verified_items):>8}  "
          f"{len(patched_points):>7}  "
          f"{rejected:>8}  {oracle.patch_count:>6}")

    # Early stopping
    found = len(candidates)
    if it == 1:
        threshold = max(1, int(found * 0.02))
    if found > best_found + threshold:
        best_found = found
        stale_count = 0
    else:
        stale_count += 1

    if MAX_PATCHES and oracle.patch_count >= MAX_PATCHES:
        print(f"\n  Patch cap reached: {oracle.patch_count}")
        break
    if stale_count >= PATIENCE:
        print(f"\n  Early stop: plateaued for {PATIENCE} iterations")
        break

elapsed = time.perf_counter() - t0
print(f"\n  Done in {elapsed:.1f}s — {oracle.patch_count} patches")

# ── Export kernel ───────────────────────────────────────────────
kernel = {
    "moe_state_dict": model.state_dict(),
    "moe_config": {
        "input_dim": OBS_DIM,
        "bottleneck_dim": BOTTLENECK,
        "num_experts": NUM_EXPERTS,
    },
    "estimator_config": estimator_config,
    "dim_stats": {k: v.clone() for k, v in dim_stats.items()},
    "dim_groups": {k: v.clone() for k, v in dim_groups.items()},
    "flywheel_patches": oracle.export_patches(),
    "strictness": {"label": STRICTNESS_LABEL, **STRICTNESS},
    "decider": "regression",
    "max_patches": MAX_PATCHES,
    "n_demo_points": len(obs),
    "env_name": ENV_NAME,
}

out_path = os.path.join(HERE, "kernels",
                        f"PointMaze_flywheel_{STRICTNESS_LABEL}_regression_cap{MAX_PATCHES}.pt")
os.makedirs(os.path.join(HERE, "kernels"), exist_ok=True)
torch.save(kernel, out_path)
print(f"  Saved: {out_path}")

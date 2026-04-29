"""Spatial 3D demo runner.

Starts the Flask API in a thread, builds HTTP clients, runs the
governance loop with predictive coverage planning + cumulative
regression testing, generates plots and convergence CSV.

Usage:
    python -m flywheel.demos.spatial_3d.run --port 5000 \\
        --loss-data /path/to/loss_values.npy
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flywheel.factory.registry import FactoryRegistry
from flywheel.api.app import start_api_in_thread
from flywheel.api.clients.http_spatial_oracle_client import HTTPSpatialOracleClient
from flywheel.api.clients.http_proposer_client import HTTPProposerClient
from flywheel.api.clients.http_flywheel_client import HTTPFlywheelClient
from flywheel.api.clients.http_enforcement_client import HTTPEnforcementClient
from flywheel.core.governance_engine import GovernanceEngine
from flywheel.roles.oracle.spatial_oracle import SpatialOracle
from flywheel.roles.blueteam.collateral_monitor import CollateralMonitor
from flywheel.demos._shared import gen_expert_path


def plot_iteration(oracle_client, expert_path, patched_pts, iteration,
                   stats, boundary, safety_floor, outdir):
    import itertools
    from scipy.spatial.distance import cdist

    axis = np.linspace(-1, 1, 20)
    grid = np.array(list(itertools.product(axis, axis, axis)), dtype=np.float32)
    vals = np.array(oracle_client.query_points(grid.tolist())["values"])
    active = vals > safety_floor
    apt = grid[active]
    aval = vals[active]

    if len(apt) > 0:
        d = cdist(apt, expert_path).min(axis=1)
        near = d <= boundary
        far = ~near
    else:
        near = np.array([], dtype=bool)
        far = near

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    sc = None
    if near.sum() > 0:
        sc = ax.scatter(apt[near, 0], apt[near, 1], apt[near, 2],
                        c=aval[near], cmap='viridis', s=8, alpha=0.7,
                        vmin=0, vmax=1, label='Reward heatmap')
    if far.sum() > 0:
        sc = ax.scatter(apt[far, 0], apt[far, 1], apt[far, 2],
                        c=aval[far], cmap='viridis', s=8, alpha=0.7,
                        vmin=0, vmax=1)
    if patched_pts:
        fp = np.array(patched_pts)
        ax.scatter(fp[:, 0], fp[:, 1], fp[:, 2], c='red', s=25,
                   marker='x', alpha=0.9, label=f'Patched')
    ax.plot(expert_path[:, 0], expert_path[:, 1], expert_path[:, 2],
            'b-', lw=2.5, label='Expert')
    ax.set_title(f'Iteration {iteration}\n', fontsize=15)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(fontsize=10, loc='upper left')
    ax.view_init(45, 300)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.75)
        cbar.set_label("Reward")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"iteration_{iteration}.png"), dpi=150)
    plt.close()
    return int(near.sum()), int(far.sum())


def main():
    parser = argparse.ArgumentParser(description="Spatial 3D Flywheel demo")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--loss-data", default=None,
                        help="Path to loss_values.npy")
    parser.add_argument("--output", default="outputs/spatial_3d")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg_path = args.config or os.path.join(
        os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    demo_cfg = config["demo"]
    os.makedirs(args.output, exist_ok=True)

    # ── Locate loss data ─────────────────────────────────────
    loss_path = args.loss_data
    if loss_path is None:
        for p in ["loss_values.npy", "/mnt/user-data/uploads/loss_values.npy"]:
            if os.path.exists(p):
                loss_path = p
                break
    if not loss_path or not os.path.exists(loss_path):
        raise FileNotFoundError(
            "Need --loss-data path or loss_values.npy in CWD.")

    # ── Build SERVER-SIDE components and start API ──────────
    factory = FactoryRegistry()
    factory.auto_register()

    loss = np.load(loss_path)
    from flywheel.roles.oracle.adapters.precomputed_grid_oracle import PrecomputedGridOracle
    grid_oracle = PrecomputedGridOracle(
        loss_values=loss,
        grid_resolution=demo_cfg["grid_resolution"],
        loss_cap=demo_cfg["loss_cap"],
    )
    spatial_oracle = SpatialOracle(adapter=grid_oracle)

    proposer = factory.create(config["proposer"]["class"])
    flywheel_overlay = factory.create(config["flywheel_overlay"]["class"])
    enforcement = factory.create(
        config["enforcement"]["class"],
        **config["enforcement"].get("params", {}),
    )

    server_components = {
        "oracle": grid_oracle,
        "oracle_adapter": grid_oracle,
        "proposer": proposer,
        "flywheel_overlay": flywheel_overlay,
        "enforcement": enforcement,
    }
    print(f"Starting Flask API on :{args.port}...")
    start_api_in_thread(server_components, port=args.port)
    print("API ready.")

    # ── Build CLIENT-SIDE HTTP clients ───────────────────────
    base_url = f"http://127.0.0.1:{args.port}"
    oracle_client = HTTPSpatialOracleClient(base_url)
    flywheel_client = HTTPFlywheelClient(base_url)
    enforcement_client = HTTPEnforcementClient(base_url)
    proposer_client = HTTPProposerClient(base_url)

    # ── Build governance-side OODA roles ─────────────────────
    redteam = factory.build_ooda_role(config["redteam"])
    verifier = factory.build_ooda_role(config["verifier"])
    refinement = factory.build_ooda_role(config["refinement"])
    triage = factory.create(config["triage"]["class"])
    blue_team = CollateralMonitor(oracle_client)

    expert_path = gen_expert_path()
    boundary = demo_cfg["boundary"]
    safety_floor = demo_cfg["safety_floor"]

    engine = GovernanceEngine(
        redteam=redteam, verifier=verifier, refinement=refinement,
        oracle_adapter=oracle_client, oracle=None,
        flywheel_overlay=flywheel_client,
        enforcement=enforcement_client,
        triage=triage, blue_team=blue_team,
        expert_path=expert_path,
        params={
            "boundary": boundary,
            "safety_floor": safety_floor,
            "grid_resolution": demo_cfg["grid_resolution"],
            "bounds": tuple(demo_cfg["bounds"]),
        },
    )

    # ── Run ──────────────────────────────────────────────────
    print("=" * 78)
    print(f"ALIGNMENT FLYWHEEL — {demo_cfg['name']}")
    print("=" * 78)
    print(f"Loss data: {loss_path} | Boundary: {boundary} | "
          f"Max patches: {refinement.params.get('max_patches')}")
    print()

    initial = engine.count_flaws()
    basin_pre = initial["basin"]
    print(f"Initial: basin={basin_pre} | flaws={initial['flaws']}")
    print()

    hdr = (f"{'It':>3}  {'Found':>6}  {'Kern':>5}  {'Predicted':>9}  "
           f"{'Reject':>6}  {'Basin':>6}  {'Flaws':>6}  {'Oracle':>12}")
    print(hdr); print("-" * len(hdr))

    data = []
    t0 = time.perf_counter()
    prev_remaining = 9999
    for it in range(1, demo_cfg["num_iterations"] + 1):
        result = engine.run_iteration(it, prev_remaining)
        counts = engine.count_flaws()
        prev_remaining = counts["flaws"]

        plot_iteration(oracle_client, expert_path,
                       result.get("patched_points", []),
                       it, result, boundary, safety_floor,
                       args.output)

        print(f"{it:>3}  {result['found']:>6}  {result['patched']:>5}  "
              f"{result.get('predicted_coverage', 0):>9}  "
              f"{result['rejected']:>6}  "
              f"{counts['basin']:>6}  {counts['flaws']:>6}  "
              f"{result['oracle_version']:>12}")

        data.append({
            "iteration": it,
            "found": result["found"],
            "kernels": result["patched"],
            "predicted_coverage": result.get("predicted_coverage", 0),
            "rejected": result["rejected"],
            "basin": counts["basin"],
            "flaws": counts["flaws"],
            "oracle_version": result["oracle_version"],
        })

        if counts["flaws"] == 0:
            print(f"\n  ✓ Converged at iteration {it}.")
            break

    total = time.perf_counter() - t0
    total_kernels = sum(d["kernels"] for d in data)
    total_predicted = sum(d["predicted_coverage"] for d in data)
    total_rejected = sum(d["rejected"] for d in data)

    print(f"\n{'=' * 78}")
    print(f"Time: {total:.1f}s | Kernels placed: {total_kernels} | "
          f"Predicted coverage: {total_predicted} | "
          f"Rejected: {total_rejected}")
    print(f"Basin: {basin_pre} → {data[-1]['basin']} "
          f"({data[-1]['basin'] / basin_pre * 100:.0f}% preserved)")

    with open(os.path.join(args.output, "convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "iteration", "found", "kernels", "predicted_coverage",
            "rejected", "basin", "flaws", "oracle_version"])
        w.writeheader()
        for d in data:
            w.writerow({k: d[k] for k in w.fieldnames})

    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump({"demo": demo_cfg, "iterations": data,
                   "total_time": round(total, 2)}, f, indent=2, default=str)

    print(f"\nOutput: {args.output}/")


if __name__ == "__main__":
    main()

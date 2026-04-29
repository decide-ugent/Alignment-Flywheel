"""Shared medical demo runner.

Each medical demo:
  1. Loads YAML config and a fixed evaluation case set.
  2. Starts the Flask API with that demo's oracle/overlay/enforcement.
  3. Builds HTTP clients + governance OODA roles.
  4. For each governance iteration:
       a. Evaluates all fixed cases via the API (allow/block/escalate).
       b. Runs the governance cycle (Red Team → Verify → Refine → Apply).
  5. Logs allow/block/escalate counts per iteration to CSV + JSON.

Usage from a demo's run.py:
    from flywheel.demos._medical_runner import run_medical_demo
    run_medical_demo(config_path, fixed_cases, output_dir, port)
"""

import csv
import json
import os
import time
from typing import Any, Dict, List

import yaml

from flywheel.factory.registry import FactoryRegistry
from flywheel.api.app import start_api_in_thread
from flywheel.api.clients.http_oracle_client import HTTPOracleClient
from flywheel.api.clients.http_proposer_client import HTTPProposerClient
from flywheel.api.clients.http_flywheel_client import HTTPFlywheelClient
from flywheel.api.clients.http_enforcement_client import HTTPEnforcementClient

from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.trajectory_step import TrajectoryStep
from flywheel.protocols.enums import TrajectoryKind, VerificationOutcome
from flywheel.core.knowledge_base.in_memory_knowledge_base import InMemoryKnowledgeBase
from flywheel.core.query_merger.default_query_merger import DefaultQueryMerger


def _evaluate_case(case, oracle_client, flywheel_client, enforcement_client, merger):
    """Evaluate a single fixed case via the live API."""
    ctx = Context(data=case)
    traj = Trajectory(
        kind=TrajectoryKind.MESSAGE,
        steps=[TrajectoryStep(
            payload={
                "draft_reply": case.get("draft_reply", ""),
                "disposition": case.get("proposed_disposition", "reply_only"),
                "patient_message": case.get("patient_message", ""),
            },
            metadata={
                "case_type": case.get("case_type"),
                "evidence_status": case.get("evidence_status"),
                "acuity": case.get("acuity", "routine"),
                "patient_age": case.get("patient_age", 45),
                "specialty": case.get("specialty"),
            },
        )],
    )
    o_out = oracle_client.predict(ctx, traj)
    f_out = flywheel_client.overlay(ctx, traj)
    unified = merger.merge(o_out, f_out)
    decision = enforcement_client.decide(unified)
    return decision


def run_medical_demo(
    config_path: str,
    fixed_cases: List[Dict[str, Any]],
    output_dir: str,
    port: int,
    title: str = "Medical Demo",
):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    demo_cfg = config["demo"]
    os.makedirs(output_dir, exist_ok=True)

    factory = FactoryRegistry()
    factory.auto_register()

    # ── Build server-side components ─────────────────────────
    oracle = factory.create(config["oracle"]["class"],
                            **config["oracle"].get("params", {}))
    proposer = factory.create(config["proposer"]["class"],
                              **config["proposer"].get("params", {}))
    flywheel_overlay = factory.create(config["flywheel_overlay"]["class"],
                                       **config["flywheel_overlay"].get("params", {}))
    enforcement = factory.create(config["enforcement"]["class"],
                                  **config["enforcement"].get("params", {}))

    server_components = {
        "oracle": oracle,
        "oracle_adapter": None,
        "proposer": proposer,
        "flywheel_overlay": flywheel_overlay,
        "enforcement": enforcement,
    }
    print(f"Starting Flask API on :{port}...")
    start_api_in_thread(server_components, port=port)
    print("API ready.")

    # ── Build client-side HTTP clients ───────────────────────
    base_url = f"http://127.0.0.1:{port}"
    oracle_client = HTTPOracleClient(base_url)
    proposer_client = HTTPProposerClient(base_url)
    flywheel_client = HTTPFlywheelClient(base_url)
    enforcement_client = HTTPEnforcementClient(base_url)

    # ── Build governance-side OODA roles ─────────────────────
    redteam = factory.build_ooda_role(config["redteam"])
    verifier = factory.build_ooda_role(config["verifier"])
    refinement = factory.build_ooda_role(config["refinement"])
    triage = factory.create(config["triage"]["class"])

    kb = InMemoryKnowledgeBase()
    merger = DefaultQueryMerger()

    # ── Run loop ─────────────────────────────────────────────
    print("=" * 76)
    print(f"ALIGNMENT FLYWHEEL — {title}")
    print("=" * 76)
    print(f"Eval cases: {len(fixed_cases)} | Iterations: {demo_cfg['num_iterations']}")
    print()

    hdr = (f"{'It':>3}  {'Allow':>5}  {'Block':>5}  {'Esc':>5}  "
           f"{'Esc%':>5}  {'RT':>4}  {'Viol':>5}  {'Oracle':>12}")
    print(hdr); print("-" * len(hdr))

    unsafe_total = sum(1 for c in fixed_cases if c.get("category") == "unsafe")
    safe_total = sum(1 for c in fixed_cases if c.get("category") == "safe")

    data = []
    t0 = time.perf_counter()

    for it in range(1, demo_cfg["num_iterations"] + 1):
        # ── EVALUATE first — show the oracle's current state ─
        counts = {"allow": 0, "block": 0, "escalate": 0, "abstain": 0}
        for case in fixed_cases:
            decision = _evaluate_case(case, oracle_client, flywheel_client,
                                       enforcement_client, merger)
            counts[decision.action.value] = counts.get(decision.action.value, 0) + 1

        total = sum(counts.values())
        esc_rate = counts["escalate"] / total if total > 0 else 0
        eval_version = oracle_client.get_version()

        # ── GOVERN — Red Team → Verify → Refine → Apply ─────
        rt_result = redteam.execute(redteam.params)
        candidates = rt_result.get("candidates", [])

        norms = flywheel_client.get_norms()
        for cand in candidates:
            v_result = verifier.execute({"candidate": cand, "norms": norms})
            if v_result["result"].outcome == VerificationOutcome.VIOLATION:
                triage.submit(v_result["result"], cand)

        verified_items = triage.pop_all()

        if verified_items:
            ref_result = refinement.execute({
                "verified_items": verified_items,
                "oracle_version": oracle_client.get_version(),
                **refinement.params,
            })
            batch = ref_result.get("batch")
            if batch and batch.local_corrections:
                oracle_client.apply_batch(batch)
                flywheel_client.apply_batch(batch)
                kb.append_batch(batch)

        print(f"{it:>3}  {counts['allow']:>5}  {counts['block']:>5}  "
              f"{counts['escalate']:>5}  {esc_rate:>4.0%}  "
              f"{len(candidates):>4}  {len(verified_items):>5}  "
              f"{eval_version:>12}")

        data.append({
            "iteration": it, "allow": counts["allow"],
            "block": counts["block"], "escalate": counts["escalate"],
            "escalation_rate": round(esc_rate, 3),
            "rt_candidates": len(candidates),
            "verified": len(verified_items),
            "oracle_version": eval_version,
        })

        if counts["escalate"] == 0 and counts["allow"] > 0:
            if counts["block"] >= unsafe_total and counts["allow"] >= safe_total:
                print(f"\n  ✓ Converged at iteration {it} "
                      f"(allow={counts['allow']}, block={counts['block']}, escalate=0).")
                break

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 76}\nTime: {elapsed:.1f}s")

    with open(os.path.join(output_dir, "convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        for d in data:
            w.writerow(d)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({"demo": demo_cfg, "iterations": data,
                   "total_time": round(elapsed, 2)}, f, indent=2, default=str)

    print(f"Output: {output_dir}/")

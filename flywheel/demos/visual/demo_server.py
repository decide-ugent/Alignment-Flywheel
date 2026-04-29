"""Visual demo server — Flask API + static React frontend.

Manages multiple demo instances (simple/complex/portal) with
step-by-step iteration control.  The React app calls these
endpoints to visualise fixed-case decisions and patches.
"""

import os
import sys
import argparse

import yaml
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from flywheel.factory.registry import FactoryRegistry
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.trajectory_step import TrajectoryStep
from flywheel.protocols.enums import (
    TrajectoryKind, VerificationOutcome, CorrectionType,
)
from flywheel.core.query_merger.default_query_merger import DefaultQueryMerger

# ── Import fixed cases ───────────────────────────────────────
from flywheel.demos.simple_medical.fixed_cases import SIMPLE_CASES
from flywheel.demos.complex_medical.fixed_cases import COMPLEX_CASES
from flywheel.demos.patient_portal.fixed_cases import PORTAL_CASES

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
FRONTEND_DIR = os.path.join(ROOT, "frontend", "dist")

DEMO_CONFIGS = {
    "simple_medical": {
        "config": os.path.join(PROJECT_ROOT, "flywheel", "demos", "simple_medical", "config.yaml"),
        "cases": SIMPLE_CASES,
        "title": "Simple Medical",
    },
    "complex_medical": {
        "config": os.path.join(PROJECT_ROOT, "flywheel", "demos", "complex_medical", "config.yaml"),
        "cases": COMPLEX_CASES,
        "title": "Complex Medical",
    },
    "patient_portal": {
        "config": os.path.join(PROJECT_ROOT, "flywheel", "demos", "patient_portal", "config.yaml"),
        "cases": PORTAL_CASES,
        "title": "Patient Portal",
    },
}

# ── Per-demo runtime state ───────────────────────────────────

class DemoInstance:
    """Holds all components and history for one demo."""

    def __init__(self, name, config_path, fixed_cases, title):
        self.name = name
        self.title = title
        self.fixed_cases = fixed_cases
        self.iteration = 0
        self.history = []        # [{iteration, cases: [...], patches: [...]}]

        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config
        self.max_iterations = config["demo"]["num_iterations"]

        factory = FactoryRegistry()
        factory.auto_register()

        self.oracle = factory.create(config["oracle"]["class"],
                                     **config["oracle"].get("params", {}))
        self.proposer = factory.create(config["proposer"]["class"],
                                       **config["proposer"].get("params", {}))
        self.flywheel_overlay = factory.create(
            config["flywheel_overlay"]["class"],
            **config["flywheel_overlay"].get("params", {}))
        self.enforcement = factory.create(
            config["enforcement"]["class"],
            **config["enforcement"].get("params", {}))

        self.redteam = factory.build_ooda_role(config["redteam"])
        self.verifier = factory.build_ooda_role(config["verifier"])
        self.refinement = factory.build_ooda_role(config["refinement"])
        self.triage = factory.create(config["triage"]["class"])

        self.merger = DefaultQueryMerger()

    def evaluate_cases(self, cases=None):
        """Evaluate cases through the full pipeline, return per-case results."""
        if cases is None:
            cases = self.fixed_cases
        results = []
        for case in cases:
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
            o_out = self.oracle.predict(ctx, traj)
            f_out = self.flywheel_overlay.overlay(ctx, traj)
            unified = self.merger.merge(o_out, f_out)
            decision = self.enforcement.decide(unified)

            results.append({
                "id": case.get("id"),
                "patient_message": case.get("patient_message", ""),
                "draft_reply": case.get("draft_reply", ""),
                "case_type": case.get("case_type", ""),
                "evidence_status": case.get("evidence_status", ""),
                "category": case.get("category", ""),
                "proposed_disposition": case.get("proposed_disposition", ""),
                "action": decision.action.value,
                "reasons": decision.reasons,
                "scores": {
                    "s": round(unified.s, 3),
                    "u": round(unified.u, 3),
                    "u_thresh": round(unified.u_thresh, 3),
                    "c_a": round(unified.c_a, 3),
                    "c_a_thresh": round(unified.c_a_thresh, 3),
                },
            })
        return results

    def run_governance(self):
        """Run one governance cycle: RedTeam → Verify → Refine → Apply."""
        rt_result = self.redteam.execute(self.redteam.params)
        candidates = rt_result.get("candidates", [])

        norms = self.flywheel_overlay.get_norms()
        for cand in candidates:
            v_result = self.verifier.execute({"candidate": cand, "norms": norms})
            if v_result["result"].outcome == VerificationOutcome.VIOLATION:
                self.triage.submit(v_result["result"], cand)

        verified_items = self.triage.pop_all()
        patches = []

        if verified_items:
            ref_result = self.refinement.execute({
                "verified_items": verified_items,
                "oracle_version": self.oracle.get_version(),
                **self.refinement.params,
            })
            batch = ref_result.get("batch")
            if batch and batch.local_corrections:
                self.oracle.apply_batch(batch)
                self.flywheel_overlay.apply_batch(batch)

                for lc in batch.local_corrections:
                    patch_info = {
                        "type": lc.correction_type.value,
                    }
                    if lc.correction_type == CorrectionType.MEDICAL_HARD_BLOCK:
                        patch_info["description"] = (
                            f"Hard-block keyword: \"{lc.payload.get('keyword', '')}\""
                        )
                    elif lc.correction_type == CorrectionType.THRESHOLD_ADJUSTMENT:
                        key = lc.payload.get("key", "")
                        disp = lc.payload.get("min_disposition", "")
                        patch_info["description"] = (
                            f"Disposition override: {key} → min {disp}"
                        )
                    elif lc.correction_type == CorrectionType.AUDIT_COVERAGE_UPDATE:
                        cc = lc.payload.get("case_class", "")
                        patch_info["description"] = (
                            f"Audit coverage: {cc}"
                        )
                    else:
                        patch_info["description"] = str(lc.payload)
                    patches.append(patch_info)

        return {
            "rt_candidates": len(candidates),
            "violations": len(verified_items),
            "patches": patches,
            "oracle_version": self.oracle.get_version(),
        }

    def step(self):
        """One full iteration: evaluate → record → govern."""
        self.iteration += 1

        # Evaluate before governance
        case_results = self.evaluate_cases()

        # Run governance
        gov_result = self.run_governance()

        record = {
            "iteration": self.iteration,
            "cases": case_results,
            "governance": gov_result,
            "oracle_version": self.oracle.get_version(),
        }
        self.history.append(record)
        return record

    def evaluate_custom_case(self, case_data):
        """Evaluate a single custom case (interactive proposer)."""
        results = self.evaluate_cases([case_data])
        return results[0] if results else None


# ── Global demo instances ────────────────────────────────────
demos = {}


def get_or_create_demo(name):
    if name not in DEMO_CONFIGS:
        return None
    if name not in demos:
        cfg = DEMO_CONFIGS[name]
        demos[name] = DemoInstance(name, cfg["config"], cfg["cases"], cfg["title"])
    return demos[name]


# ── Flask app ────────────────────────────────────────────────

def create_app():
    app = Flask(__name__, static_folder=None)
    CORS(app)

    # ── API routes ───────────────────────────────────────────

    @app.route("/api/demos", methods=["GET"])
    def list_demos():
        return jsonify([
            {"id": k, "title": v["title"], "case_count": len(v["cases"])}
            for k, v in DEMO_CONFIGS.items()
        ])

    @app.route("/api/demo/<demo_id>/init", methods=["POST"])
    def init_demo(demo_id):
        if demo_id in demos:
            del demos[demo_id]
        demo = get_or_create_demo(demo_id)
        if not demo:
            return jsonify({"error": "Unknown demo"}), 404
        return jsonify({
            "name": demo.name,
            "title": demo.title,
            "case_count": len(demo.fixed_cases),
            "max_iterations": demo.max_iterations,
            "cases": [{
                "id": c.get("id"),
                "patient_message": c.get("patient_message", ""),
                "draft_reply": c.get("draft_reply", ""),
                "case_type": c.get("case_type", ""),
                "evidence_status": c.get("evidence_status", ""),
                "category": c.get("category", ""),
                "proposed_disposition": c.get("proposed_disposition", ""),
            } for c in demo.fixed_cases],
        })

    @app.route("/api/demo/<demo_id>/step", methods=["POST"])
    def step_demo(demo_id):
        demo = get_or_create_demo(demo_id)
        if not demo:
            return jsonify({"error": "Unknown demo"}), 404
        if demo.iteration >= demo.max_iterations:
            return jsonify({"error": "Max iterations reached"}), 400
        result = demo.step()
        return jsonify(result)

    @app.route("/api/demo/<demo_id>/history", methods=["GET"])
    def get_history(demo_id):
        demo = demos.get(demo_id)
        if not demo:
            return jsonify({"error": "Demo not initialized"}), 404
        return jsonify({
            "iteration": demo.iteration,
            "history": demo.history,
        })

    @app.route("/api/demo/<demo_id>/reset", methods=["POST"])
    def reset_demo(demo_id):
        if demo_id in demos:
            del demos[demo_id]
        demo = get_or_create_demo(demo_id)
        if not demo:
            return jsonify({"error": "Unknown demo"}), 404
        return jsonify({"status": "reset", "name": demo.name})

    @app.route("/api/demo/<demo_id>/evaluate_custom", methods=["POST"])
    def evaluate_custom(demo_id):
        demo = get_or_create_demo(demo_id)
        if not demo:
            return jsonify({"error": "Unknown demo"}), 404
        case_data = request.get_json() or {}
        result = demo.evaluate_custom_case(case_data)
        return jsonify(result)

    # ── Serve React frontend ─────────────────────────────────

    @app.route("/")
    def serve_index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/<path:path>")
    def serve_static(path):
        file_path = os.path.join(FRONTEND_DIR, path)
        if os.path.isfile(file_path):
            return send_from_directory(FRONTEND_DIR, path)
        return send_from_directory(FRONTEND_DIR, "index.html")

    return app


def main():
    parser = argparse.ArgumentParser(description="Flywheel Visual Demo Server")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--dev", action="store_true",
                        help="Run without serving static files (React dev server)")
    args = parser.parse_args()

    app = create_app()
    print(f"  Flywheel Visual Demo Server")
    print(f"  API:      http://localhost:{args.port}/api/demos")
    if not args.dev:
        print(f"  Frontend: http://localhost:{args.port}/")
    else:
        print(f"  Dev mode: connect React dev server to port {args.port}")
    print()
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

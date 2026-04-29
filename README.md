# Alignment Flywheel — Reference Implementation



## Summary

The **Alignment Flywheel** is a governance-centric hybrid multi-agent system (MAS) for architecture-agnostic AI safety. Rather than baking safety constraints into a model's training objective or prompt, the Flywheel wraps *any* AI system (the "Oracle") in a closed governance loop that continuously discovers safety violations, verifies them against formal norms, and deploys targeted corrections — all without retraining or human interaction.

<p align="center">
  <img src="./paper/spatial_3d_progression.png" alt="Spatial 3D demo progression" width="100%"/>
</p>

The governance loop follows an **OODA** (Observe → Orient → Decide → Act) decomposition:

1. **Red Team** probes the Oracle to discover candidate flaws (outputs that violate safety norms).
2. **Verifier** checks each candidate against typed norms loaded from a Flywheel Overlay.
3. **Triage** prioritises verified violations by severity.
4. **Refinement** plans a batch of minimal, regression-tested corrections and deploys them.
5. **Blue Team** monitors for collateral damage after each batch.

The loop repeats until no violations remain, producing a monotonically improving safety trajectory with full auditability. The architecture is domain-agnostic: this repository demonstrates it on both **continuous spatial reward surfaces** (Inverse IRL) and **discrete medical decision-making** scenarios using the exact same engine, protocols, and HTTP API — only the pluggable OODA strategy modules differ.

## Citation

If you use this work, please cite both papers:

The original vision and introduction of the Alignment Flywheel concept:

```bibtex
@article{malomgre2026interactionless,
  title={Interactionless Inverse Reinforcement Learning: A Data-Centric Framework for Durable Alignment},
  author={Malomgr{\'e}, Elias and Simoens, Pieter},
  journal={arXiv preprint arXiv:2602.14844},
  year={2026}
}
```

The formalization and practical implementation (this repository):

```bibtex
@article{malomgre2026alignment,
  title={The Alignment Flywheel: A Governance-Centric Hybrid MAS for Architecture-Agnostic Safety},
  author={Malomgr{\'e}, Elias and Simoens, Pieter},
  journal={arXiv preprint arXiv:2603.02259},
  year={2026}
}
```

## What's in this codebase

A clean separation between **abstract protocols**, **concrete
implementations** (one class per file), an **HTTP API layer**
that decouples Oracle / Proposer / Flywheel / Enforcement into
service-style endpoints, and **5 demos** that drive the same
governance loop across different scenarios.

```
flywheel/
  protocols/
    enums.py
    artifacts/                # 13 typed dataclasses, one per file
    ooda/                     # ObserveStep, OrientStep, DecideStep, ActStep, OODARole
    interfaces/               # 10 abstract interfaces, one per file
  core/
    knowledge_base/in_memory_knowledge_base.py
    query_merger/default_query_merger.py
    batch_applier/default_batch_applier.py
    governance_engine.py
  api/
    app.py                    # Flask app factory + start_api_in_thread
    blueprints/               # /oracle/*, /proposer/*, /flywheel/*, /enforcement/*
    clients/                  # HTTP clients implementing each interface
  factory/
    registry.py               # FactoryRegistry — name → class
    auto_register.py          # imports every implementation
  roles/
    redteam/{observe,orient,decide,act}/   # OODA steps, one file each
    verifier/{observe,orient,decide,act}/
    refinement/{observe,orient,decide,act}/
    triage/                   # FIFOTriage, PriorityTriage
    blueteam/collateral_monitor.py
    proposer/                 # PassthroughProposer, SpatialProposer
    oracle/                   # SpatialOracle, PatientPortalOracle, SimpleMedicalOracle, ComplexMedicalOracle
    oracle/adapters/          # PrecomputedGridOracle
    flywheel_overlay/         # SpatialOverlay, PatientPortalOverlay, SimpleMedicalOverlay, ComplexMedicalOverlay
    enforcement/default_enforcement.py
  demos/
    spatial_3d/               # adaptive bandwidth + cumulative regression
    spatial_3d_fixed_bw/      # baseline (no regression test)
    simple_medical/           # minimal viable medical pipeline
    complex_medical/          # 5-D oracle, drug interactions, multi-specialty
    patient_portal/           # generative Red Team, structured-output norms
```

## Running the demos

Each demo runs end-to-end through the HTTP API. The API is
started in-process by the demo runner; `--port` lets you pick a
port (default varies per demo).

```bash
# Main IIRL demo — adaptive bandwidth, regression-tested
python -m flywheel.demos.spatial_3d.run \
    --port 5000 --loss-data loss_values.npy

# Baseline comparison — fixed bandwidth, no regression test
python -m flywheel.demos.spatial_3d_fixed_bw.run \
    --port 5001 --loss-data loss_values.npy

# Minimal medical pipeline (smallest viable demo)
python -m flywheel.demos.simple_medical.run --port 5002

# Rich medical scenario (5-D oracle, drug interactions)
python -m flywheel.demos.complex_medical.run --port 5003

# Patient portal (paper's main medical scenario)
python -m flywheel.demos.patient_portal.run --port 5004
```

## Architecture

**Components run as services.** Oracle, Proposer, Flywheel
overlay, and Enforcement live behind a Flask app at
`localhost:<port>` with four route prefixes:

```
POST /oracle/query        POST /proposer/propose
POST /oracle/apply_batch  POST /flywheel/overlay
GET  /oracle/version      POST /flywheel/apply_batch
                          GET  /flywheel/norms
                          GET  /flywheel/version
                          POST /enforcement/decide
```

**Governance code never imports concrete components.** It uses
HTTP client classes (`HTTPOracleClient`,
`HTTPSpatialOracleClient`, `HTTPProposerClient`,
`HTTPFlywheelClient`, `HTTPEnforcementClient`) that implement
the abstract interfaces. So the same governance code works
whether components are local or remote.

**Two HTTP calls per batch deploy.** When refinement emits a
`GovernanceBatch`, the engine calls `POST /oracle/apply_batch`
*and* `POST /flywheel/apply_batch` separately — because in a
production deployment these are two different services.

## OODA roles — extension model

Every governance role (Red Team, Verifier, Refinement) is built
from 4 swappable `ObserveStep` / `OrientStep` / `DecideStep` /
`ActStep` instances. To add a new strategy, write the new step
file, register it, and change one line in YAML:

```yaml
redteam:
  observe: MyNewObserver       # swap just this step
  orient: DistanceOrienter     # reuse existing
  decide: FarthestFirstDecider
  act: CandidateSubmitter
```

## Demo results

| Demo                  | Result                                         |
|-----------------------|------------------------------------------------|
| `spatial_3d`          | 4927 → 0 flaws in 14 iters, basin 837 → 837    |
| `spatial_3d_fixed_bw` | 4927 → 2879 flaws after 25 iters (baseline)    |
| `simple_medical`      | 38% → 0% escalation in 2 iters                 |
| `complex_medical`     | 33% → 0% escalation in 2 iters                 |
| `patient_portal`      | 33% → 13% escalation in 2 iters                |

## Dependencies

`numpy`, `scipy`, `matplotlib`, `pyyaml`, `flask`, `requests`.

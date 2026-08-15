# Alignment Flywheel — Reference Implementation



## Summary

The **Alignment Flywheel** is a governance-centric hybrid multi-agent system (MAS) for architecture-agnostic AI safety. Rather than baking safety constraints into a model's training objective or prompt, the Flywheel wraps *any* AI system (the "Oracle") in a closed governance loop that continuously discovers safety violations, verifies them against formal norms, and deploys targeted corrections — all without retraining or human interaction.

<p align="center">
  <img src="spatial_3d_progression.png" alt="Spatial 3D demo progression" width="100%"/>
</p>

The governance loop follows an **OODA** (Observe → Orient → Decide → Act) decomposition:

1. **Red Team** probes the Oracle to discover candidate flaws (outputs that violate safety norms).
2. **Verifier** checks each candidate against typed norms loaded from a Flywheel Overlay.
3. **Triage** prioritises verified violations by severity.
4. **Refinement** plans a batch of minimal, regression-tested corrections and deploys them.
5. **Blue Team** monitors for collateral damage after each batch.

The loop repeats until no violations remain, producing a monotonically improving safety trajectory with full auditability. The architecture is domain-agnostic: this repository demonstrates it on both **continuous spatial reward surfaces** (Inverse IRL) and **discrete medical decision-making** scenarios using the exact same engine, protocols, and HTTP API, only the pluggable OODA strategy modules differ.

This repository collects the full research codebase spanning both papers — the governance framework, the MoE-based inverse imitation reward learning, and the patching infrastructure (bottleneck-space Gaussian suppression with SimHash LSH acceleration).

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

Allowance of Alignment Flywheel guided kernel supresion patching at scale

```bibtex
@inproceedings{malomgrealignment,
  title={Alignment Flywheel-Guided Suppression Patching for Interactionless Inverse Reinforcement Learning},
  author={Malomgr{\'e}, Elias and Simoens, Pieter},
  booktitle={Third Reinforcement Learning Beyond Rewards Workshop at the Reinforcement Learning Conference}
}
```

## What's in this codebase

This release is structured as three self-contained packages:

| Package | Purpose |
|---------|---------|
| **flywheel/** | Governance framework — protocols, OODA roles, engine |
| **IIRL/** | Inverse Imitation Reward Learning — the learned MoE models |
| **patching/** | Kernel infrastructure — z-space patching, LSH, kernel building |

```
flywheel/                       Governance framework (OODA loop)
├── protocols/
│   ├── enums.py                CorrectionType, NormKind, VerificationOutcome
│   ├── artifacts/              GovernanceBatch, LocalCorrection, CandidateFlaw
│   ├── interfaces/             BaseSpatialOracleAdapter, BaseBatchApplier
│   └── ooda/                   ObserveStep, OrientStep, DecideStep, ActStep, OODARole
├── roles/
│   ├── oracle/                 MoE2DOracle, MoELocomotionOracle
│   ├── redteam/observers/      LaneDisciplineObserver, GridObserver, LocomotionObserver
│   └── refinement/
│       ├── orienters/          LaneBandwidthOrienter, AdaptiveBandwidthOrienter, ...
│       └── deciders/           LaneRegressionDecider, CumulativeRegressionDecider, ...
└── engine/
    ├── governance_engine.py    Full OODA loop orchestrator
    └── default_batch_applier.py

IIRL/                           Inverse Imitation Reward Learning
├── models.py                   MixtureOfExperts, GatedAutoencoder, GatingNetwork
└── mapping_function.py         MappingFunction: MSE → reward via exp(-norm * steepness)

patching/                       Kernel patching infrastructure
├── kernel/
│   └── flywheel_kernel.py      FlywheelKernel (MoE + z-space Gaussian patches + SimHash LSH)
├── build/
│   ├── build_kernels.py        End-to-end: MoE + patches + LSH → single .pt kernel
│   ├── build_constraints.py    Constraint extraction from demonstration data
│   └── build_safety_constraints.py
├── load/
│   ├── load_kernel.py          FlywheelKernel loader
│   └── load_constraints.py
├── evaluation/
│   ├── stress_test_safety.py   Multi-environment adversarial stress test
│   └── benchmark_lsh.py        SimHash LSH accuracy + latency benchmark
└── environments/
    ├── benchmark_layouts.py    PointMaze 8×8 layouts (5 variants + routes)
    └── q_iteration*.py         Tabular value iteration with patched reward
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
HTTP client classes that implement the abstract interfaces. So
the same governance code works whether components are local or remote.

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

## Quick start

```python
# ── Inference (single .pt file, no governance loop needed) ──
from patching import FlywheelKernel

fk = FlywheelKernel("kernels/Ant_flywheel_tight.pt")
safety = fk.safety(observations)       # (N,) float32 in [0, 1]
reward = fk.reward(observations)       # (N,) base MoE reward
suppression = fk.suppression(observations)  # (N,) patch suppression

# ── Training the MoE ──
from IIRL import MixtureOfExperts, MappingFunction

model = MixtureOfExperts(input_dim=111, bottleneck_dim=8, num_experts=5)
# ... train on demonstrations ...
mapping = MappingFunction(l_min=0.001, l_max=0.15, steepness=4.0)

# ── Full governance loop ──
from flywheel.engine import GovernanceEngine
from flywheel.roles.oracle import MoE2DOracle
from flywheel.roles.redteam.observers import LaneDisciplineObserver
from flywheel.roles.refinement.orienters import LaneBandwidthOrienter
from flywheel.roles.refinement.deciders import LaneRegressionDecider
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

- Python >= 3.10
- PyTorch >= 2.0
- NumPy, SciPy

## License

Apache-2.0

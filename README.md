# Alignment Flywheel — Reference Implementation



## Summary

The **Alignment Flywheel** is a governance-centric hybrid multi-agent system (MAS) for architecture-agnostic AI safety. Rather than baking safety constraints into a model's training objective or prompt, the Flywheel wraps *any* AI system (the "Oracle") in a closed governance loop that continuously discovers safety violations, verifies them against formal norms, and deploys targeted corrections — all without retraining or human interaction.

The governance loop follows an **OODA** (Observe → Orient → Decide → Act) decomposition:

1. **Red Team** probes the Oracle to discover candidate flaws (outputs that violate safety norms).
2. **Verifier** checks each candidate against typed norms loaded from a Flywheel Overlay.
3. **Triage** prioritises verified violations by severity.
4. **Refinement** plans a batch of minimal, regression-tested corrections and deploys them.
5. **Blue Team** monitors for collateral damage after each batch.

The loop repeats until no violations remain, producing a monotonically improving safety trajectory with full auditability.

## Citation

```bibtex
@article{malomgre2026interactionless,
  title={Interactionless Inverse Reinforcement Learning: A Data-Centric Framework for Durable Alignment},
  author={Malomgr{\'e}, Elias and Simoens, Pieter},
  journal={arXiv preprint arXiv:2602.14844},
  year={2026}
}
```

```bibtex
@article{malomgre2026alignment,
  title={The Alignment Flywheel: A Governance-Centric Hybrid MAS for Architecture-Agnostic Safety},
  author={Malomgr{\'e}, Elias and Simoens, Pieter},
  journal={arXiv preprint arXiv:2603.02259},
  year={2026}
}
```

## Package structure

This release contains three packages:

```
flywheel/                       Governance framework (OODA loop)
├── protocols/
│   ├── enums.py                CorrectionType, NormKind, VerificationOutcome
│   ├── artifacts/              GovernanceBatch, LocalCorrection, CandidateFlaw
│   ├── interfaces/             BaseSpatialOracleAdapter, BaseBatchApplier
│   └── ooda/                   ObserveStep, OrientStep, DecideStep, ActStep, OODARole
├── roles/
│   ├── oracle/                 MoE2DOracle, MoELocomotionOracle
│   ├── redteam/observers/      LaneDisciplineObserver, GridObserver, ...
│   └── refinement/
│       ├── orienters/          LaneBandwidthOrienter, AdaptiveBandwidthOrienter
│       └── deciders/           LaneRegressionDecider, CumulativeRegressionDecider
└── engine/
    ├── governance_engine.py    Full OODA loop orchestrator
    └── default_batch_applier.py

IIRL/                           Inverse Imitation Reward Learning
├── models.py                   MixtureOfExperts, GatedAutoencoder, GatingNetwork
└── mapping_function.py         MappingFunction: MSE → reward via exp(-norm * steepness)

patching/                       Kernel patching infrastructure
├── kernel/
│   └── flywheel_kernel.py      FlywheelKernel (MoE + z-space patches + SimHash LSH)
├── build/
│   ├── build_kernels.py        End-to-end: MoE + patches + LSH → .pt kernel
│   ├── build_constraints.py    Constraint extraction from demonstration data
│   └── build_safety_constraints.py
├── load/
│   ├── load_kernel.py          FlywheelKernel loader
│   └── load_constraints.py
├── evaluation/
│   ├── stress_test_safety.py   Multi-environment adversarial stress test
│   └── benchmark_lsh.py        LSH accuracy + latency benchmark
└── environments/
    ├── benchmark_layouts.py    PointMaze 8×8 layouts
    └── q_iteration*.py         Value iteration with patched reward
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

# ── Governance loop ──
from flywheel.engine import GovernanceEngine
from flywheel.roles.oracle import MoE2DOracle
from flywheel.roles.redteam.observers import LaneDisciplineObserver
from flywheel.roles.refinement.orienters import LaneBandwidthOrienter
from flywheel.roles.refinement.deciders import LaneRegressionDecider
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy, SciPy

## License

Apache-2.0

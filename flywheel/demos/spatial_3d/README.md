# Spatial 3D — Adaptive Bandwidth + Cumulative Regression

The main IIRL demo.  Pre-computed loss values from the MoE
autoencoder define the oracle's reward field.  The flywheel
discovers reward regions far from the expert path, patches them
with adaptively-sized suppression kernels, and verifies basin
preservation with a cumulative regression test.

Expected: ~11 iterations, 0 flaws, 100 % basin preserved.

## Run

```bash
python -m flywheel.demos.spatial_3d.run \
    --port 5000 \
    --loss-data /path/to/loss_values.npy \
    --output outputs/spatial_3d
```

## What's in this demo

- **Server side (Flask API at :5000)**
  - `PrecomputedGridOracle` adapter wrapped by `SpatialOracle`
  - `SpatialProposer`
  - `SpatialOverlay` (1 spatial-boundary norm)
  - `DefaultEnforcement`
- **Governance side (OODA roles, in-process, HTTP clients to the API)**
  - Red Team: `FineSamplingObserver` → `DistanceOrienter` → `FarthestFirstDecider` → `CandidateSubmitter`
  - Verifier: `NormLoader` → `SpatialNormMatcher` → `SpatialViolationDecider` → `VerificationEmitter`
  - Refinement: `QueueObserver` → `AdaptiveBandwidthOrienter` → `CumulativeRegressionDecider` → `BatchDeployer`
  - `PriorityTriage` + `CollateralMonitor`

## Outputs

- `iteration_*.png` — per-iteration 3D scatter
- `convergence.csv` — flaws/basin/patches per iteration
- `results.json` — full run metadata

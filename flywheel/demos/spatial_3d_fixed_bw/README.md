# Spatial 3D — Fixed Bandwidth Baseline

Same architecture as `spatial_3d`, swaps three OODA steps via YAML:

| Role        | Step      | Adaptive demo                  | Fixed-BW baseline           |
|-------------|-----------|--------------------------------|-----------------------------|
| Red Team    | observe   | `FineSamplingObserver`         | `GridObserver`              |
| Refinement  | orient    | `AdaptiveBandwidthOrienter`    | `FixedBandwidthOrienter`    |
| Refinement  | decide    | `CumulativeRegressionDecider`  | `NoCumulativeDecider`       |

Expected: **2,879 flaws remaining after 25 iterations** (vs 0 in
adaptive demo). Basin still preserved (kernels are narrow).

The point of this demo: the architecture admits baseline
comparisons by editing YAML — no code changes.

## Run

```bash
python -m flywheel.demos.spatial_3d_fixed_bw.run \
    --port 5001 \
    --loss-data /path/to/loss_values.npy \
    --output outputs/spatial_3d_fixed_bw
```

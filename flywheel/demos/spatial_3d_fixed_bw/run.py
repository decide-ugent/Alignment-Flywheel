"""Spatial 3D fixed-bandwidth baseline.

Same architecture as spatial_3d, swaps:
  - GridObserver (instead of FineSamplingObserver)
  - FixedBandwidthOrienter (instead of AdaptiveBandwidthOrienter)
  - NoCumulativeDecider (instead of CumulativeRegressionDecider)

Demonstrates that the architecture supports baseline comparisons
through nothing more than YAML changes.

Usage:
    python -m flywheel.demos.spatial_3d_fixed_bw.run --port 5001
"""

import argparse
import os
import sys

from flywheel.demos.spatial_3d.run import main as spatial_main


def main():
    """Re-use the spatial_3d runner with this folder's config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--loss-data", default=None)
    parser.add_argument("--output", default="outputs/spatial_3d_fixed_bw")
    args, _ = parser.parse_known_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    sys.argv = [
        "run.py",
        "--port", str(args.port),
        "--output", args.output,
        "--config", cfg_path,
    ]
    if args.loss_data:
        sys.argv += ["--loss-data", args.loss_data]
    spatial_main()


if __name__ == "__main__":
    main()

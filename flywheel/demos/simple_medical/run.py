"""Simple medical demo runner.

Minimal viable medical pipeline: 60-line heuristic oracle, 2
norms, FIFO triage, 8 cases. Architecture at maximum transparency.

Usage:
    python -m flywheel.demos.simple_medical.run --port 5003
"""

import argparse
import os

from flywheel.demos._medical_runner import run_medical_demo
from flywheel.demos.simple_medical.fixed_cases import SIMPLE_CASES


def main():
    parser = argparse.ArgumentParser(description="Simple medical demo")
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--output", default="outputs/simple_medical")
    args = parser.parse_args()

    cfg = os.path.join(os.path.dirname(__file__), "config.yaml")
    run_medical_demo(
        config_path=cfg,
        fixed_cases=SIMPLE_CASES,
        output_dir=args.output,
        port=args.port,
        title="Simple Medical (minimal viable pipeline)",
    )


if __name__ == "__main__":
    main()

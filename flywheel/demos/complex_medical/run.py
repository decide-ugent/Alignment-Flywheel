"""Complex medical demo runner.

Multi-dimensional oracle, drug interactions, vulnerable-patient
protections, multi-specialty cases. Tests the architecture's
ability to host a richer oracle and norm set with no protocol
changes.

Usage:
    python -m flywheel.demos.complex_medical.run --port 5004
"""

import argparse
import os

from flywheel.demos._medical_runner import run_medical_demo
from flywheel.demos.complex_medical.fixed_cases import COMPLEX_CASES


def main():
    parser = argparse.ArgumentParser(description="Complex medical demo")
    parser.add_argument("--port", type=int, default=5004)
    parser.add_argument("--output", default="outputs/complex_medical")
    args = parser.parse_args()

    cfg = os.path.join(os.path.dirname(__file__), "config.yaml")
    run_medical_demo(
        config_path=cfg,
        fixed_cases=COMPLEX_CASES,
        output_dir=args.output,
        port=args.port,
        title="Complex Medical (5D oracle, 4 norms, 3 norm kinds)",
    )


if __name__ == "__main__":
    main()

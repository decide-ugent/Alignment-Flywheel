"""Patient portal demo runner.

Generative Red Team explores the case space; the flywheel installs
disposition overrides and hard-blocks; safety score on the fixed
eval set climbs to 100% as governance iterates.

Usage:
    python -m flywheel.demos.patient_portal.run --port 5002
"""

import argparse
import os

from flywheel.demos._medical_runner import run_medical_demo
from flywheel.demos.patient_portal.fixed_cases import PORTAL_CASES


def main():
    parser = argparse.ArgumentParser(description="Patient portal demo")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--output", default="outputs/patient_portal")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    run_medical_demo(
        config_path=cfg_path,
        fixed_cases=PORTAL_CASES,
        output_dir=args.output,
        port=args.port,
        title="Patient Portal Governance",
    )


if __name__ == "__main__":
    main()

"""BatchDeployer — assembles spatial GovernanceBatch from accepted patches."""

from typing import Any, Dict

from flywheel.protocols.ooda.act_step import ActStep
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch
from flywheel.protocols.artifacts.local_correction import LocalCorrection
from flywheel.protocols.enums import CorrectionType


class BatchDeployer(ActStep):
    """Act: build spatial GovernanceBatch with flaw patches + coverage updates."""

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        accepted = decision["accepted"]
        oracle_version = decision["oracle_version"]
        v = int(oracle_version.split("v")[-1]) if "v" in oracle_version else 0

        corrections = []
        patched_points = []
        for item in accepted:
            pt = item["point"]
            bw = item["final_bw"]
            n_cov = item.get("predicted_covered", 0)
            corrections.append(LocalCorrection(
                correction_type=CorrectionType.SPATIAL_FLAW_PATCH,
                payload={"flaw_point": pt, "support_radius": bw},
                description=(
                    f"Suppress at [{pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f}] "
                    f"bw={bw:.3f}"
                ),
            ))
            case_class = (f"spatial|bw={bw:.3f}|cov={n_cov}"
                          if n_cov else f"spatial|d={item['dist']:.1f}")
            corrections.append(LocalCorrection(
                correction_type=CorrectionType.AUDIT_COVERAGE_UPDATE,
                payload={"case_class": case_class},
            ))
            patched_points.append(pt)

        predicted_coverage = decision.get("predicted_coverage", 0)
        batch = GovernanceBatch(
            from_oracle_version=oracle_version,
            to_oracle_version=f"oracle:v{v + 1}",
            local_corrections=corrections,
            regression_evidence={
                "patched": len(patched_points),
                "rejected": decision["rejected"],
                "shrinks": decision["shrinks"],
                "predicted_coverage": predicted_coverage,
            },
            signature="regression-verified",
        )

        return {
            "batch": batch,
            "patched_points": patched_points,
            "rejected": decision["rejected"],
            "shrinks": decision["shrinks"],
            "predicted_coverage": predicted_coverage,
        }

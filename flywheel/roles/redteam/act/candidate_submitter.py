"""CandidateSubmitter — emits spatial CandidateFlaw artifacts."""

from typing import Any, Dict

from flywheel.protocols.ooda.act_step import ActStep
from flywheel.protocols.artifacts.candidate_flaw import CandidateFlaw


class CandidateSubmitter(ActStep):
    """Act: emit CandidateFlaw artifacts from spatial flaw indices."""

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        indices = decision["sorted_flaw_indices"]
        points = decision["points"]
        values = decision["values"]
        dists = decision["dists"]
        version = decision["oracle_version"]

        candidates = []
        for i in indices:
            pt = points[i].tolist()
            candidates.append(CandidateFlaw(
                context={"point": pt, "dist_to_path": float(dists[i])},
                trajectory={"kind": "spatial",
                             "steps": [{"payload": {"point": pt}}]},
                s=float(values[i]), u=0.2, u_thresh=0.5,
                v_O=version))

        return {"candidates": candidates, "count": len(candidates)}

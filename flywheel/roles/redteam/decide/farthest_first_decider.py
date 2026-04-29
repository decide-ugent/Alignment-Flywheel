"""FarthestFirstDecider — prioritises flaws by distance descending."""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.decide_step import DecideStep


class FarthestFirstDecider(DecideStep):
    """Decide: prioritise far flaws (wider kernels, more collateral coverage)."""

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        indices = oriented["flaw_indices"]
        dists = oriented["dists"]

        order = np.argsort(-dists[indices])
        sorted_indices = indices[order]

        return {
            "sorted_flaw_indices": sorted_indices,
            "points": oriented["points"],
            "values": oriented["values"],
            "dists": oriented["dists"],
            "oracle_version": oriented["oracle_version"],
        }

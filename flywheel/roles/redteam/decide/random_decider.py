"""RandomDecider — random ordering (baseline)."""

from typing import Any, Dict

import numpy as np

from flywheel.protocols.ooda.decide_step import DecideStep


class RandomDecider(DecideStep):
    """Decide: random ordering of flaws (baseline comparison)."""

    def decide(self, oriented: Dict[str, Any]) -> Dict[str, Any]:
        indices = oriented["flaw_indices"].copy()
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        return {
            "sorted_flaw_indices": indices,
            "points": oriented["points"],
            "values": oriented["values"],
            "dists": oriented["dists"],
            "oracle_version": oriented["oracle_version"],
        }

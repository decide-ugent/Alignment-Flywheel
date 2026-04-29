"""CollateralMonitor — re-verifies unpatched flaws to measure collateral."""

from typing import Any, List, Tuple

import numpy as np

from flywheel.protocols.interfaces.base_blueteam import BaseBlueTeam


class CollateralMonitor(BaseBlueTeam):
    """Re-queries unpatched flaws; drops those now suppressed by nearby patches."""

    def __init__(self, oracle_adapter):
        self.oracle = oracle_adapter
        self.total_queries = 0

    def check_collateral(
        self,
        unpatched_points: List[Any],
        safety_floor: float = 0.01,
    ) -> Tuple[List[Any], int]:
        if not unpatched_points:
            return [], 0
        result = self.oracle.query_points(unpatched_points)
        vals = np.array(result["values"])
        self.total_queries += len(unpatched_points)
        still_active = [
            unpatched_points[i]
            for i in range(len(vals))
            if vals[i] > safety_floor
        ]
        collateral = len(unpatched_points) - len(still_active)
        return still_active, collateral

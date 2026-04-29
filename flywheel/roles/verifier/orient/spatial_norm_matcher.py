"""SpatialNormMatcher — matches spatial candidates against Phi."""

from typing import Any, Dict

from scipy.spatial.distance import cdist

from flywheel.protocols.ooda.orient_step import OrientStep
from flywheel.protocols.enums import NormKind


class SpatialNormMatcher(OrientStep):
    """Orient: match spatial candidate against SPATIAL_BOUNDARY norms."""

    def orient(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        candidate = observation["candidate"]
        expert = observation.get("expert_path")
        boundary = observation["boundary"]
        norms = observation["norms"]

        point = candidate.context.get("point")
        dist = candidate.context.get("dist_to_path")
        if point is not None and dist is None and expert is not None:
            dist = float(cdist([point], expert).min())

        matched_norms = []
        for norm in norms:
            if norm.kind == NormKind.SPATIAL_BOUNDARY:
                if dist is not None and dist > boundary:
                    matched_norms.append((norm, dist))
            else:
                matched_norms.append((norm, None))

        return {
            "candidate": candidate,
            "matched_norms": matched_norms,
            "dist": dist,
            "boundary": boundary,
        }

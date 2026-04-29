"""SpatialProposer — wraps a spatial point as a Trajectory."""

from flywheel.protocols.interfaces.base_proposer import BaseProposer
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.trajectory_step import TrajectoryStep
from flywheel.protocols.enums import TrajectoryKind


class SpatialProposer(BaseProposer):
    """Wraps a 3D query point as a spatial Trajectory."""

    def propose(self, context: Context, **kwargs) -> Trajectory:
        return Trajectory(
            kind=TrajectoryKind.SPATIAL,
            steps=[TrajectoryStep(
                step_index=0,
                payload={"point": context.data.get("point", [0, 0, 0])},
            )],
            metadata={"source": "spatial"},
        )

"""PassthroughProposer — wraps input data as a message Trajectory."""

from flywheel.protocols.interfaces.base_proposer import BaseProposer
from flywheel.protocols.artifacts.context import Context
from flywheel.protocols.artifacts.trajectory import Trajectory
from flywheel.protocols.artifacts.trajectory_step import TrajectoryStep
from flywheel.protocols.enums import TrajectoryKind


class PassthroughProposer(BaseProposer):
    """Wraps context.data as a message Trajectory without modification."""

    def propose(self, context: Context, **kwargs) -> Trajectory:
        return Trajectory(
            kind=TrajectoryKind.MESSAGE,
            steps=[TrajectoryStep(step_index=0,
                                  payload=context.data,
                                  metadata=context.data)],
            metadata={"source": "passthrough"},
        )

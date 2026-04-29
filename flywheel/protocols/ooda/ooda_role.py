"""OODARole — composes 4 OODA steps into a single execute() call.

Every governance role (Red Team, Verifier, Refinement) is built from
4 swappable step objects. The YAML config picks which concrete class
implements each step.
"""

from typing import Any, Dict

from flywheel.protocols.ooda.observe_step import ObserveStep
from flywheel.protocols.ooda.orient_step import OrientStep
from flywheel.protocols.ooda.decide_step import DecideStep
from flywheel.protocols.ooda.act_step import ActStep


class OODARole:
    """Generic OODA-loop role — observe → orient → decide → act."""

    def __init__(
        self,
        observe: ObserveStep,
        orient: OrientStep,
        decide: DecideStep,
        act: ActStep,
        params: Dict[str, Any] = None,
    ):
        self.observe_step = observe
        self.orient_step = orient
        self.decide_step = decide
        self.act_step = act
        self.params = params or {}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full OODA loop once."""
        observation = self.observe_step.observe(context)
        oriented = self.orient_step.orient(observation)
        decision = self.decide_step.decide(oriented)
        result = self.act_step.act(decision)
        return result

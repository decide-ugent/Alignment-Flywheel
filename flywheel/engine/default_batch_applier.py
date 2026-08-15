"""DefaultBatchApplier — sends a batch to both Oracle and Flywheel services."""

from typing import Any, Dict

from flywheel.protocols.interfaces.base_batch_applier import BaseBatchApplier
from flywheel.protocols.interfaces.base_oracle import BaseOracle
from flywheel.protocols.interfaces.base_flywheel_overlay import BaseFlywheelOverlay
from flywheel.protocols.artifacts.governance_batch import GovernanceBatch


class DefaultBatchApplier(BaseBatchApplier):
    """Routes a GovernanceBatch to an Oracle and a Flywheel overlay."""

    def __init__(self, oracle: BaseOracle, flywheel: BaseFlywheelOverlay):
        self._oracle = oracle
        self._flywheel = flywheel

    def apply(self, batch: GovernanceBatch) -> Dict[str, Any]:
        return {
            "batch_id": batch.batch_id,
            "oracle_applied": self._oracle.apply_batch(batch),
            "flywheel_applied": self._flywheel.apply_batch(batch),
            "oracle_version": self._oracle.get_version(),
            "governance_version": self._flywheel.get_version(),
        }

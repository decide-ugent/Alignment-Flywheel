"""Abstract role interfaces — one per file."""

from flywheel.protocols.interfaces.base_proposer import BaseProposer
from flywheel.protocols.interfaces.base_oracle import BaseOracle
from flywheel.protocols.interfaces.base_spatial_oracle_adapter import BaseSpatialOracleAdapter
from flywheel.protocols.interfaces.base_flywheel_overlay import BaseFlywheelOverlay
from flywheel.protocols.interfaces.base_enforcement_policy import BaseEnforcementPolicy
from flywheel.protocols.interfaces.base_triage import BaseTriage
from flywheel.protocols.interfaces.base_blueteam import BaseBlueTeam
from flywheel.protocols.interfaces.base_knowledge_base import BaseKnowledgeBase
from flywheel.protocols.interfaces.base_query_merger import BaseQueryMerger
from flywheel.protocols.interfaces.base_batch_applier import BaseBatchApplier

__all__ = [
    "BaseProposer", "BaseOracle", "BaseSpatialOracleAdapter",
    "BaseFlywheelOverlay", "BaseEnforcementPolicy", "BaseTriage",
    "BaseBlueTeam", "BaseKnowledgeBase", "BaseQueryMerger", "BaseBatchApplier",
]

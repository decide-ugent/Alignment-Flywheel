"""BaseBlueTeam — abstract interface for Blue Team monitoring."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class BaseBlueTeam(ABC):
    """Blue Team — monitors live traffic, re-verifies after patches."""

    @abstractmethod
    def check_collateral(
        self,
        unpatched_points: List[Any],
        safety_floor: float = 0.01,
    ) -> Tuple[List[Any], int]:
        """Return (still_active, collateral_count)."""
        ...

"""MappingFunction — maps MoE reconstruction error to a reward signal.

Converts raw MSE loss into a [0, 1] reward:
    normalized = (mse - l_min) / (l_max - l_min)
    reward     = clip(exp(-normalized * steepness))

The exponential mapping gives high reward (near 1.0) for low reconstruction
error (on-distribution observations) and decays smoothly toward 0 for
out-of-distribution inputs.
"""

import math


class MappingFunction:
    """Maps reconstruction error to reward in [0, 1].

    Args:
        l_min: minimum expected MSE (from training data)
        l_max: maximum expected MSE (98th percentile)
        steepness: exponential decay rate (higher = sharper cutoff)
        clip_min: floor for output reward
        clip_max: ceiling for output reward
    """

    def __init__(self, l_min: float, l_max: float, steepness: float = 1.0,
                 clip_min: float = 0.0, clip_max: float = 1.0):
        self.l_min = l_min
        self.l_max = l_max
        self.steepness = steepness
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, mse: float) -> float:
        """Map a single MSE value to reward."""
        normalized = (mse - self.l_min) / (self.l_max - self.l_min + 1e-9)
        reward = math.exp(-normalized * self.steepness)
        return min(self.clip_max, max(self.clip_min, reward))

    def to_config(self) -> dict:
        """Serialize to config dict (stored in .pt kernel files)."""
        return {
            "l_min": self.l_min,
            "l_max": self.l_max,
            "steepness": self.steepness,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "MappingFunction":
        """Load from config dict."""
        return cls(
            l_min=cfg["l_min"],
            l_max=cfg["l_max"],
            steepness=cfg.get("steepness", 1.0),
        )

    def __repr__(self):
        return (f"MappingFunction(l_min={self.l_min:.6f}, "
                f"l_max={self.l_max:.6f}, steepness={self.steepness})")

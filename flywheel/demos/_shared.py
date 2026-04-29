"""Shared utilities used by demo runners."""

import numpy as np


def gen_expert_path(n: int = 20) -> np.ndarray:
    """Canonical synthetic expert path used by spatial demos."""
    pts = []
    for i in range(n):
        t = i / (n - 1)
        pts.append([
            -1 + 2 * t,
            np.clip(-1 + 2 * t + np.sin(np.pi * t) * 0.8, -1, 1),
            np.clip(-1 + 2 * t + np.cos(np.pi * t) * 0.8, -1, 1),
        ])
    return np.array(pts, dtype=np.float32)

"""Patching — kernel infrastructure for bottleneck + obs-space Gaussian patching.

FlywheelKernel bundles: MoE model + suppressive patches + SimHash LSH
into a single .pt file for O(1) inference.
"""
__version__ = "1.0.0"
from .kernel import FlywheelKernel

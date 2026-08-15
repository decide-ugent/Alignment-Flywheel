"""IIRL — Inverse Imitation Reward Learning.

Learns a reward function from demonstrations via a Mixture-of-Experts
autoencoder. Low reconstruction error = high reward (on-distribution).
"""
__version__ = "1.0.0"
from .models import MixtureOfExperts, GatedAutoencoder, GatingNetwork
from .mapping_function import MappingFunction

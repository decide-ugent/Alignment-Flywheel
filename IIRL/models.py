"""MoE model definitions for Inverse Imitation Reward Learning.

Architecture:
    Input (obs) → GatingNetwork → expert weights
    Input (obs) → Expert_i.encoder → z_i (bottleneck)
    z_i → Expert_i.decoder → reconstruction_i
    Output = Σ_i weight_i × reconstruction_i

Reward = f(MSE(input, output))  — low MSE = on-distribution = safe.
"""

import torch
from torch import nn
import numpy as np


class GatedAutoencoder(nn.Module):
    """Single expert: linear encoder (+ ReLU) → bottleneck → linear decoder."""

    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class GatingNetwork(nn.Module):
    """Soft gating: input → softmax weights over experts."""

    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x))


class MixtureOfExperts(nn.Module):
    """MoE autoencoder: N experts with shared gating.

    Args:
        input_dim: observation dimensionality
        bottleneck_dim: latent (z-space) dimensionality per expert
        num_experts: number of parallel autoencoder experts
    """

    def __init__(self, input_dim, bottleneck_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([
            GatedAutoencoder(input_dim, bottleneck_dim)
            for _ in range(num_experts)
        ])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        weights = self.gating_network(x)
        reconstructions = []
        for i, expert in enumerate(self.experts):
            w = weights[:, i].unsqueeze(1)
            reconstructions.append(expert(x) * w)
        return torch.stack(reconstructions, dim=1).sum(dim=1)

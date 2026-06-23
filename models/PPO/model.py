from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 4, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.actor = nn.Linear(hidden_dim, out_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.shared(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._encode(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_dist(self, x: torch.Tensor) -> Categorical:
        logits, _ = self.forward(x)
        return Categorical(logits=logits)

    def act(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
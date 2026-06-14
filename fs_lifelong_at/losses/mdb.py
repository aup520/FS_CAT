from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MDBStats:
    objective: float
    loss_sum: float
    variance: float
    num_domains: int


class MultiDomainBalancedLoss(nn.Module):
    """MDB objective: sum replay losses minus lambda2 times domain-loss variance."""

    def __init__(self, lambda2: float = 0.1) -> None:
        super().__init__()
        self.lambda2 = float(lambda2)

    def forward(self, domain_losses: list[torch.Tensor]) -> tuple[torch.Tensor, MDBStats]:
        if not domain_losses:
            raise ValueError("MDB requires at least one domain loss")
        losses = torch.stack([loss.reshape(()) for loss in domain_losses])
        loss_sum = losses.sum()
        variance = losses.var(unbiased=False) if losses.numel() > 1 else losses.new_zeros(())
        objective = loss_sum - self.lambda2 * variance
        stats = MDBStats(
            objective=float(objective.detach().cpu()),
            loss_sum=float(loss_sum.detach().cpu()),
            variance=float(variance.detach().cpu()),
            num_domains=int(losses.numel()),
        )
        return objective, stats

    def detached_domain_weights(self, domain_losses: list[torch.Tensor]) -> torch.Tensor:
        if not domain_losses:
            return torch.empty(0)
        losses = torch.stack([loss.detach().reshape(()) for loss in domain_losses])
        if losses.numel() == 1:
            return torch.ones_like(losses)
        centered = losses - losses.mean()
        return 1.0 - 2.0 * self.lambda2 * centered / losses.numel()

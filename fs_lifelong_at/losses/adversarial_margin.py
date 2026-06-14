from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


PNorm = Literal["inf", "2"]


@dataclass(frozen=True)
class AdversarialMarginConfig:
    epsilon: float = 8.0 / 255.0
    step_size: float = 2.0 / 255.0
    steps: int = 20
    p_norm: PNorm = "inf"
    grad_eps: float = 1.0e-12


def classification_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """z_y - max_{y' != y} z_y'."""

    true_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    masked_logits = logits.masked_fill(
        torch.zeros_like(logits, dtype=torch.bool).scatter_(1, targets.view(-1, 1), True),
        -torch.inf,
    )
    other_logits = masked_logits.max(dim=1).values
    return true_logits - other_logits


def normalize_direction(grad: torch.Tensor, p_norm: PNorm, eps: float) -> torch.Tensor:
    flat = grad.flatten(1)
    if p_norm == "inf":
        return grad.sign()
    if p_norm == "2":
        norm = flat.norm(p=2, dim=1).clamp_min(eps).view(-1, 1, 1, 1)
        return grad / norm
    raise ValueError(f"Unsupported p-norm: {p_norm}")


def project_delta(delta: torch.Tensor, epsilon: float, p_norm: PNorm, eps: float) -> torch.Tensor:
    if p_norm == "inf":
        return delta.clamp(-epsilon, epsilon)
    if p_norm == "2":
        flat = delta.flatten(1)
        norm = flat.norm(p=2, dim=1).clamp_min(eps)
        factor = torch.minimum(torch.ones_like(norm), torch.full_like(norm, epsilon) / norm)
        return delta * factor.view(-1, 1, 1, 1)
    raise ValueError(f"Unsupported p-norm: {p_norm}")


def dual_norm(grad: torch.Tensor, p_norm: PNorm, eps: float) -> torch.Tensor:
    flat = grad.flatten(1)
    if p_norm == "inf":
        return flat.norm(p=1, dim=1).clamp_min(eps)
    if p_norm == "2":
        return flat.norm(p=2, dim=1).clamp_min(eps)
    raise ValueError(f"Unsupported p-norm: {p_norm}")


class AdversarialMarginLoss(nn.Module):
    """ADM loss from the FS-CAT paper.

    The nearest boundary point is approximated by descending the classification
    margin under an epsilon ball. The final objective backpropagates the closed
    form margin-gradient factor from the DyART-style derivation:

        -phi_y(x_hat) / ||grad_x phi_y(x_hat)||_q

    The boundary search is internal to the ADM loss and does not save generated
    adversarial samples.
    """

    def __init__(self, config: AdversarialMarginConfig) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def from_dict(cls, config: dict) -> "AdversarialMarginLoss":
        p_norm = str(config.get("p_norm", "inf"))
        if p_norm not in {"inf", "2"}:
            raise ValueError("ADM p_norm must be 'inf' or '2'")
        return cls(
            AdversarialMarginConfig(
                epsilon=float(config.get("epsilon", 8.0 / 255.0)),
                step_size=float(config.get("step_size", 2.0 / 255.0)),
                steps=int(config.get("steps", 20)),
                p_norm=p_norm,  # type: ignore[arg-type]
                grad_eps=float(config.get("grad_eps", 1.0e-12)),
            )
        )

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        boundary = self.find_boundary_points(model, images, targets)
        boundary = boundary.detach().requires_grad_(True)
        logits = model(boundary)
        margin = classification_margin(logits, targets)
        input_grad = torch.autograd.grad(
            margin.sum(),
            boundary,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = dual_norm(input_grad, self.config.p_norm, self.config.grad_eps)
        loss = -(margin / grad_norm.detach()).mean()
        distance = (boundary.detach() - images).flatten(1)
        stats = {
            "adm_loss": float(loss.detach().cpu()),
            "adm_margin": float(margin.detach().mean().cpu()),
            "adm_boundary_distance": float(distance.norm(p=2, dim=1).mean().cpu()),
        }
        return loss, stats

    @torch.enable_grad()
    def find_boundary_points(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()
        delta = torch.zeros_like(images)
        for _ in range(self.config.steps):
            candidate = (images + delta).detach().clamp(0.0, 1.0).requires_grad_(True)
            logits = model(candidate)
            margin = classification_margin(logits, targets)
            grad = torch.autograd.grad(margin.mean(), candidate, only_inputs=True)[0]
            direction = -normalize_direction(grad, self.config.p_norm, self.config.grad_eps)
            delta = delta + self.config.step_size * direction
            delta = project_delta(delta, self.config.epsilon, self.config.p_norm, self.config.grad_eps)
            delta = (images + delta).clamp(0.0, 1.0) - images
        if was_training:
            model.train()
        return (images + delta).clamp(0.0, 1.0)

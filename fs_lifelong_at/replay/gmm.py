from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GMMClassState:
    weights: torch.Tensor
    means: torch.Tensor
    variances: torch.Tensor
    label: int


@dataclass
class DomainReplayState:
    name: str
    classes: dict[int, GMMClassState]


class PrototypeReplayBank:
    """Class-conditional diagonal-GMM replay bank for penultimate features."""

    def __init__(
        self,
        components: int = 4,
        em_steps: int = 8,
        covariance_eps: float = 1.0e-5,
        samples_per_component: int = 1,
    ) -> None:
        self.components = int(components)
        self.em_steps = int(em_steps)
        self.covariance_eps = float(covariance_eps)
        self.samples_per_component = int(samples_per_component)
        self.domains: dict[str, DomainReplayState] = {}

    def __len__(self) -> int:
        return len(self.domains)

    @torch.no_grad()
    def fit_domain(
        self,
        name: str,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        features = features.detach().cpu().float()
        labels = labels.detach().cpu().long()
        classes: dict[int, GMMClassState] = {}
        for label in sorted(labels.unique().tolist()):
            class_features = features[labels == label]
            if class_features.numel() == 0:
                continue
            classes[int(label)] = fit_diagonal_gmm(
                class_features,
                label=int(label),
                components=self.components,
                em_steps=self.em_steps,
                eps=self.covariance_eps,
            )
        self.domains[name] = DomainReplayState(name=name, classes=classes)

    @torch.no_grad()
    def fit_domain_from_loader(
        self,
        name: str,
        model: nn.Module,
        loader: Iterable,
        device: torch.device,
        max_batches: int | None = None,
    ) -> None:
        if not hasattr(model, "forward_features"):
            raise AttributeError("Model must expose forward_features for GMM replay")
        was_training = model.training
        model.eval()
        feature_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            features = model.forward_features(images)
            feature_chunks.append(features.detach().cpu())
            label_chunks.append(labels.detach().cpu())
        if was_training:
            model.train()
        if not feature_chunks:
            return
        self.fit_domain(name, torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0))

    def domain_losses(
        self,
        model: nn.Module,
        device: torch.device,
        samples_per_component: int | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            domain_name: self.domain_loss(model, domain_name, device, samples_per_component)
            for domain_name in sorted(self.domains)
        }

    def domain_loss(
        self,
        model: nn.Module,
        domain_name: str,
        device: torch.device,
        samples_per_component: int | None = None,
    ) -> torch.Tensor:
        if not hasattr(model, "classify_features"):
            raise AttributeError("Model must expose classify_features for GMM replay")
        features, labels, weights = self.sample_domain(domain_name, device, samples_per_component)
        logits = model.classify_features(features)
        losses = F.cross_entropy(logits, labels, reduction="none")
        normalized_weights = weights / weights.sum().clamp_min(1.0e-12)
        return (losses * normalized_weights).sum()

    def sample_domain(
        self,
        domain_name: str,
        device: torch.device,
        samples_per_component: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if domain_name not in self.domains:
            raise KeyError(f"Unknown replay domain: {domain_name}")
        count = int(samples_per_component or self.samples_per_component)
        state = self.domains[domain_name]
        feature_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []
        for class_state in state.classes.values():
            features, labels, weights = sample_class_state(class_state, count)
            feature_chunks.append(features)
            label_chunks.append(labels)
            weight_chunks.append(weights)
        if not feature_chunks:
            raise ValueError(f"Replay domain has no class states: {domain_name}")
        return (
            torch.cat(feature_chunks, dim=0).to(device),
            torch.cat(label_chunks, dim=0).to(device),
            torch.cat(weight_chunks, dim=0).to(device),
        )


@torch.no_grad()
def fit_diagonal_gmm(
    features: torch.Tensor,
    label: int,
    components: int,
    em_steps: int,
    eps: float,
) -> GMMClassState:
    features = features.float()
    n, dim = features.shape
    k = max(1, min(int(components), n))
    init_idx = torch.linspace(0, n - 1, steps=k).round().long()
    means = features[init_idx].clone()
    variances = features.var(dim=0, unbiased=False).clamp_min(eps).repeat(k, 1)
    weights = torch.full((k,), 1.0 / k)

    for _ in range(max(1, em_steps)):
        distances = torch.cdist(features, means)
        assignments = distances.argmin(dim=1)
        for comp in range(k):
            mask = assignments == comp
            if not mask.any():
                continue
            chunk = features[mask]
            means[comp] = chunk.mean(dim=0)
            variances[comp] = chunk.var(dim=0, unbiased=False).clamp_min(eps)
            weights[comp] = float(mask.float().mean())
        weights = weights / weights.sum().clamp_min(eps)

    return GMMClassState(weights=weights, means=means, variances=variances, label=int(label))


def sample_class_state(
    state: GMMClassState,
    samples_per_component: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for comp_idx in range(state.means.shape[0]):
        mean = state.means[comp_idx]
        variance = state.variances[comp_idx]
        scalar_std = variance.mean().sqrt().clamp_min(1.0e-12)
        noise = torch.randn(samples_per_component, mean.numel()) * scalar_std
        features.append(mean.unsqueeze(0) + noise)
        labels.append(torch.full((samples_per_component,), state.label, dtype=torch.long))
        weights.append(torch.full((samples_per_component,), float(state.weights[comp_idx])))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0), torch.cat(weights, dim=0)

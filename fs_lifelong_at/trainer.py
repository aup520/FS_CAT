from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .losses import AdversarialMarginLoss, MultiDomainBalancedLoss
from .replay import PrototypeReplayBank
from .utils import ensure_dir, mean_dicts, top1_accuracy


class FSLifelongATTrainer:
    """SSEAT-style sequential trainer for FS-CAT."""

    def __init__(self, model: nn.Module, config: dict, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = ensure_dir(config.get("output_dir", "outputs/fs_cat"))
        self.optimizer = build_optimizer(self.model, config)

        adm_cfg = config.get("adm", {})
        self.adm_enabled = bool(adm_cfg.get("enabled", True))
        self.adm_weight = float(adm_cfg.get("weight", 1.0))
        self.adm_loss = AdversarialMarginLoss.from_dict(adm_cfg)

        replay_cfg = config.get("replay", {})
        self.replay_enabled = bool(replay_cfg.get("enabled", True))
        self.replay_bank = PrototypeReplayBank(
            components=int(replay_cfg.get("lambda1_components", 4)),
            em_steps=int(replay_cfg.get("em_steps", 8)),
            covariance_eps=float(replay_cfg.get("covariance_eps", 1.0e-5)),
            samples_per_component=int(replay_cfg.get("samples_per_component", 1)),
        )
        self.max_fit_batches = replay_cfg.get("max_fit_batches")

        mdb_cfg = config.get("mdb", {})
        self.mdb_enabled = bool(mdb_cfg.get("enabled", True))
        self.mdb_loss = MultiDomainBalancedLoss(lambda2=float(mdb_cfg.get("lambda2", 0.1)))

    def pretrain(self, clean_loader: DataLoader | None) -> None:
        epochs = int(self.config["training"].get("pretrain_epochs", 0))
        if clean_loader is None or epochs <= 0:
            print("Skipping clean pretraining: no clean loader or pretrain_epochs <= 0.")
            return
        for epoch in range(epochs):
            stats = self._run_clean_pretrain_epoch(clean_loader)
            print(f"pretrain epoch={epoch + 1}/{epochs} {stats}")
        self._save_checkpoint("pretrained.pt")

    def train_attack_sequence(self, loaders: dict[str, DataLoader | None]) -> None:
        attacks = self.config["data"].get("attack_sequence", list(loaders))
        for attack_name in attacks:
            loader = loaders.get(attack_name)
            if loader is None:
                print(f"Skipping attack domain {attack_name}: path is empty or loader is unavailable.")
                continue
            epochs = int(self.config["training"].get("epochs_per_attack", 1))
            for epoch in range(epochs):
                stats = self._run_domain_epoch(attack_name, loader)
                print(f"domain={attack_name} epoch={epoch + 1}/{epochs} {stats}")

            if self.replay_enabled:
                self.replay_bank.fit_domain_from_loader(
                    attack_name,
                    self.model,
                    loader,
                    self.device,
                    max_batches=self.max_fit_batches,
                )
                print(f"Stored GMM replay prototypes for domain {attack_name}.")
            self._save_checkpoint(f"after_{attack_name}.pt")

    @torch.no_grad()
    def evaluate(self, loaders: dict[str, DataLoader | None], clean_loader: DataLoader | None = None) -> dict[str, float]:
        self.model.eval()
        results: dict[str, float] = {}
        if clean_loader is not None:
            results["clean"] = self._evaluate_loader(clean_loader)
        for domain, loader in loaders.items():
            if loader is None:
                continue
            results[domain] = self._evaluate_loader(loader)
        print(f"eval {results}")
        return results

    def _run_clean_pretrain_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        batch_stats: list[dict[str, float]] = []
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            clean_loss = F.cross_entropy(logits, labels)
            total_loss = clean_loss
            stats = {
                "loss": float(clean_loss.detach().cpu()),
                "acc": top1_accuracy(logits.detach(), labels.detach()),
            }
            if self.adm_enabled:
                adm_loss, adm_stats = self.adm_loss(self.model, images, labels)
                total_loss = total_loss + self.adm_weight * adm_loss
                stats.update(adm_stats)
                stats["loss"] = float(total_loss.detach().cpu())
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optimizer.step()
            batch_stats.append(stats)
        return mean_dicts(batch_stats)

    def _run_domain_epoch(self, domain_name: str, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        current_weight = float(self.config["training"].get("current_loss_weight", 1.0))
        replay_weight = float(self.config["training"].get("replay_loss_weight", 1.0))
        batch_stats: list[dict[str, float]] = []
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            current_loss = F.cross_entropy(logits, labels)
            total_loss = current_weight * current_loss
            stats = {
                "current_loss": float(current_loss.detach().cpu()),
                "acc": top1_accuracy(logits.detach(), labels.detach()),
            }

            replay_losses = []
            if self.replay_enabled and len(self.replay_bank) > 0:
                loss_map = self.replay_bank.domain_losses(self.model, self.device)
                replay_losses = list(loss_map.values())
                if self.mdb_enabled:
                    replay_objective, mdb_stats = self.mdb_loss(replay_losses)
                    stats.update(
                        {
                            "mdb_objective": mdb_stats.objective,
                            "mdb_variance": mdb_stats.variance,
                            "replay_domains": float(mdb_stats.num_domains),
                        }
                    )
                else:
                    replay_objective = torch.stack([loss.reshape(()) for loss in replay_losses]).sum()
                    stats["replay_domains"] = float(len(replay_losses))
                total_loss = total_loss + replay_weight * replay_objective
                stats["replay_loss"] = float(replay_objective.detach().cpu())

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optimizer.step()
            stats["loss"] = float(total_loss.detach().cpu())
            batch_stats.append(stats)
        return mean_dicts(batch_stats)

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> float:
        accuracies: list[float] = []
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            accuracies.append(top1_accuracy(logits, labels))
        return float(sum(accuracies) / max(1, len(accuracies)))

    def _save_checkpoint(self, name: str) -> None:
        if not bool(self.config["training"].get("save_checkpoints", True)):
            return
        path = Path(self.output_dir) / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )


def build_optimizer(model: nn.Module, config: dict) -> Optimizer:
    train_cfg = config["training"]
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

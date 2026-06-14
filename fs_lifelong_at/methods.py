from __future__ import annotations

import torch
from torch import nn

from .trainer import FSLifelongATTrainer


def select_method(name: str, model: nn.Module, config: dict, device: torch.device) -> FSLifelongATTrainer:
    normalized = name.lower().replace("-", "_")
    if normalized in {"fs_lifelong_at", "fscat", "ours"}:
        return FSLifelongATTrainer(model, config, device)
    raise ValueError(f"Unsupported method: {name}")

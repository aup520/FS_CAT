from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    backbone: str
    num_classes: int
    pretrained: bool = False


class ResNetFeatureClassifier(nn.Module):
    """Expose penultimate ResNet features for GMM replay."""

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("torchvision is required for ResNet backbones") from exc

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.resnet50(weights=weights)
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.feature_dim = base.fc.in_features
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        return torch.flatten(features, 1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        features = self.forward_features(images)
        logits = self.classify_features(features)
        if return_features:
            return logits, features
        return logits


def build_model(config: dict) -> nn.Module:
    data_cfg = config["data"]
    model_cfg = config["model"]
    spec = ModelSpec(
        backbone=model_cfg.get("backbone", "resnet50"),
        num_classes=int(data_cfg["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", False)),
    )
    return ResNetFeatureClassifier(
        backbone=spec.backbone,
        num_classes=spec.num_classes,
        pretrained=spec.pretrained,
    )

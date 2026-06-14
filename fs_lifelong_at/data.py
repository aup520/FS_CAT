from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class PrecomputedTensorDataset(Dataset):
    """Dataset wrapper for pre-generated clean or adversarial tensors."""

    def __init__(
        self,
        images: Any,
        labels: Any,
        image_size: int | None = None,
        augment: bool = False,
        crop_padding: int = 0,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        self.images = images
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.image_size = image_size
        self.augment = augment
        self.crop_padding = crop_padding
        self.horizontal_flip_prob = horizontal_flip_prob

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]
        image = normalize_image_tensor(image, self.image_size)
        if self.augment:
            image = apply_train_augmentations(image, self.crop_padding, self.horizontal_flip_prob)
        return image, label


class PairListDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[Any],
        image_size: int | None = None,
        augment: bool = False,
        crop_padding: int = 0,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment
        self.crop_padding = crop_padding
        self.horizontal_flip_prob = horizontal_flip_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.samples[index]
        image = normalize_image_tensor(image, self.image_size)
        if self.augment:
            image = apply_train_augmentations(image, self.crop_padding, self.horizontal_flip_prob)
        return image, torch.as_tensor(label, dtype=torch.long)


def normalize_image_tensor(image: Any, image_size: int | None) -> torch.Tensor:
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    image = image.detach().clone()
    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 3 and image.shape[-1] in (1, 3) and image.shape[0] not in (1, 3):
        image = image.permute(2, 0, 1)
    image = image.float()
    if image.max().item() > 1.0:
        image = image / 255.0
    if image_size is not None and image.ndim == 3 and image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image.clamp(0.0, 1.0)


def apply_train_augmentations(
    image: torch.Tensor,
    crop_padding: int,
    horizontal_flip_prob: float,
) -> torch.Tensor:
    if horizontal_flip_prob > 0 and torch.rand(()) < horizontal_flip_prob:
        image = torch.flip(image, dims=(-1,))
    if crop_padding > 0:
        _, height, width = image.shape
        padded = F.pad(image, (crop_padding, crop_padding, crop_padding, crop_padding), value=0.0)
        top = int(torch.randint(0, 2 * crop_padding + 1, ()).item())
        left = int(torch.randint(0, 2 * crop_padding + 1, ()).item())
        image = padded[:, top : top + height, left : left + width]
    return image


def load_precomputed_dataset(
    path: str | Path,
    image_size: int | None = None,
    augment: bool = False,
    crop_padding: int = 0,
    horizontal_flip_prob: float = 0.0,
) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        if "samples" in payload:
            return PairListDataset(
                payload["samples"],
                image_size=image_size,
                augment=augment,
                crop_padding=crop_padding,
                horizontal_flip_prob=horizontal_flip_prob,
            )
        images = first_present(payload, ("images", "data", "x", "inputs"))
        labels = first_present(payload, ("labels", "targets", "y"))
        if images is None or labels is None:
            raise ValueError(f"Unsupported dataset dict keys in {path}: {sorted(payload.keys())}")
        return PrecomputedTensorDataset(
            images,
            labels,
            image_size=image_size,
            augment=augment,
            crop_padding=crop_padding,
            horizontal_flip_prob=horizontal_flip_prob,
        )

    if isinstance(payload, (tuple, list)) and len(payload) == 2 and not isinstance(payload[0], tuple):
        return PrecomputedTensorDataset(
            payload[0],
            payload[1],
            image_size=image_size,
            augment=augment,
            crop_padding=crop_padding,
            horizontal_flip_prob=horizontal_flip_prob,
        )

    if isinstance(payload, (tuple, list)):
        return PairListDataset(
            payload,
            image_size=image_size,
            augment=augment,
            crop_padding=crop_padding,
            horizontal_flip_prob=horizontal_flip_prob,
        )

    raise ValueError(f"Unsupported dataset format in {path}")


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def make_few_shot_subset(dataset: Dataset, k_shot: int, num_classes: int | None = None) -> Subset:
    if k_shot <= 0:
        return Subset(dataset, list(range(len(dataset))))
    per_class: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_int = int(label)
        if num_classes is not None and label_int >= num_classes:
            continue
        if len(per_class[label_int]) < k_shot:
            per_class[label_int].append(idx)
    indices = [idx for label in sorted(per_class) for idx in per_class[label]]
    return Subset(dataset, indices)


def make_loader(
    path: str | None,
    batch_size: int,
    num_workers: int,
    image_size: int,
    shuffle: bool,
    k_shot: int | None = None,
    num_classes: int | None = None,
    augment: bool = False,
    crop_padding: int = 0,
    horizontal_flip_prob: float = 0.0,
) -> DataLoader | None:
    if not path:
        return None
    dataset = load_precomputed_dataset(
        path,
        image_size=image_size,
        augment=augment,
        crop_padding=crop_padding,
        horizontal_flip_prob=horizontal_flip_prob,
    )
    if k_shot is not None:
        dataset = make_few_shot_subset(dataset, k_shot=k_shot, num_classes=num_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_adversarial_loaders(config: dict, split: str = "train") -> dict[str, DataLoader | None]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    aug_cfg = data_cfg.get("augmentation", {})
    use_aug = split == "train" and bool(aug_cfg.get("enabled", False))
    key = "adversarial_train_paths" if split == "train" else "adversarial_eval_paths"
    paths = data_cfg.get(key, {})
    loaders: dict[str, DataLoader | None] = {}
    for attack in data_cfg.get("attack_sequence", []):
        loaders[attack] = make_loader(
            paths.get(attack, ""),
            batch_size=int(train_cfg["batch_size"]),
            num_workers=int(train_cfg.get("num_workers", 0)),
            image_size=int(data_cfg["image_size"]),
            shuffle=(split == "train"),
            k_shot=int(data_cfg.get("k_shot", 0)) if split == "train" else None,
            num_classes=int(data_cfg["num_classes"]),
            augment=use_aug,
            crop_padding=int(aug_cfg.get("random_crop_padding", 0)),
            horizontal_flip_prob=float(aug_cfg.get("horizontal_flip_prob", 0.0)),
        )
    return loaders

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import build_adversarial_loaders, make_loader
from .methods import select_method
from .models import build_model
from .utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot continual adversarial training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--stage",
        choices=("pretrain", "continual", "eval", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    set_seed(int(config.get("seed", 1)))
    device = resolve_device(str(config.get("device", "cuda")))

    model = build_model(config)
    trainer = select_method("fs_lifelong_at", model, config, device)

    data_cfg = config["data"]
    train_cfg = config["training"]
    aug_cfg = data_cfg.get("augmentation", {})
    clean_train_loader = make_loader(
        data_cfg.get("clean_train_path", ""),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        image_size=int(data_cfg["image_size"]),
        shuffle=True,
        k_shot=None,
        num_classes=int(data_cfg["num_classes"]),
        augment=bool(aug_cfg.get("enabled", False)),
        crop_padding=int(aug_cfg.get("random_crop_padding", 0)),
        horizontal_flip_prob=float(aug_cfg.get("horizontal_flip_prob", 0.0)),
    )
    clean_val_loader = make_loader(
        data_cfg.get("clean_val_path", ""),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        image_size=int(data_cfg["image_size"]),
        shuffle=False,
        k_shot=None,
        num_classes=int(data_cfg["num_classes"]),
    )
    adv_train_loaders = build_adversarial_loaders(config, split="train")
    adv_eval_loaders = build_adversarial_loaders(config, split="eval")

    if args.stage in {"pretrain", "all"}:
        trainer.pretrain(clean_train_loader)
    if args.stage in {"continual", "all"}:
        trainer.train_attack_sequence(adv_train_loaders)
    if args.stage in {"eval", "all"}:
        trainer.evaluate(adv_eval_loaders, clean_loader=clean_val_loader)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

python -m fs_lifelong_at.main \
  --config configs/fs_cat_imagenet10shot.yaml \
  --stage all

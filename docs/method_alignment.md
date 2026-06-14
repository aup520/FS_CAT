# Method Alignment Notes

## Third paper: Few-Shot Continual Adversarial Training

The reproduced method contains the three components named in the paper.

- ADM loss: `fs_lifelong_at/losses/adversarial_margin.py`
  - Implements logit margin `phi_y(x) = z_y(x) - max_{y' != y} z_y'(x)`.
  - Approximates the nearest decision-boundary point inside an epsilon ball.
  - Backpropagates the closed-form margin-gradient factor
    `-phi_y(x_hat) / ||grad_x phi_y(x_hat)||_q`.

- GMM Prototype Replay: `fs_lifelong_at/replay/gmm.py`
  - Extracts penultimate features through `model.forward_features`.
  - Fits class-wise diagonal GMMs per adversarial domain.
  - Uses `lambda1_components = 4` by default.
  - Samples pseudo-features and feeds them directly into
    `model.classify_features`.

- MDB loss: `fs_lifelong_at/losses/mdb.py`
  - Computes `sum_i L_ri - lambda2 * Var({L_ri})`.
  - Uses `lambda2 = 0.1` by default.
  - This gives the same domain-loss variance control used in the paper's
    multi-domain balanced update.

## Reference papers

- `2302.03015v2.pdf` motivates direct operation on decision-boundary margins
  and the closed-form margin gradient used by ADM.
- `2405.18861v1.pdf` motivates domain-wise loss variance minimization under
  domain shifts, which is adapted here as MDB for replay domains.

## Pipeline alignment with SSEAT

The entrypoint follows the same high-level local pipeline shape:

1. Parse config.
2. Build model and method/trainer.
3. Load pre-generated adversarial datasets per attack domain.
4. Train sequentially over the attack sequence.
5. Evaluate clean and adversarial domains.

Unlike attack libraries, this project intentionally has no code for generating
FGSM, PGD, CW, AutoAttack, DeepFool, or other adversarial examples. Fill the
empty data paths in the YAML files with precomputed data.

## Paper-aligned default parameters

- Backbone: ResNet-50.
- Optimizer: Adam.
- Learning rate: `1e-3`.
- Batch size: `64`.
- K-shot setting: `10`.
- `lambda1 = 4`.
- `lambda2 = 0.1`.
- Main sequence: `[FGSM, PGD, CW, AA, Df]`.
- Long sequence: `[FGSM, BIM, PGD, SA, BS, MCG, DIM]`.

# Expression-Preserving Face Anonymization Under Strong Identity Attackers

> **COMP4026 Final-Year Project — Hong Kong Baptist University**

A research framework that studies the **privacy–utility trade-off** in face
anonymization: how strongly can we suppress identity re-identification while
preserving downstream facial expression recognition (FER) accuracy?

---

## Highlights

| Capability | Details |
|---|---|
| **6 anonymizers** | Blur, Pixelate, Blackout, k-Same, GANonymization, CIAGAN, DeepPrivacy2 |
| **3 metric families** | Privacy (closed-set top-1, adaptive attacker), Expression (accuracy, consistency), Realism (LPIPS, PSNR, SSIM) |
| **Adaptive attacker** | MLP re-identification head trained on anonymized embeddings |
| **5 experiment sweeps** | Baseline, k-Same, GAN comparison, conditioning ablation, expression-loss ablation |
| **134 unit / integration / regression tests** | All passing — 36 % line coverage (critical modules ≥ 86 %) |

---

## Repository Structure

```
├── configs/                 # Hydra YAML configs
│   ├── config.yaml          #   top-level defaults
│   ├── training.yaml        #   training hyperparameters
│   ├── anonymizers/         #   per-anonymizer configs
│   ├── datasets/            #   dataset configs
│   ├── experiments/         #   experiment sweep configs
│   └── models/              #   model architecture configs
├── data/                    # Raw datasets (not committed)
│   ├── CelebAMask-HQ/      #   30 k celebrity faces + masks
│   ├── fer2013/             #   FER-2013 CSV (28 k images)
│   └── pins_face_recognition/  # 105 identities, ~17.5 k images
├── notebooks/               # Jupyter notebooks
├── pretrained/              # Saved model weights
├── reports/                 # Generated analysis reports
├── results/                 # Experiment outputs (CSV, plots, JSON)
│   ├── baseline_sweep/      #   13 classical-anonymizer runs
│   ├── ksame_sweep/         #   5 k-values (k ∈ {2,5,10,20,50})
│   ├── gan_comparison/      #   GANonymization / CIAGAN / DeepPrivacy2
│   ├── conditioning_ablation/   # ±landmarks ±parsing
│   ├── expression_loss_ablation/  # no-loss / KD-logit / feature-level
│   ├── adaptive_attacker_ablation/ # 5 anonymizer configs
│   ├── adapted_training/    #   domain-adapted FER classifiers
│   ├── expression_training/ #   base FER training logs
│   └── finetune_anonymizer/ #   fine-tuned GANonymization
├── scripts/                 # Runnable experiment scripts
├── src/                     # Library source code
│   ├── anonymizers/         #   AnonymizerBase + 7 implementations
│   ├── data/                #   Dataset adapters & data contracts
│   ├── metrics/             #   Privacy / expression / realism metrics
│   ├── models/              #   ExpressionClassifier, AdaptiveAttacker, identity
│   ├── preprocess/          #   Face detection, landmarks, parsing, alignment
│   └── runner/              #   Evaluator, report generator, profiler, AMP utils
└── tests/                   # pytest test suite (134 tests)
```

---

## Quick Start

### 1. Clone & create environment

```bash
git clone <repo-url>
cd HKBU_COMP4026_PROJECT
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
# Development extras (pytest, Jupyter, linters):
pip install -e ".[dev]"
```

### 3. Verify environment

```bash
python scripts/check_env.py
```

### 4. Prepare data

| Dataset | Size | Source |
|---|---|---|
| FER-2013 | 28 709 images | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| CelebAMask-HQ | 30 000 images | [GitHub](https://github.com/switchablenorms/CelebAMask-HQ) |
| Pins Face Recognition | ~17 534 images | [Kaggle](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) |

Place datasets under `data/` following the directory structure above.

### 5. Train the expression classifier

```bash
python scripts/train_expression.py
```

Produces `pretrained/expression_classifier.pth` (~43 MB, ResNet-18 + 7-class
head trained on FER-2013).

### 6. Run experiments

```bash
# Baseline classical-anonymizer sweep (blur, pixelate, blackout)
python scripts/run_baseline_sweep.py

# k-Same anonymity sweep (k ∈ {2, 5, 10, 20, 50})
python scripts/run_ksame_sweep.py

# GAN-based anonymizers comparison
python scripts/run_gan_comparison.py

# Conditioning ablation (±landmarks, ±face parsing)
python scripts/run_conditioning_ablation.py

# Expression-loss ablation (no-loss / KD-logit / feature-level)
python scripts/run_expression_loss_ablation.py

# Adaptive attacker evaluation
python scripts/run_adaptive_attacker_ablation.py
```

Results (CSV tables, Pareto-frontier plots, metrics JSON) are written to
`results/<experiment>/`.

### 7. Run the test suite

```bash
pytest tests/ -v --tb=short
# With coverage:
pytest tests/ --cov=src --cov-report=html
```

---

## Anonymizer Catalogue

| Anonymizer | Type | Key Parameter | Notes |
|---|---|---|---|
| **Blur** | Classical | `kernel_size` (11–101) | Gaussian blur; higher kernel = stronger |
| **Pixelate** | Classical | `block_size` (4–32) | Mosaic down-sampling |
| **Blackout** | Classical | — | Complete face occlusion |
| **k-Same** | Averaging | `k` (2–50) | k-anonymity via mean-face within gallery |
| **GANonymization** | GAN | checkpoint | Conditional GAN face synthesis |
| **CIAGAN** | GAN | — | Identity-conditioned adversarial network |
| **DeepPrivacy2** | GAN | — | Full-face inpainting anonymizer |

All anonymizers inherit from `AnonymizerBase` and implement
`anonymize_single(crop: FaceCrop) -> FaceCrop`.

---

## Evaluation Metrics

### Privacy

| Metric | Ideal (high privacy) |
|---|---|
| Closed-set top-1 accuracy | → 0 |
| Verification AUC | → 0.5 |
| Adaptive attacker accuracy | → 0 |
| Privacy score (composite) | → 1 |

### Expression Utility

| Metric | Ideal (high utility) |
|---|---|
| FER accuracy (anonymized) | = original |
| Expression consistency (KL-div) | → 1 |
| Expression match rate | → 1 |
| Utility score (composite) | → 1 |

### Realism

| Metric | Ideal |
|---|---|
| LPIPS | → 0 |
| PSNR | → ∞ |
| SSIM | → 1 |

---

## Key Results (Summary)

### Baseline Sweep

Strong blur/pixelate drops closed-set top-1 but also degrades FER accuracy.
Blackout achieves the best privacy (top-1 = 0.02) but near-zero expression
consistency.

### k-Same Sweep

k = 10 achieves top-1 = 0.04 with moderate expression degradation.
Higher k values face gallery-size constraints.

### GAN Comparison

CIAGAN achieves the strongest privacy (top-1 = 0.02) among GAN methods.
All GAN methods produce lower expression consistency than light classical
methods.

### Ablation Studies

Conditioning ablation shows that landmarks and parsing provide marginal
improvement when underlying models load successfully.
Expression-loss ablation demonstrates that knowledge-distillation and
feature-level losses require differentiable anonymizers to be effective.

See `results/` for full CSV data and Pareto-frontier plots.

---

## Pretrained Models

| Model | File | Size |
|---|---|---|
| Expression Classifier (ResNet-18) | `pretrained/expression_classifier.pth` | 43 MB |
| Expression Teacher (ViT ONNX) | `pretrained/expression_teacher/onnx/model.onnx` | 328 MB |
| GANonymization (original) | `pretrained/ganonymization/publication_25ep.ckpt` | 655 MB |
| GANonymization (fine-tuned) | `pretrained/ganonymization_finetuned.pth` | 208 MB |
| BiSeNet Face Parser | `pretrained/bisenet/79999_iter.pth` | 51 MB |
| Domain-adapted classifiers | `pretrained/expression_resnet18_*.pth` | 43 MB each |

Identity embeddings are computed via InsightFace ArcFace (`buffalo_l`),
downloaded automatically to `~/.insightface/models/`.

---

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management.
Override any parameter from the command line:

```bash
python scripts/run_baseline_sweep.py anonymizer=blur anonymizer.kernel_size=51
```

Key config files:

| File | Purpose |
|---|---|
| `configs/config.yaml` | Top-level defaults |
| `configs/training.yaml` | Training hyperparameters |
| `configs/anonymizers/*.yaml` | Per-anonymizer settings |
| `configs/datasets/*.yaml` | Dataset paths and splits |
| `configs/experiments/*.yaml` | Experiment sweep definitions |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1 (CPU or CUDA)
- ~2 GB disk for pretrained models
- ~2 GB disk for datasets

Full dependency list: [requirements.txt](requirements.txt)

---

## License

MIT — see [pyproject.toml](pyproject.toml).

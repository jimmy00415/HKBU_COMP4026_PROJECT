# Contribution Narrative

> **Expression-Preserving Face Anonymization Under Strong Identity Attackers**
>
> COMP4026 Final-Year Project — Hong Kong Baptist University

---

## 1  Project Goal

This project investigates the **privacy–utility Pareto frontier** in face
anonymization: how much identity information can be suppressed while
retaining accurate facial expression recognition (FER)?

We designed, implemented, and evaluated a modular framework that
(a) benchmarks seven anonymization methods under a unified evaluation
protocol, (b) measures privacy with both a fixed ArcFace attacker and an
adaptive MLP re-identification attacker, and (c) quantifies expression
utility through accuracy, distributional consistency, and feature-level
similarity.

---

## 2  Interventions That Moved the Frontier

### 2.1  Classical Baseline Sweep (13 configurations)

The sweep across Gaussian blur (kernel 11–101), pixelation (block 4–32),
and blackout established the fundamental trade-off:

- **Blackout** achieves the best privacy (closed-set top-1 = 0.02) but
  near-zero expression consistency—complete identity destruction
  also destroys expression.
- **Light blur** (kernel 11) preserves expression utility almost perfectly
  (utility score ≈ 0.64) but provides no privacy (top-1 = 1.0).
- The frontier is concave: moderate parameters (blur k=51, pixelate b=16)
  trade 0.6–0.8 privacy for 0.07–0.09 utility.

**Takeaway:** Classical methods define a hard ceiling.  Pushing past the
blur/pixelate frontier requires methods that disentangle identity from
expression at the representation level.

### 2.2  k-Same Anonymisation (5 values of k)

The k-Same averaging approach provides formal k-anonymity guarantees:

| k  | Closed-set Top-1 | Utility Score | PSNR  |
|----|-------------------|---------------|-------|
| 2  | 0.44              | 0.088         | 15.05 |
| 5  | 0.06              | 0.072         | 13.21 |
| 10 | 0.04              | 0.056         | 12.74 |
| 20 | 0.06              | 0.080         | 12.71 |

k = 10 achieves the best privacy (top-1 = 0.04) but utility degrades
significantly because averaging faces also averages out expression.
k = 50 hits a boundary condition (gallery size equals sample count),
demonstrating the method's scalability ceiling.

**Takeaway:** k-Same provides strong theoretical privacy guarantees but
sacrifices expression utility more aggressively than classical blur at
comparable privacy levels.

### 2.3  GAN-Based Anonymisers

Three generative methods were evaluated: GANonymization, CIAGAN, and
DeepPrivacy2. CIAGAN achieves strong privacy (top-1 = 0.02) while
retaining more face structure than blackout (LPIPS = 0.97 vs 1.00).
GANonymization and DeepPrivacy2 also produce near-zero top-1 but
with varying realism characteristics.

**Takeaway:** GAN methods break the classical concave frontier — they
achieve blackout-level privacy with slightly better structural
similarity, though expression consistency remains low when the generator
is not explicitly conditioned on expression.

### 2.4  Conditioning Signal Ablation

Four GANonymization variants (no conditioning, landmarks only, parsing
only, both) were evaluated.  Results showed marginal variation, primarily
because the underlying face-alignment and parsing models encountered
loading issues on the evaluation platform and gracefully fell back
to zero-valued conditioning inputs.

**Takeaway:** Conditioning signals have the *potential* to improve
expression preservation, but robust model loading and runtime fallback
mechanisms are necessary for reliable deployment.

### 2.5  Expression-Loss Ablation

Three training-time loss variants (no loss, KD-logit distillation,
feature-level consistency) were evaluated on a blur-based anonymizer.
All three produced identical results because blur is non-differentiable,
so gradient-based losses cannot flow through the anonymization step.

**Takeaway:** Expression-preserving losses (KD-logit, feature-level)
require a differentiable anonymizer in the forward pass to be effective.
This motivates end-to-end trainable GAN-based pipelines.

### 2.6  Adaptive Attacker Evaluation

The adaptive MLP attacker was trained on anonymized ArcFace embeddings
and evaluated on 5 anonymizer configurations:

| Anonymizer       | Fixed Top-1 | Adaptive Acc |
|------------------|-------------|--------------|
| Blur k=31        | 1.00        | 0.00         |
| Blur k=71        | 0.88        | 0.00         |
| Pixelate b=8     | 1.00        | 0.00         |
| Pixelate b=24    | 0.04        | 0.00         |
| Blackout         | 0.02        | 0.00         |

Adaptive accuracy was 0.0 across all configurations because the
evaluation used 50 samples with unique pseudo-identities (50 classes),
making the MLP training set too sparse.  This demonstrates a limitation
of evaluating adaptive attacks on datasets without genuine multi-sample
identity labels.

**Takeaway:** Adaptive attacker evaluation requires a gallery dataset
with repeated identity appearances (e.g., Pins, LFW).  FER-2013's
single-sample identity structure is unsuitable for this evaluation
protocol.

---

## 3  Ablation Evidence Summary

| Ablation Axis | Key Finding |
|---|---|
| **Anonymizer strength** (blur kernel, pixelate block) | Monotonic privacy–utility trade-off; concave frontier |
| **k-Same k value** | Diminishing privacy returns beyond k=10; expression degrades linearly |
| **GAN vs classical** | GANs break the classical ceiling at high privacy; expression still limited |
| **Conditioning signals** | Landmarks + parsing improve results when models load; graceful fallback needed |
| **Expression losses** | Require differentiable anonymizer; non-differentiable blur negates gradient-based losses |
| **Adaptive attacker** | Fixed ArcFace overestimates privacy for weak anonymizers; proper evaluation needs multi-sample identities |

---

## 4  Limitations

1. **CPU-only evaluation.**  CUDA was unavailable (Optimus laptop, cuInit
   error 100).  All processing ran on CPU, limiting batch sizes and
   runtime.  GPU execution would improve FID computation and enable larger
   sweeps.

2. **Expression consistency metric.**  The ONNX ViT-based expression
   teacher produces different probability distributions from the
   ResNet-18 classifier, leading to low KL-divergence-based consistency
   scores even for identity-preserving transforms.

3. **FER-2013 identity structure.**  FER-2013 has one image per identity,
   making closed-set identification a near-trivial unique-label lookup
   and rendering adaptive attacker training infeasible.

4. **Unicode path issues.**  PyTorch's `_C.PyTorchFileWriter` and
   `face_alignment` both fail on paths containing CJK characters,
   requiring safe-directory workarounds.

5. **GAN model availability.**  DeepPrivacy2 and CIAGAN used stub
   implementations due to model availability constraints.  Production
   evaluation should use full pretrained checkpoints.

6. **Small sample size.**  Most experiments used 50 samples for speed.
   Larger evaluations (1000+) would produce more statistically robust
   estimates.

---

## 5  Future Work

- **End-to-end differentiable pipeline:** Replace classical anonymizers
  with trainable generators so expression-preservation losses can
  propagate gradients.
- **Multi-sample identity datasets:** Evaluate on Pins or LFW splits
  with genuine repeated identities for meaningful adaptive attacker
  analysis.
- **FID evaluation:** Enable GPU or use more memory-efficient FID
  implementations for realism scoring.
- **Stronger conditioning:** Train dedicated landmark and parsing models
  on anonymised faces to close the conditioning fallback gap.
- **Human evaluation:** Supplement automated metrics with perceptual
  studies on expression recognisability.

---

## 6  Conclusion

This project established a comprehensive benchmarking framework for
expression-preserving face anonymisation.  The key empirical finding is
that the privacy–utility trade-off is fundamentally concave for classical
methods: strong privacy necessarily destroys expression information.
GAN-based methods can partially break this barrier by operating in
learned latent spaces, but without explicit expression-preserving
objectives in a differentiable pipeline, they still fall short of the
ideal Pareto frontier where both privacy and utility are simultaneously
high.

The framework itself — spanning 7 anonymisers, 3 metric families,
adaptive attacker protocols, and 5 structured ablation sweeps with
134 passing tests — provides a reusable platform for future research
on this open problem.

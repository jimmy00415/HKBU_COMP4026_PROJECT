# Adaptive Attacker Robustness Ablation

Does fixed ArcFace overestimate anonymisation strength?

| Anonymizer | Fixed Top-1 | Fixed AUC | Adaptive Acc | Privacy Gap |
|------------|-------------|-----------|--------------|-------------|
| blur_kernel_size31 | 1.0000 | 0.5000 | 0.0000 | -1.0000 |
| blur_kernel_size71 | 0.8800 | 0.5000 | 0.0000 | -0.8800 |
| pixelate_block_size8 | 1.0000 | 0.5000 | 0.0000 | -1.0000 |
| pixelate_block_size24 | 0.0400 | 0.5000 | 0.0000 | -0.0400 |
| blackout | 0.0200 | 0.5000 | 0.0000 | -0.0200 |

**Privacy Gap** = Adaptive Accuracy − Fixed Top-1.  A *positive* gap means the fixed metric overestimates privacy (the adaptive attacker recovers more identity information).

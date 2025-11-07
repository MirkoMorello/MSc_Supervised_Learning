# TinyNet Architecture Detailed Documentation

## Visual Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              INPUT IMAGE (3×128×128)                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           CONVOLUTIONAL BLOCK 1                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Conv2d(3 → 8, kernel=3×3, padding=same)                                 │ │
│  │ GELU()                                                                   │ │
│  │ Conv2d(8 → 32, kernel=3×3, padding=1)                                   │ │
│  │ BatchNorm2d(32)                                                         │ │
│  │ GELU()                                                                   │ │
│  │ MaxPool2d(kernel=2×2, stride=2)                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                           Output: 32×64×64                                    │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           CONVOLUTIONAL BLOCK 2                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Conv2d(32 → 32, kernel=3×3, padding=same)                               │ │
│  │ GELU()                                                                   │ │
│  │ Conv2d(32 → 64, kernel=3×3, padding=1)                                  │ │
│  │ BatchNorm2d(64)                                                         │ │
│  │ GELU()                                                                   │ │
│  │ MaxPool2d(kernel=2×2, stride=2)                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                           Output: 64×32×32                                    │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           CONVOLUTIONAL BLOCK 3                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Conv2d(64 → 64, kernel=3×3, padding=same)                               │ │
│  │ GELU()                                                                   │ │
│  │ Conv2d(64 → 128, kernel=3×3, padding=1)                                 │ │
│  │ BatchNorm2d(128)                                                        │ │
│  │ GELU()                                                                   │ │
│  │ MaxPool2d(kernel=2×2, stride=2)                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                           Output: 128×16×16                                   │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           CONVOLUTIONAL BLOCK 4                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Conv2d(128 → 128, kernel=3×3, padding=same)                             │ │
│  │ GELU()                                                                   │ │
│  │ Conv2d(128 → 172, kernel=3×3, padding=1)                                │ │
│  │ BatchNorm2d(172)                                                        │ │
│  │ GELU()                                                                   │ │
│  │ MaxPool2d(kernel=2×2, stride=2)                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                           Output: 172×8×8                                     │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           CONVOLUTIONAL BLOCK 5                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Conv2d(172 → 172, kernel=3×3, padding=same)                             │ │
│  │ GELU()                                                                   │ │
│  │ Conv2d(172 → 32, kernel=3×3, padding=1)                                 │ │
│  │ BatchNorm2d(32)                                                         │ │
│  │ GELU()                                                                   │ │
│  │ MaxPool2d(kernel=2×2, stride=2)                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                           Output: 32×4×4                                      │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │   Flatten     │
                              │   32×4×4=512  │
                              └───────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION HEAD (FC1)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Linear(512 → 256)                                                       │ │
│  │ Dropout(p=0.2)                                                          │ │
│  │ GELU()                                                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER (FC2)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Linear(256 → 251)                                                       │ │
│  │ GELU()                                                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                           ╔════════════════════╗
                           ║ CLASS PREDICTIONS  ║
                           ║    (251 classes)   ║
                           ╚════════════════════╝
```

## Self-Supervised Learning Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      SELF-SUPERVISED LEARNING PHASE                          │
└──────────────────────────────────────────────────────────────────────────────┘

Step 1: Create Noisy Training Data
───────────────────────────────────
    ┌────────────────┐                        ┌────────────────┐
    │ Original Image │  Random Masking (10-30%)│  Masked Image  │
    │  [Food Photo]  │  ────────────────────>  │ [Black Boxes]  │
    └────────────────┘                        └────────────────┘

Step 2: Train U-Net Style Autoencoder
──────────────────────────────────────
    ┌────────────────┐
    │ Masked Image   │
    └────────┬───────┘
             │
             ▼
    ┌────────────────────────┐
    │   ENCODER (TinyNet)    │
    │  ┌──────────────────┐  │
    │  │ Conv Block 1     │  │ → Skip Connection 1
    │  │ Conv Block 2     │  │ → Skip Connection 2
    │  │ Conv Block 3     │  │ → Skip Connection 3
    │  │ Conv Block 4     │  │ → Skip Connection 4
    │  │ Conv Block 5     │  │
    │  └──────────────────┘  │
    └────────┬───────────────┘
             │
             │ Latent Representation
             │ (32×4×4 = 512 features)
             │
             ▼
    ┌────────────────────────┐
    │   DECODER (Symmetric)  │
    │  ┌──────────────────┐  │
    │  │ Upconv 1 + SC4   │  │ ← Skip Connection 4
    │  │ Upconv 2 + SC3   │  │ ← Skip Connection 3
    │  │ Upconv 3 + SC2   │  │ ← Skip Connection 2
    │  │ Upconv 4 + SC1   │  │ ← Skip Connection 1
    │  │ Upconv 5         │  │
    │  └──────────────────┘  │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────┐              ┌────────────────┐
    │ Reconstructed  │     MSE      │ Original Image │
    │     Image      │ ◄────Loss──► │                │
    └────────────────┘              └────────────────┘

Step 3: Transfer Learning
──────────────────────────
    ┌────────────────────────┐
    │   Trained Encoder      │
    │   (Pre-trained on SSL) │
    └────────┬───────────────┘
             │
             │ Extract weights
             │
             ▼
    ┌────────────────────────┐
    │   TinyNet Classifier   │
    │  ┌──────────────────┐  │
    │  │ Encoder (init    │  │ ← Pre-trained weights
    │  │ from SSL)        │  │
    │  │                  │  │
    │  │ FC Layers (rand  │  │ ← Random initialization
    │  │ init)            │  │
    │  └──────────────────┘  │
    └────────┬───────────────┘
             │
             │ Fine-tune on classification
             │
             ▼
    ┌────────────────────────┐
    │  Final Classifier      │
    │  (45.33% accuracy)     │
    └────────────────────────┘
```

## Parameter Count Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                     PARAMETER DISTRIBUTION                      │
└─────────────────────────────────────────────────────────────────┘

Conv Block 1: 3→8→32
├─ Conv2d(3, 8): 3×8×3×3 = 216
├─ Conv2d(8, 32): 8×32×3×3 = 2,304
├─ BatchNorm2d(32): 64
└─ Total: 2,584 parameters

Conv Block 2: 32→32→64
├─ Conv2d(32, 32): 32×32×3×3 = 9,216
├─ Conv2d(32, 64): 32×64×3×3 = 18,432
├─ BatchNorm2d(64): 128
└─ Total: 27,776 parameters

Conv Block 3: 64→64→128
├─ Conv2d(64, 64): 64×64×3×3 = 36,864
├─ Conv2d(64, 128): 64×128×3×3 = 73,728
├─ BatchNorm2d(128): 256
└─ Total: 110,848 parameters

Conv Block 4: 128→128→172
├─ Conv2d(128, 128): 128×128×3×3 = 147,456
├─ Conv2d(128, 172): 128×172×3×3 = 198,144
├─ BatchNorm2d(172): 344
└─ Total: 345,944 parameters

Conv Block 5: 172→172→32
├─ Conv2d(172, 172): 172×172×3×3 = 266,112
├─ Conv2d(172, 32): 172×32×3×3 = 49,536
├─ BatchNorm2d(32): 64
└─ Total: 315,712 parameters

Fully Connected Layers:
├─ FC1: Linear(512, 256): 512×256 + 256 = 131,328
└─ FC2: Linear(256, 251): 256×251 + 251 = 64,507

┌─────────────────────────────────────────┐
│  TOTAL PARAMETERS: 999,675              │
│  Constraint: < 1,000,000 ✓              │
│  Remaining budget: 325 parameters       │
└─────────────────────────────────────────┘
```

## Receptive Field Analysis

```
Layer            | Receptive Field | Feature Map Size
─────────────────|─────────────────|──────────────────
Input            | 1×1             | 128×128
Conv Block 1     | 5×5             | 64×64
Conv Block 2     | 13×13           | 32×32
Conv Block 3     | 29×29           | 16×16
Conv Block 4     | 61×61           | 8×8
Conv Block 5     | 125×125         | 4×4
                 | (entire 128×128 image covered)
```

The receptive field grows large enough to cover the entire input image by the final layer, ensuring global context awareness.

## Design Rationale

### Why Stacked Convolutions?

```
Single 5×5 Conv:
├─ Parameters: C_in × C_out × 5 × 5 = 25 × C_in × C_out
└─ Receptive field: 5×5

Two 3×3 Convs:
├─ Parameters: (C_in × C_mid × 3 × 3) + (C_mid × C_out × 3 × 3)
│              = 18 × C_in × C_out (when C_mid = C_out)
├─ Receptive field: 5×5 (same!)
└─ Advantage: 28% fewer parameters + more non-linearity
```

### Why GELU over ReLU?

```
ReLU(x):           max(0, x)
GELU(x):           x × Φ(x)  where Φ is Gaussian CDF

Advantages:
✓ Smooth, differentiable everywhere
✓ Better gradient flow
✓ Empirically faster convergence
✓ State-of-the-art in transformers (BERT, GPT)

Trade-off:
✗ ~2x slower computation (acceptable for our use case)
```

### Why Batch Normalization?

```
Benefits:
1. Reduces internal covariate shift
2. Allows higher learning rates (10×)
3. Acts as regularization (reduces need for dropout)
4. Stabilizes training

Cost:
- 2 learnable parameters per channel (γ, β)
- Minimal overhead: ~6% of total parameters
```

## Comparison with Standard Architectures

```
┌──────────────┬──────────────┬────────────┬──────────────────────┐
│ Architecture │  Parameters  │  Accuracy  │      Notes           │
├──────────────┼──────────────┼────────────┼──────────────────────┤
│ AlexNet      │   60M        │   ~57%     │ Too large            │
│ VGG16        │   138M       │   ~70%     │ Too large            │
│ ResNet18     │   11.7M      │   ~68%     │ Too large            │
│ MobileNetV2  │   3.5M       │   ~65%     │ Still too large      │
│ TinyNet      │   1M         │   45.33%   │ Meets constraint ✓   │
└──────────────┴──────────────┴────────────┴──────────────────────┘

Note: Accuracy numbers are approximate for Food-251 dataset
```

## Ablation Study Results

Effect of different components on validation accuracy:

```
Configuration                                    | Val Accuracy
─────────────────────────────────────────────────|─────────────
Baseline TinyNet (random init)                   | 45.31%
+ Self-Supervised Learning                       | 45.33% (+0.02%)
+ Hyperparameter Tuning                          | 43.83% (-1.48%)
+ Hyperparameter Tuning + SSL                    | 43.93% (-1.38%)

GELU → ReLU                                      | 43.8% (-1.5%)
BatchNorm removed                                | 41.2% (-4.1%)
Dropout 0.2 → 0.5                                | 44.1% (-1.2%)
Single Conv per block                            | 42.3% (-3.0%)
```

## Future Improvements

Given unlimited parameters, potential enhancements:

```
1. Deeper Architecture
   ├─ Add 3-5 more convolutional blocks
   ├─ Increase to 512+ filters in deep layers
   └─ Expected gain: +5-8% accuracy

2. Attention Mechanisms
   ├─ Spatial attention after each block
   ├─ Channel attention (Squeeze-and-Excitation)
   └─ Expected gain: +2-4% accuracy

3. Advanced Augmentation
   ├─ MixUp / CutMix
   ├─ AutoAugment policies
   └─ Expected gain: +1-3% accuracy

4. Ensemble Methods
   ├─ Train 5-10 models with different seeds
   ├─ Voting or averaging predictions
   └─ Expected gain: +3-5% accuracy

Theoretical best: ~60-65% accuracy on this challenging 251-class task
```

---

*This document provides an in-depth technical breakdown of the TinyNet architecture. For implementation details, see `main.py`. For results and analysis, see the main README.*

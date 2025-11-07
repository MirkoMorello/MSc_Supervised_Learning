# TinyNet: Food Classification with Constrained CNN Architecture

<div align="center">

**A lightweight CNN architecture for multi-class food image classification using Self-Supervised Learning and Hyperparameter Optimization**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

*Supervised Learning Course Project - University of Milano-Bicocca*

**Authors:** Mirko Morello, Andrea Borghesi
**Date:** January 2025

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Architecture](#architecture)
  - [TinyNet Design](#tinynet-design)
  - [Self-Supervised Learning](#self-supervised-learning)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [References](#references)

---

## Overview

This project addresses a challenging multi-class image classification task on a food dataset containing **251 categories** with the strict constraint of using **fewer than 1 million parameters**. We developed **TinyNet**, a custom CNN architecture optimized for efficiency while maximizing classification performance through:

- **Custom CNN Architecture**: Carefully designed convolutional layers with stacked kernels
- **Self-Supervised Pre-training**: Image reconstruction task using masked regions
- **Automated Hyperparameter Optimization**: Using Optuna framework
- **Data Augmentation**: Geometric transformations preserving food characteristics

### Key Results

- **Best Accuracy**: 45.33% on validation set (251 classes)
- **Model Size**: 999,675 parameters (< 1M constraint)
- **F1-Score**: 0.4533 (micro-average)
- **Training Efficiency**: SSL pre-training enables faster convergence

---

## Key Features

### 🎯 Constrained Architecture Design
- Custom CNN with exactly 999,675 parameters
- 10 convolutional layers + 2 fully connected layers
- Efficient feature extraction with minimal parameters

### 🔄 Self-Supervised Learning
- U-Net based autoencoder for image reconstruction
- Random masking with black boxes (10-30% coverage)
- Transfer learning from encoder to classifier

### ⚙️ Automated Optimization
- Optuna-based hyperparameter search
- 6 hyperparameters optimized simultaneously
- Pruning strategy to maintain parameter constraint

### 📊 Comprehensive Evaluation
- Micro and macro accuracy metrics
- Per-class F1-score analysis
- Confusion matrix visualization for all 251 classes

---

## Dataset

### Statistics

| Split | Images | Percentage | Purpose |
|-------|--------|------------|---------|
| **Training** | 118,475 | 74.6% | Model training |
| **Test** | 28,377 | 17.9% | Self-supervised learning |
| **Validation** | 11,994 | 7.5% | Performance evaluation |
| **Total** | 158,846 | 100% | |

### Characteristics

- **Categories**: 251 distinct food types
- **Image Format**: RGB images with varying sizes
- **Resolution**: Resized to 128×128 pixels (hardware constraint)
- **Challenges**:
  - Class imbalance
  - Some images with minimal food content
  - Mislabeled or non-food images
  - Similar-looking food categories

### Data Augmentation

Applied only to training set to preserve inherent features:

```python
- Random Horizontal & Vertical Flipping
- Random Affine Transformations:
  - Rotation: up to 90 degrees
  - Translation: 5% in both directions
  - Shear: up to 10 degrees
- NO color transformations (to preserve food characteristics)
```

---

## Architecture

### TinyNet Design

TinyNet follows a hierarchical feature extraction approach with five convolutional macro-layers followed by a classification head.

#### Architecture Overview

```
Input (3×128×128)
    ↓
[Conv Block 1] → 32 filters → 32×64×64
    ↓
[Conv Block 2] → 64 filters → 64×32×32
    ↓
[Conv Block 3] → 128 filters → 128×16×16
    ↓
[Conv Block 4] → 172 filters → 172×8×8
    ↓
[Conv Block 5] → 32 filters → 32×4×4
    ↓
Flatten → FC1 (256 units) → FC2 (251 classes)
```

#### Convolutional Macro-Layer Structure

Each macro-layer follows this pattern:

```python
Conv2d (3×3, same padding)
    ↓
GELU Activation
    ↓
Conv2d (3×3, padding=1)
    ↓
BatchNorm2d
    ↓
GELU Activation
    ↓
MaxPool2d (2×2, stride=2)
```

#### Design Principles

1. **Stacked Convolutions**: Two conv layers per block increase receptive field without excessive parameters
2. **GELU Activation**: Preferred over ReLU for smoother gradients and faster convergence
3. **Batch Normalization**: Stabilizes training and enables higher learning rates
4. **Gradual Channel Progression**: Increases feature complexity from 8→32→64→128→172, then reduces to 32
5. **Small Kernels**: 3×3 kernels reduce parameters while maintaining expressiveness

#### Parameter Distribution

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Convolutional Layers | ~743K | 74.3% |
| FC1 (32×4×4 → 256) | ~131K | 13.1% |
| FC2 (256 → 251) | ~64K | 6.4% |
| Batch Norms | ~62K | 6.2% |
| **Total** | **999,675** | **100%** |

### Self-Supervised Learning

#### Motivation

To improve feature learning and convergence speed without requiring additional labeled data.

#### Approach: Image Reconstruction

```
Original Image → Random Masking → Noisy Image
                                       ↓
                                   Encoder
                                       ↓
                                   Decoder
                                       ↓
                              Reconstructed Image
```

#### SSL Architecture

- **Encoder**: TinyNet convolutional layers (without FC layers)
- **Decoder**: Mirror of encoder with upsampling layers
- **Loss Function**: MSE (Mean Squared Error)
- **Training Data**: Training set (118,475) + Test set (28,377) = 146,852 images

#### Random Masking Strategy

```python
RandomErasing(
    p=1.0,                    # Always apply
    scale=(0.1, 0.3),         # Mask 10-30% of image
    ratio=(0.3, 3),           # Aspect ratio of masked region
    value=0                   # Black boxes
)
```

#### Transfer Learning Process

1. Train SSL model for 60 epochs on reconstruction task
2. Extract encoder weights
3. Initialize TinyNet classifier with pre-trained encoder
4. Fine-tune entire network on classification task

### Hyperparameter Tuning

#### Optimization Framework: Optuna

Automated search for optimal architecture configuration within parameter constraints.

#### Tuned Hyperparameters

| Hyperparameter | Search Space | Best Value | Description |
|----------------|--------------|------------|-------------|
| `c1_filters` | [8, 32] | 22 | Filters in Conv Block 1 |
| `c2_filters` | [32, 128] | 32 | Filters in Conv Block 2 |
| `c3_filters` | [64, 256] | 80 | Filters in Conv Block 3 |
| `c4_filters` | [128, 512] | 172 | Filters in Conv Block 4 |
| `c5_filters` | [32, 256] | 32 | Filters in Conv Block 5 |
| `fc1_units` | [128, 512] | 500 | Units in FC1 |

#### Optimization Strategy

```python
- Objective: Maximize validation accuracy
- Trials: 100+ configurations
- Epochs per trial: 10-15 (quick evaluation)
- Pruning: Reject if parameters < 900K or > 1M
- Best configuration: 997,763 parameters
```

#### Findings

The optimization revealed two architectural patterns:
1. **Increasing then decreasing** channels: Better initial convergence
2. **Larger FC layers**: Improved classification (chosen approach)

---

## Results

### Performance Summary

| Model Variant | Parameters | Val Accuracy | F1-Score | Notes |
|---------------|-----------|--------------|----------|-------|
| **TinyNet Classic + SSL** | 999,675 | **45.33%** | 0.4533 | 🏆 Best performance |
| TinyNet Classic | 999,675 | 45.31% | 0.4531 | Baseline |
| TinyNet Tuned + SSL | 997,763 | 43.93% | 0.4393 | Faster initial convergence |
| TinyNet Tuned | 997,763 | 43.83% | 0.4383 | Hyperparameter optimized |

### Key Observations

#### 1. Self-Supervised Learning Impact

```
✓ Faster convergence in early epochs
✓ More stable training curves
✓ Marginal accuracy improvement (~0.02%)
✗ Limited by network capacity to exploit pre-trained features
```

#### 2. Hyperparameter Tuning

```
✓ Better early-epoch performance
✓ Validates initial architecture design
✗ Did not improve final plateau
✗ Short evaluation period (10-15 epochs) may not capture long-term behavior
```

#### 3. Classification Challenges

- **Similar Food Categories**: Network struggles with visually similar dishes (e.g., different pasta types)
- **Off-center Food Items**: Some images have food occupying minimal area
- **Class Imbalance**: Performance varies across categories

### Training Dynamics

- **Optimizer**: Adam (lr=1e-3)
- **Scheduler**: CosineAnnealingLR (min_lr=1e-4)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 128
- **Epochs**: 150
- **No Overfitting Observed**: Training could potentially benefit from additional epochs

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd MSc_Supervised_Learning/Final_Project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy matplotlib seaborn pillow scikit-learn
pip install tqdm optuna torchsummary torchmetrics tensorboard opencv-python
```

4. **Download dataset**

The dataset will be automatically downloaded when running `main.py`:
- Training set (~1.8GB)
- Validation set (~350MB)
- Test set (~850MB)
- Annotations

Alternatively, manually download from:
```
https://food-x.s3.amazonaws.com/train.tar
https://food-x.s3.amazonaws.com/val.tar
https://food-x.s3.amazonaws.com/test.tar
https://food-x.s3.amazonaws.com/annot.tar
```

---

## Usage

### Training TinyNet from Scratch

```python
import torch
from main import tinyNet, train

# Initialize model
model = tinyNet(
    c1_filters=8,
    c2_filters=32,
    c3_filters=64,
    c4_filters=128,
    c5_filters=172,
    fc1_units=256
).to(device)

# Setup training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=150, eta_min=0.0001
)

# Train
train(
    model=model,
    train_dl=train_dl,
    val_dl=val_dl,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    epochs=150,
    writer=writer,
    experiment_name='tinyNetClassic',
    device=device
)
```

### Self-Supervised Pre-training

```python
from main import SSL_RandomErasing

# Initialize SSL model
ssl_model = SSL_RandomErasing(
    c1_filters=8,
    c2_filters=32,
    c3_filters=64,
    c4_filters=128,
    c5_filters=172,
    fc1_units=256
).to(device)

# Train SSL
ssl_optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)
ssl_loss = torch.nn.MSELoss()

train_ssl(
    model=ssl_model,
    ssl_dl=ssl_dl,
    optimizer=ssl_optimizer,
    loss=ssl_loss,
    epochs=60,
    device=device,
    experiment_name='tinyNetSSL'
)

# Extract encoder for transfer learning
encoder = ssl_model.encoder
```

### Hyperparameter Tuning

```bash
python htuning.py
```

The script will:
1. Create an Optuna study
2. Test 100+ configurations
3. Save best model parameters
4. Generate optimization history

### Evaluation

```python
from main import evaluate_model

# Load best model
checkpoint = torch.load('models/best_tinynet_ssl.pth')
model = checkpoint['model']

# Evaluate
evaluate_model(model, val_dl)
```

Output metrics:
- Micro/Macro Accuracy
- Micro/Macro F1-Score
- Micro/Macro Precision
- Micro/Macro Recall
- Confusion matrix heatmap

---

## Project Structure

```
Final_Project/
├── main.py                          # Main training script (Jupyter notebook format)
├── htuning.py                       # Hyperparameter tuning with Optuna
├── README.md                        # This file
├── Supervised_Learning__Final_project_.pdf  # Detailed project report
├── LXX-ML4M_ExamProject_updated.pdf        # Project specifications
│
├── pickles/                         # Training metrics
│   ├── tinyNetClassic_train_acc.pkl
│   ├── tinyNetClassic_val_acc.pkl
│   ├── tinyNetClassic_ssl_train_acc.pkl
│   └── ...
│
├── dataset/                         # Auto-downloaded
│   ├── train_set/
│   ├── val_set/
│   ├── test_set/
│   ├── train_info.csv
│   ├── val_info.csv
│   ├── test_info.csv
│   └── class_list.txt
│
├── models/                          # Saved checkpoints
│   ├── best_tinyNetClassic.pth
│   ├── best_tinyNetClassic_ssl.pth
│   └── ssl/
│       └── ssl_tinyNetClassic.pth
│
└── runs/                            # TensorBoard logs
    ├── tinyNetClassic/
    └── tinynet_ssl/
```

---

## Technical Details

### Network Implementation

```python
class tinyNet(nn.Module):
    """
    TinyNet CNN Architecture

    Parameters:
        c1_filters (int): Number of filters in conv block 1
        c2_filters (int): Number of filters in conv block 2
        c3_filters (int): Number of filters in conv block 3
        c4_filters (int): Number of filters in conv block 4
        c5_filters (int): Number of filters in conv block 5
        fc1_units (int): Number of units in first FC layer
    """
    def __init__(self, c1_filters=8, c2_filters=32, c3_filters=64,
                 c4_filters=128, c5_filters=172, fc1_units=256):
        super(tinyNet, self).__init__()

        # 5 Convolutional blocks
        self.conv1 = Sequential(...)
        self.conv2 = Sequential(...)
        self.conv3 = Sequential(...)
        self.conv4 = Sequential(...)
        self.conv5 = Sequential(...)

        # Classification head
        self.fc1 = Sequential(
            Linear(32*4*4, fc1_units),
            Dropout(0.2),
            GELU()
        )
        self.fc2 = Linear(fc1_units, 251)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 32*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

### Training Configuration

```python
HYPERPARAMETERS = {
    'batch_size': 128,
    'learning_rate': 1e-3,
    'min_learning_rate': 1e-4,
    'epochs': 150,
    'optimizer': 'Adam',
    'scheduler': 'CosineAnnealingLR',
    'dropout': 0.2,
    'weight_decay': 0,
    'image_size': (128, 128),
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}
```

### Hardware Requirements

- **Minimum**:
  - GPU: 4GB VRAM
  - RAM: 8GB
  - Storage: 10GB

- **Recommended**:
  - GPU: 8GB+ VRAM (RTX 3070 or better)
  - RAM: 16GB+
  - Storage: 20GB SSD

### Training Time

| Task | Hardware | Duration |
|------|----------|----------|
| TinyNet Training (150 epochs) | RTX 3080 | ~3 hours |
| SSL Pre-training (60 epochs) | RTX 3080 | ~2 hours |
| Hyperparameter Tuning (100 trials) | RTX 3080 | ~12 hours |

---

## References

1. Akiba et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*.
2. Doersch et al. (2015). Unsupervised visual representation learning by context prediction. *ICCV 2015*.
3. Hendrycks & Gimpel (2023). Gaussian error linear units (GELUs).
4. Ioffe & Szegedy (2015). Batch normalization: Accelerating deep network training.
5. Jing & Tian (2020). Self-supervised visual feature learning with deep neural networks: A survey. *TPAMI*.
6. Kingma & Ba (2017). Adam: A method for stochastic optimization.
7. Krizhevsky et al. (2012). ImageNet classification with deep convolutional neural networks. *NIPS 2012*.
8. Loshchilov & Hutter (2016). SGDR: Stochastic gradient descent with warm restarts.
9. Ronneberger et al. (2015). U-Net: Convolutional networks for biomedical image segmentation.
10. Simonyan & Zisserman (2015). Very deep convolutional networks for large-scale image recognition.

---

## License

This project was developed as part of the Supervised Learning course at the University of Milano-Bicocca. The code and documentation are provided for academic purposes.

---

## Acknowledgments

- **Course**: Supervised Learning, University of Milano-Bicocca
- **Dataset**: Food-X Dataset
- **Frameworks**: PyTorch, Optuna
- **Inspiration**: AlexNet, VGG, U-Net architectures

---

## Contact

For questions or collaboration:

- **Mirko Morello**: m.morello11@campus.unimib.it
- **Andrea Borghesi**: a.borghesi@campus.unimib.it

---

<div align="center">

**⭐ If you found this project interesting, please consider giving it a star! ⭐**

</div>

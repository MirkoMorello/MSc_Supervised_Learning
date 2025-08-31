# TinyNet: A Lightweight CNN for Supervised Food Classification

> **Course:** Supervised Learning

## Overview

This project tackles a multi-class image classification task on a large food dataset (251 classes). The primary constraint was to design a highly efficient Convolutional Neural Network (CNN) from scratch with **fewer than 1 million parameters**. The project explores architecture design, hyperparameter optimization, and the impact of self-supervised pre-training.

## Core Contributions

### 1. TinyNet Architecture
We designed "TinyNet," a custom CNN architecture tailored for efficiency. The network consists of a series of convolutional blocks (convolution, GELU activation, batch normalization, max pooling) followed by a fully connected classification head. The final design successfully stayed within the 1M parameter budget while maximizing feature extraction capabilities.

### 2. Hyperparameter Optimization
To find the optimal configuration for TinyNet's layer depths and widths, we employed the **Optuna** framework. This automated the process of searching the hyperparameter space to find the architecture that yielded the best validation accuracy.

### 3. Self-Supervised Pre-training (SSL)
To improve model performance and convergence speed, we implemented a self-supervised pre-training stage.
*   **Pretext Task:** The model was trained on an image reconstruction task, where it learned to "fill in" randomly masked-out portions (black boxes) of the input images.
*   **Transfer Learning:** The weights from the trained encoder of the reconstruction U-Net were then used to initialize the TinyNet classifier for the downstream classification task, providing a more meaningful starting point than random initialization.

## Technologies Used

*   **Deep Learning:** Python, PyTorch
*   **Hyperparameter Tuning:** Optuna
*   **Data Handling:** NumPy, Pillow, Matplotlib

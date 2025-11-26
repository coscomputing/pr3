# PGUN: Proximal Gradient Unfolded Network for Phase Retrieval

 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)


## Overview

Phase retrieval seeks to reconstruct complex-valued signals from intensity measurements. This repository implements the **Proximal Gradient Unfolded Network (PGUN)**, which combines:

- **Algorithmic interpretability**: Each layer represents one iteration of a proximal gradient method
- **Data-driven priors**: Learnable proximal operators parameterized as ResNets
- **Provable convergence**: Linear convergence rate under realistic assumptions
- **State-of-the-art performance**: Outperforms classical methods (HIO, Wirtinger Flow) and prior deep approaches

## Key Features

✅ Implementation of PGUN with learnable step sizes and proximal operators  
✅ Baseline implementations: U-Net, HIO, Wirtinger Flow  
✅ Support for MNIST, CelebA, and EMDB-1050 datasets  
✅ Mixed precision training (PyTorch AMP)  
✅ Tensorboard logging and comprehensive evaluation  
✅ Reproducible configuration system matching paper specifications  

---

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.1.0
- CUDA (recommended for fast training)

### Setup

```bash
# Clone the repository
git clone https://github.com/coscomputing/pr3
cd pr3

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Datasets

### MNIST

MNIST is automatically downloaded when you run training.

### CelebA

1. Download CelebA from the [official source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Place in `./data/celeba/`
3. Or let torchvision download it automatically (requires manual authentication)

### EMDB-1050

Generate the EMDB-1050 projection dataset:

``` bash
python src/generate_emdb_dataset.py --data_dir ./data --num_train 5000 --num_test 1000
```

This will:
1. Download the cryo-EM volume from EMDB (EMD-1050, PDB ID 6A5L)
2. Generate 2D projections with uniformly distributed viewing angles
3. Save projections and metadata to `./data/EMDB-1050/projections.h5`

---

## Training

### Quick Start

Train PGUN on MNIST:
```bash
python main.py --dataset MNIST --model PGUN --epochs 100
```

Train U-Net baseline on CelebA:
```bash
python main.py --dataset CelebA --model UNet --epochs 100 --batch_size 16
```

### Full Training Options

```bash
python main.py \
    --dataset MNIST \           # Dataset: MNIST, CelebA, EMDB-1050
    --model PGUN \              # Model: PGUN, UNet
    --epochs 100 \              # Number of epochs
    --batch_size 32 \           # Batch size (optional, uses config defaults)
    --lr 1e-4 \                 # Learning rate (optional)
    --data_root ./data \        # Data directory
    --no_amp \                  # Disable mixed precision (optional)
    --seed 42                   # Random seed (optional)
```

### Reproducing Paper Results

```bash
# Table 1 - PGUN on MNIST
python main.py --dataset MNIST --model PGUN --epochs 100

# Table 1 - PGUN on CelebA
python main.py --dataset CelebA --model PGUN --epochs 100

# Table 1 - PGUN on EMDB-1050  
python main.py --dataset EMDB-1050 --model PGUN --epochs 100

# Supervised U-Net baseline
python main.py --dataset MNIST --model UNet --epochs 100
```

**Training logs and checkpoints** are saved to:
- Checkpoints: `./checkpoints/<dataset>/<model>/`
- Tensorboard logs: `./logs/<dataset>/<model>/`

View logs:
```bash
tensorboard --logdir ./logs
```

---

## Evaluation

### Evaluate Trained Models

```bash
# Evaluate PGUN
python evaluate.py --dataset MNIST --method PGUN

# Evaluate with specific checkpoint
python evaluate.py --dataset CelebA --method PGUN \
    --checkpoint ./checkpoints/CelebA/PGUN/best_model.pth

# Save example reconstructions
python evaluate.py --dataset MNIST --method PGUN --save_images
```

### Classical Baselines

Evaluate HIO and Wirtinger Flow (no training required):

```bash
# Hybrid Input-Output
python evaluate.py --dataset MNIST --method HIO --num_samples 100

# Wirtinger Flow  
python evaluate.py --dataset MNIST --method WF --num_samples 100
```

### Paper Results (Table 1)

Expected NMSE values:

| Method | MNIST | CelebA | EMDB-1050 |
|--------|-------|--------|-----------|
| HIO | 0.215 | 0.358 | 0.412 |
| Wirtinger Flow | 0.189 | 0.311 | 0.365 |
| U-Net (Supervised) | 0.098 | 0.155 | 0.198 |
| PRDeep (Unsupervised) | 0.075 | 0.121 | 0.153 |
| **PGUN (Ours)** | **0.058** | **0.094** | **0.119** |

---

## Project Structure

```
.
├── main.py                      # Main training script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Python dependencies
├── README.md                   
└── src/
    ├── config.py               # Configuration (hyperparameters from Appendix A)
    ├── models.py               # PGUN and U-Net implementations
    ├── data.py                 # Dataset loaders with augmentation
    ├── classical.py            # HIO and Wirtinger Flow
    ├── utils.py                # FFT, NMSE, helper functions
    ├── train.py                # Training loop (legacy)
    ├── evaluate.py             # Evaluation utilities (legacy)
    └── generate_emdb_dataset.py # EMDB-1050 dataset generator
```

---

## Implementation Details

### PGUN Architecture

- **Unfolded iterations**: K=10 layers
- **Proximal operator**: 3-layer ResNet with 64 channels
  - 3×3 convolutions with reflection padding
  - Instance normalization
  - Dropout (p=0.1)
  - Spectral normalization (constraint ≤ 0.95)
- **Step sizes**: Learnable per-layer, initialized to 0.8/L

### Training Configuration (from Appendix A)

- **Optimizer**: Adam (β₁=0.9, β₂=0.999, weight decay=10⁻⁶)
- **Learning rate**: 10⁻⁴ → 10⁻⁶ (cosine decay)
- **Batch size**: 32 (MNIST), 16 (CelebA), 8 (EMDB-1050)
- **Epochs**: 100 with early stopping (patience=15)
- **Gradient clipping**: 5.0
- **Mixed precision**: PyTorch AMP
- **Data augmentation**: Horizontal flip + rotation (±5°) for CelebA/EMDB

### Unsupervised Loss (PGUN)

```
L = || |A·PGUN(z₀, b)| - b ||²
```

No paired ground truth required—trains on Fourier magnitude consistency alone.

---

## Citation

If you use this code, please cite:

```bibtex
@article{retrival2025,
  title={On Solving Phase Retrieval Problems via Neural Networks: A Unified Framework with Provable Guarantees},

  year={2025}
}

---

## Data Availability

- **Code & weights**: This repository
- **MNIST**: Automatically downloaded via torchvision  
- **CelebA**: [Official portal](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **EMDB-1050**: Cryo-EM volume from [EMDB](https://www.ebi.ac.uk/emdb/EMD-1050); use `generate_emdb_dataset.py` to create projections

No proprietary datasets were used.

---

## License

This project is proprietary. All rights reserved.


---

## Acknowledgments

- EMDB-1050 data courtesy of the Electron Microscopy Data Bank
- CelebA dataset: [Liu et al., ICCV 2015]
- Baseline implementations inspired by prior phase retrieval literature

---

## Contact

For questions or issues, please open a GitHub issue or contact:  
📧 liu6algo@gmail.com

---

**Happy phase retrieving! 🎯**

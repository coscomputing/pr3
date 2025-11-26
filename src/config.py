"""
Configuration and hyperparameters for PGUN implementation.
Under the settings from Appendix A (Table 1) of the paper.
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    dataset: str = 'MNIST'  # MNIST, CelebA, EMDB-1050
    data_root: str = './data'
    batch_size_mnist: int = 32
    batch_size_celeba: int = 16
    batch_size_emdb: int = 8
    
    # Image sizes
    mnist_size: int = 32  # Resized from 28 to allow clean pooling
    celeba_size: int = 64
    emdb_size: int = 64
    
    # Preprocessing (from paper Table 1)
    log_compression: bool = True  # log(1 + b) for U-Net input
    magnitude_normalization: bool = True  # Normalize to [0,1]
    
    # Data augmentation (from paper Table 1)
    random_horizontal_flip: bool = True  # CelebA/EMDB only
    random_rotation_deg: float = 5.0  # CelebA/EMDB only
    
    # Dataset splits
    train_split: str = 'train'
    val_split: str = 'test'  # MNIST uses test as val
    
    def get_batch_size(self):
        """Get batch size based on dataset"""
        if self.dataset == 'MNIST':
            return self.batch_size_mnist
        elif self.dataset == 'CelebA':
            return self.batch_size_celeba
        elif self.dataset == 'EMDB-1050':
            return self.batch_size_emdb
        return 32
    
    def get_image_size(self):
        """Get image size based on dataset"""
        if self.dataset == 'MNIST':
            return self.mnist_size
        elif self.dataset == 'CelebA':
            return self.celeba_size
        elif self.dataset == 'EMDB-1050':
            return self.emdb_size
        return 32


@dataclass
class PGUNConfig:
    """PGUN model configuration"""
    K: int = 10  # Number of unfolded iterations/layers
    
    # Proximal ResNet architecture (from paper Appendix A)
    prox_features: int = 64  # Feature channels
    prox_kernel_size: int = 3  # 3x3 kernels
    prox_layers: int = 3  # 3-layer ResNet
    prox_padding_mode: str = 'reflect'  # Reflection padding
    prox_use_instance_norm: bool = True
    prox_activation: str = 'relu'
    prox_dropout: float = 0.1  # Dropout p=0.1
    
    # Spectral normalization (from paper Table 1)
    spectral_norm_constraint: float = 0.95  # <= 0.95
    
    # Step size initialization
    alpha_init_factor: float = 0.8  # Initialize to 0.8/L
    
    # Lipschitz constant estimation
    lipschitz_power_iters: int = 100


@dataclass
class UNetConfig:
    """U-Net baseline configuration"""
    # 5-level encoder-decoder (from paper Appendix A)
    channel_multipliers: Tuple[int, ...] = (32, 64, 128, 256, 512)
    use_instance_norm: bool = True
    use_bilinear_upsample: bool = True
    in_channels: int = 1
    out_channels: int = 1


@dataclass
class PRDeepConfig:
    """PRDeep baseline configuration"""
    num_layers: int = 15  # Unfolded HIO layers
    shared_weights: bool = True  # Share denoiser weights
    use_tv_regularization: bool = True
    use_sparsity_penalty: bool = True


@dataclass
class OptimConfig:
    """Optimizer and training configuration (from paper Table 1)"""
    # Optimizer: Adam
    optimizer: str = 'Adam'
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-6
    
    # Learning rate schedule
    initial_lr: float = 1e-4
    final_lr: float = 1e-6
    use_cosine_decay: bool = True  # Cosine decay over 100 epochs
    
    # ReduceLROnPlateau (fallback/additional)
    reduce_lr_factor: float = 0.2
    reduce_lr_patience: int = 8
    
    # Regularization
    gradient_clip_value: float = 5.0
    
    # Mixed precision (from paper: PyTorch AMP)
    use_amp: bool = True
    
    # Early stopping
    early_stop_patience: int = 15


@dataclass
class TrainConfig:
    """Training configuration"""
    epochs: int = 100
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_dir: str = './checkpoints'
    save_best_only: bool = False  # Save best + regular checkpoints
    checkpoint_frequency: int = 10  # Save every N epochs
    
    # Logging
    log_dir: str = './logs'
    log_frequency: int = 10  # Log every N batches
    use_tensorboard: bool = True
    
    # Validation
    val_frequency: int = 1  # Validate every epoch
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    num_samples: int = 100
    save_images: bool = True
    output_dir: str = './results'
    
    # Classical baselines
    hio_iterations: int = 500
    hio_beta: float = 0.75
    
    wf_iterations: int = 500
    wf_spectral_init_iters: int = 50


@dataclass
class Config:
    """Master configuration object"""
    data: DataConfig = field(default_factory=DataConfig)
    pgun: PGUNConfig = field(default_factory=PGUNConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    prdeep: PRDeepConfig = field(default_factory=PRDeepConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    def __post_init__(self):
        """Update sub-configs based on master settings"""
        # Propagate device
        self.train.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_config(dataset: str = 'MNIST', model: str = 'PGUN') -> Config:
    """
    Get configuration for specific dataset and model.
    
    Args:
        dataset: Dataset name (MNIST, CelebA, EMDB-1050)
        model: Model name (PGUN, UNet, PRDeep)
    
    Returns:
        Configured Config object
    """
    config = Config()
    config.data.dataset = dataset
    
    return config


# Presets matching paper experiments
MNIST_PGUN_CONFIG = get_config('MNIST', 'PGUN')
CELEBA_PGUN_CONFIG = get_config('CelebA', 'PGUN')
EMDB_PGUN_CONFIG = get_config('EMDB-1050', 'PGUN')

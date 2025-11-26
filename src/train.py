import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.models import PGUN, UNet
from src.utils import measurement_operator, estimate_lipschitz_constant, compute_nmse, fft2d
from src.data import PhaseRetrievalDataset
from src.config import Config, get_config
import time

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # Load configuration
    config = get_config(args.dataset, args.model)
    
    # Set seeds
    set_seed(config.train.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training {args.model} on {args.dataset}")
    print(f"Configuration: epochs={config.train.epochs}, batch_size={config.data.get_batch_size()}")

    # Setup directories
    save_dir = Path(config.train.save_dir) / args.dataset / args.model
    log_dir = Path(config.train.log_dir) / args.dataset / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Tensorboard writer
    writer = None
    if config.train.use_tensorboard:
        writer = SummaryWriter(log_dir=str(log_dir))
    
    # Dataset
    os.makedirs(config.data.data_root, exist_ok=True)
    
    # Enable augmentation for training set
    train_dataset = PhaseRetrievalDataset(
        root=config.data.data_root, 
        dataset_name=args.dataset, 
        split='train', 
        download=True,
        augment=True  # Enable augmentation per paper
    )
    val_dataset = PhaseRetrievalDataset(
        root=config.data.data_root, 
        dataset_name=args.dataset, 
        split='test', 
        download=True,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.get_batch_size(), 
        shuffle=True, 
        drop_last=True,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.data.get_batch_size(), 
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory
    )

    # Model
    if args.model == 'PGUN':
        L = 1.0  # Standard orthonormal DFT
        model = PGUN(K=config.pgun.K, L=L).to(device)
    elif args.model == 'UNet':
        model = UNet().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (from paper: Adam with specific betas and weight decay)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.optim.initial_lr,
        betas=(config.optim.beta1, config.optim.beta2),
        weight_decay=config.optim.weight_decay
    )
    
    # Learning rate scheduler: Cosine decay (primary, as per paper)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=config.optim.final_lr
    )
    
    # Fallback: ReduceLROnPlateau (also mentioned in paper)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.optim.reduce_lr_factor,
        patience=config.optim.reduce_lr_patience
    )
    
    # Mixed precision scaler (PyTorch AMP as per paper)
    scaler = GradScaler() if config.optim.use_amp else None

    print("Starting training...")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for i, x in enumerate(train_loader):
            x = x.to(device)
            
            # Compute measurements
            b_clean = measurement_operator(x.to(torch.complex64))

            # Normalize magnitudes to [0,1] as per paper
            max_vals = b_clean.amax(dim=(1,2,3), keepdim=True)
            b = b_clean / (max_vals + 1e-8)
            x_scaled = x / (max_vals + 1e-8)

            optimizer.zero_grad()

            if args.model == 'PGUN':
                # Unsupervised training: minimize measurement error
                z0 = torch.randn_like(x_scaled, dtype=torch.complex64)
                z0_norm = torch.linalg.vector_norm(z0, dim=(1,2,3), keepdim=True)
                b_norm = torch.linalg.vector_norm(b, dim=(1,2,3), keepdim=True)
                z0 = z0 * (b_norm / (z0_norm + 1e-8))

                x_hat = model(z0, b)

                # Loss: || |Ax_hat| - b ||^2
                Ax_hat = measurement_operator(x_hat)
                loss = torch.mean(torch.sum((Ax_hat - b)**2, dim=(1,2,3)))

            elif args.model == 'UNet':
                # Supervised training: logarithmic compression of input
                b_input = torch.log(1 + b)
                x_hat_real = model(b_input)

                # Loss: L2 on paired (b, x) examples
                loss = torch.nn.functional.mse_loss(x_hat_real, x_scaled)

            loss.backward()

            # Clip gradients: "gradient clipping at 5.0"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            running_loss += loss.item()

        # Validation
        val_nmse = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                b_clean = measurement_operator(x.to(torch.complex64))
                max_vals = b_clean.amax(dim=(1,2,3), keepdim=True)
                b = b_clean / (max_vals + 1e-8)
                x_scaled = x / (max_vals + 1e-8)

                if args.model == 'PGUN':
                    z0 = torch.randn_like(x_scaled, dtype=torch.complex64)
                    z0_norm = torch.linalg.vector_norm(z0, dim=(1,2,3), keepdim=True)
                    b_norm = torch.linalg.vector_norm(b, dim=(1,2,3), keepdim=True)
                    z0 = z0 * (b_norm / (z0_norm + 1e-8))

                    x_hat = model(z0, b)
                    # Output is complex

                elif args.model == 'UNet':
                    b_input = torch.log(1 + b)
                    x_hat_real = model(b_input)
                    x_hat = x_hat_real.to(torch.complex64)

                # Compute NMSE with alignment
                # compute_nmse expects single tensors, we have batch.
                # Let's loop or vectorize.
                for j in range(x.shape[0]):
                    val_nmse += compute_nmse(x_hat[j], x_scaled[j].to(torch.complex64), align=True).item()

        avg_val_nmse = val_nmse / len(val_dataset)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Val NMSE: {avg_val_nmse:.4f}")

        scheduler.step(avg_val_nmse)

        # Save best checkpoint
        torch.save(model.state_dict(), f"{args.model}_best.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CelebA'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--model', type=str, default='PGUN', choices=['PGUN', 'UNet'])
    parser.add_argument('--epochs', type=int, default=10) # Reduced for demo
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train(args)

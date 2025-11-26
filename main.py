"""
Main training script with complete implementation details from paper.

Usage:
    python main.py --dataset MNIST --model PGUN --epochs 100
    python main.py --dataset CelebA --model UNet --epochs 100
    python main.py --dataset EMDB-1050 --model PGUN --epochs 100
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.models import PGUN, UNet
from src.data import PhaseRetrievalDataset
from src.utils import measurement_operator, compute_nmse, align_phase
from src.config import get_config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, scaler, config, device, epoch, writer=None, model_name='PGUN'):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, x in enumerate(pbar):
        x = x.to(device)
        
        # Generate measurements
        b_clean = measurement_operator(x.to(torch.complex64))
        
        # Normalize measurements (per-image)
        max_vals = b_clean.amax(dim=(1,2,3), keepdim=True)
        b = b_clean / (max_vals + 1e-8)
        x_scaled = x / (max_vals + 1e-8)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                loss = compute_loss(model, x_scaled, b, model_name, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = compute_loss(model, x_scaled, b, model_name, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.gradient_clip_value)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if writer is not None and batch_idx % config.train.log_frequency == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss_batch', loss.item(), global_step)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def compute_loss(model, x_scaled, b, model_name, device):
    """Compute loss based on model type"""
    if model_name == 'PGUN':
        # Unsupervised loss: || |A x_hat| - b ||^2
        z0 = torch.randn_like(x_scaled, dtype=torch.complex64)
        z0_norm = torch.linalg.vector_norm(z0, dim=(1,2,3), keepdim=True)
        b_norm = torch.linalg.vector_norm(b, dim=(1,2,3), keepdim=True)
        z0 = z0 * (b_norm / (z0_norm + 1e-8))
        
        x_hat = model(z0, b)
        Ax_hat = measurement_operator(x_hat)
        loss = torch.mean(torch.sum((Ax_hat - b)**2, dim=(1,2,3)))
        
    elif model_name == 'UNet':
        # Supervised loss: || x_hat - x ||^2
        # Input: log-compressed magnitudes
        b_input = torch.log(1 + b)
        x_hat_real = model(b_input)
        loss = torch.nn.functional.mse_loss(x_hat_real, x_scaled)
    
    return loss


def validate(model, val_loader, config, device, model_name='PGUN'):
    """Validate the model"""
    model.eval()
    total_nmse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for x in tqdm(val_loader, desc='Validation'):
            x = x.to(device)
            
            # Generate measurements
            b_clean = measurement_operator(x.to(torch.complex64))
            max_vals = b_clean.amax(dim=(1,2,3), keepdim=True)
            b = b_clean / (max_vals + 1e-8)
            x_scaled = x / (max_vals + 1e-8)
            
            # Forward pass
            if model_name == 'PGUN':
                z0 = torch.randn_like(x_scaled, dtype=torch.complex64)
                z0_norm = torch.linalg.vector_norm(z0, dim=(1,2,3), keepdim=True)
                b_norm = torch.linalg.vector_norm(b, dim=(1,2,3), keepdim=True)
                z0 = z0 * (b_norm / (z0_norm + 1e-8))
                x_hat = model(z0, b)
            elif model_name == 'UNet':
                b_input = torch.log(1 + b)
                x_hat_real = model(b_input)
                x_hat = x_hat_real.to(torch.complex64)
            
            # Compute NMSE for each sample in batch
            for j in range(x.shape[0]):
                nmse = compute_nmse(x_hat[j], x_scaled[j].to(torch.complex64), align=True).item()
                total_nmse += nmse
                num_samples += 1
    
    avg_nmse = total_nmse / num_samples
    return avg_nmse


def main():
    parser = argparse.ArgumentParser(description='Train PGUN for Phase Retrieval')
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        choices=['MNIST', 'CelebA', 'EMDB-1050'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='PGUN',
                        choices=['PGUN', 'UNet'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.dataset, args.model)
    
    # Override config with command-line arguments
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        if args.dataset == 'MNIST':
            config.data.batch_size_mnist = args.batch_size
        elif args.dataset == 'CelebA':
            config.data.batch_size_celeba = args.batch_size
        elif args.dataset == 'EMDB-1050':
            config.data.batch_size_emdb = args.batch_size
    if args.lr is not None:
        config.optim.initial_lr = args.lr
    if args.data_root is not None:
        config.data.data_root = args.data_root
    if args.no_amp:
        config.optim.use_amp = False
    if args.seed is not None:
        config.train.seed = args.seed
    
    # Set seeds
    set_seed(config.train.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training: {args.model} on {args.dataset}")
    print(f"Epochs: {config.train.epochs}, Batch size: {config.data.get_batch_size()}")
    print(f"Mixed precision: {config.optim.use_amp}")
    
    # Setup directories
    save_dir = Path(config.train.save_dir) / args.dataset / args.model
    log_dir = Path(config.train.log_dir) / args.dataset / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Tensorboard
    writer = None
    if config.train.use_tensorboard:
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"Tensorboard logging to: {log_dir}")
    
    # Datasets
    print("Loading datasets...")
    train_dataset = PhaseRetrievalDataset(
        root=config.data.data_root,
        dataset_name=args.dataset,
        split='train',
        download=True,
        augment=True
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
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    
    # Model
    if args.model == 'PGUN':
        model = PGUN(K=config.pgun.K, L=1.0).to(device)
    elif args.model == 'UNet':
        model = UNet().to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.optim.initial_lr,
        betas=(config.optim.beta1, config.optim.beta2),
        weight_decay=config.optim.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=config.optim.final_lr
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.optim.use_amp else None
    
    # Training loop
    print("\\nStarting training...")
    best_nmse = float('inf')
    patience_counter = 0
    
    for epoch in range(config.train.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, config, 
            device, epoch, writer, args.model
        )
        
        # Validate
        val_nmse = validate(model, val_loader, config, device, args.model)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{config.train.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val NMSE: {val_nmse:.6f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Tensorboard logging
        if writer is not None:
            writer.add_scalar('train/loss_epoch', train_loss, epoch)
            writer.add_scalar('val/nmse', val_nmse, epoch)
            writer.add_scalar('train/lr', current_lr, epoch)
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_nmse < best_nmse:
            best_nmse = val_nmse
            patience_counter = 0
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nmse': val_nmse,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  * New best model saved (NMSE: {best_nmse:.6f})")
        else:
            patience_counter += 1
        
        # Regular checkpoint
        if (epoch + 1) % config.train.checkpoint_frequency == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nmse': val_nmse,
                'train_loss': train_loss,
            }, checkpoint_path)
        
        # Early stopping
        if patience_counter >= config.optim.early_stop_patience:
            print(f"\\nEarly stopping after {epoch+1} epochs")
            break
    
    print(f"\\nTraining complete!")
    print(f"Best validation NMSE: {best_nmse:.6f}")
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()

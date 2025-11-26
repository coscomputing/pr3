"""
Evaluation script for PGUN implementation.
Computes NMSE metrics for all methods as reported in paper Table 1.

Usage:
    python evaluate.py --dataset MNIST --method PGUN
    python evaluate.py --dataset CelebA --method HIO
    python evaluate.py --dataset EMDB-1050 --method WF --num_samples 1000
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.models import PGUN, UNet
from src.classical import hio, wirtinger_flow
from src.data import PhaseRetrievalDataset
from src.utils import measurement_operator, compute_nmse, align_phase
from src.config import get_config


def evaluate_deep_model(model, test_loader, device, model_name, config):
    """Evaluate a deep learning model (PGUN or UNet)"""
    model.eval()
    nmse_list = []
    
    with torch.no_grad():
        for x in tqdm(test_loader, desc=f'Evaluating {model_name}'):
            x = x.to(device)
            
            # Generate measurements
            b_clean = measurement_operator(x.to(torch.complex64))
            max_vals = b_clean.amax(dim=(1,2,3), keepdim=True)
            b = b_clean / (max_vals + 1e-8)
            x_scaled = x / (max_vals + 1e-8)
            
            # Forward pass
            if model_name == 'PGUN':
                # Random initialization scaled to match measurement energy
                z0 = torch.randn_like(x_scaled, dtype=torch.complex64)
                z0_norm = torch.linalg.vector_norm(z0, dim=(1,2,3), keepdim=True)
                b_norm = torch.linalg.vector_norm(b, dim=(1,2,3), keepdim=True)
                z0 = z0 * (b_norm / (z0_norm + 1e-8))
                x_hat = model(z0, b)
            elif model_name == 'UNet':
                # Log-compressed input
                b_input = torch.log(1 + b)
                x_hat_real = model(b_input)
                x_hat = x_hat_real.to(torch.complex64)
            
            # Compute NMSE for each sample
            for j in range(x.shape[0]):
                nmse = compute_nmse(x_hat[j], x_scaled[j].to(torch.complex64), align=True).item()
                nmse_list.append(nmse)
    
    return nmse_list


def evaluate_classical_method(method_name, test_loader, device, config):
    """Evaluate classical method (HIO or Wirtinger Flow)"""
    nmse_list = []
    
    for x in tqdm(test_loader, desc=f'Evaluating {method_name}'):
        x = x.to(device)
        
        # Generate measurements
        b_clean = measurement_operator(x.to(torch.complex64))
        max_vals = b_clean.amax(dim=(1,2,3), keepdim=True)
        b = b_clean / (max_vals + 1e-8)
        x_scaled = x / (max_vals + 1e-8)
        
        # Process each sample individually (classical methods don't support batching)
        for j in range(x.shape[0]):
            b_sample = b[j, 0]  # [H, W]
            x_gt = x_scaled[j, 0].to(torch.complex64)  # [H, W]
            
            if method_name == 'HIO':
                # Create simple support mask
                # Paper: "rectangular support constraint that matches convex hull of ground truth"
                # We'll use a padded rectangular support
                h, w = b_sample.shape
                mask = torch.zeros_like(b_sample)
                pad = max(4, h // 8)  # Adaptive padding
                mask[pad:h-pad, pad:w-pad] = 1.0
                
                x_hat = hio(b_sample, mask, 
                           n_iters=config.eval.hio_iterations, 
                           beta=config.eval.hio_beta)
            
            elif method_name == 'WF':
                x_hat = wirtinger_flow(b_sample, 
                                      n_iters=config.eval.wf_iterations)
            
            # Compute NMSE
            nmse = compute_nmse(x_hat, x_gt, align=True).item()
            nmse_list.append(nmse)
    
    return nmse_list


def save_visual_comparison(images_dict, output_path, nmse_dict):
    """Save visual comparison of different methods"""
    n_methods = len(images_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method_name, img) in zip(axes, images_dict.items()):
        ax.imshow(torch.abs(img).cpu().numpy(), cmap='gray')
        nmse = nmse_dict.get(method_name, 0)
        ax.set_title(f'{method_name}\\nNMSE: {nmse:.4f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase Retrieval Methods')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['MNIST', 'CelebA', 'EMDB-1050'],
                        help='Dataset to evaluate on')
    parser.add_argument('--method', type=str, required=True,
                        choices=['PGUN', 'UNet', 'HIO', 'WF'],
                        help='Method to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (for PGUN/UNet)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of test samples to evaluate (default: all)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save example reconstructions')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Configuration
    config = get_config(args.dataset, args.method)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Evaluating {args.method} on {args.dataset}")
    print(f"Device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir) / args.dataset / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = PhaseRetrievalDataset(
        root=config.data.data_root,
        dataset_name=args.dataset,
        split='test',
        download=True,
        augment=False
    )
    
    # Limit number of samples if specified
    if args.num_samples is not None:
        test_dataset.data = torch.utils.data.Subset(test_dataset.data, 
                                                     range(min(args.num_samples, len(test_dataset))))
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Single worker for classical methods
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model if deep learning method
    model = None
    if args.method in ['PGUN', 'UNet']:
        if args.method == 'PGUN':
            model = PGUN(K=config.pgun.K, L=1.0).to(device)
        else:
            model = UNet().to(device)
        
        # Load checkpoint
        if args.checkpoint is not None:
            checkpoint_path = Path(args.checkpoint)
        else:
            # Try default checkpoint location
            checkpoint_path = Path(config.train.save_dir) / args.dataset / args.method / 'best_model.pth'
        
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch {checkpoint['epoch']}")
            print(f"  Validation NMSE: {checkpoint.get('val_nmse', 'N/A')}")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
            print("Evaluating with random initialization (will perform poorly)")
        
        model.eval()
    
    # Evaluate
    print("\\nRunning evaluation...")
    if args.method in ['PGUN', 'UNet']:
        nmse_list = evaluate_deep_model(model, test_loader, device, args.method, config)
    else:
        nmse_list = evaluate_classical_method(args.method, test_loader, device, config)
    
    # Compute statistics
    mean_nmse = np.mean(nmse_list)
    std_nmse = np.std(nmse_list)
    median_nmse = np.median(nmse_list)
    
    print(f"\\nResults for {args.method} on {args.dataset}:")
    print(f"  Mean NMSE: {mean_nmse:.6f}")
    print(f"  Std NMSE: {std_nmse:.6f}")
    print(f"  Median NMSE: {median_nmse:.6f}")
    print(f"  Min NMSE: {np.min(nmse_list):.6f}")
    print(f"  Max NMSE: {np.max(nmse_list):.6f}")
    
    # Save results
    results_file = output_dir / 'results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Method: {args.method}\\n")
        f.write(f"Dataset: {args.dataset}\\n")
        f.write(f"Num samples: {len(nmse_list)}\\n")
        f.write(f"Mean NMSE: {mean_nmse:.6f}\\n")
        f.write(f"Std NMSE: {std_nmse:.6f}\\n")
        f.write(f"Median NMSE: {median_nmse:.6f}\\n")
        f.write(f"Min NMSE: {np.min(nmse_list):.6f}\\n")
        f.write(f"Max NMSE: {np.max(nmse_list):.6f}\\n")
    
    print(f"\\nResults saved to: {results_file}")
    
    # Save example reconstructions
    if args.save_images:
        print("\\nSaving example reconstructions...")
        # TODO: Implement visualization
        print("  (Visualization implementation deferred)")
    
    print("\\nEvaluation complete!")


if __name__ == '__main__':
    main()

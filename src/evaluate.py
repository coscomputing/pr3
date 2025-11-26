import torch
from torch.utils.data import DataLoader
import argparse
import os
import matplotlib.pyplot as plt
from src.models import PGUN, UNet
from src.classical import hio, wirtinger_flow
from src.utils import measurement_operator, compute_nmse, align_phase
from src.data import PhaseRetrievalDataset

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    test_dataset = PhaseRetrievalDataset(root=args.data_root, dataset_name=args.dataset, split='test', download=True)
    # Use small batch for eval or just iterate
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load Model if Deep
    model = None
    if args.method in ['PGUN', 'UNet']:
        if args.method == 'PGUN':
            model = PGUN(K=10).to(device)
        else:
            model = UNet().to(device)

        try:
            model.load_state_dict(torch.load(f"{args.method}_best.pth", map_location=device))
            print(f"Loaded {args.method} checkpoint.")
        except FileNotFoundError:
            print("Checkpoint not found, using random init (will fail/perform poorly).")
        model.eval()

    nmse_list = []

    print(f"Evaluating {args.method} on {args.dataset}...")

    for i, x in enumerate(test_loader):
        if i >= args.num_samples:
            break

        x = x.to(device)
        b_clean = measurement_operator(x.to(torch.complex64))
        max_vals = b_clean.amax(dim=(1,2,3), keepdim=True)
        b = b_clean / (max_vals + 1e-8)
        x_scaled = x / (max_vals + 1e-8)

        # Ground truth complex
        x_gt = x_scaled[0].to(torch.complex64)

        x_hat = None

        if args.method == 'PGUN':
            z0 = torch.randn_like(x_scaled, dtype=torch.complex64)
            # torch.norm(z0) calculates Frobenius norm over all dims, which is fine for batch=1
            z0_norm = torch.norm(z0)
            b_norm = torch.norm(b)
            z0 = z0 * (b_norm / (z0_norm + 1e-8))

            with torch.no_grad():
                x_hat = model(z0, b)[0]

        elif args.method == 'UNet':
            b_input = torch.log(1 + b)
            with torch.no_grad():
                x_hat = model(b_input)[0].to(torch.complex64)

        elif args.method == 'HIO':
            # Create support mask (simple thresholding of GT or convex hull?)
            # Since we don't have masks, let's use tight crop or just non-zero pixels of x.
            # MNIST is centered 28x28 in 28x28.
            # Usually HIO needs a loose support.
            # Let's assume support is the whole image for now or a central box.
            # MNIST digits are mostly in center.
            # Let's use a central box of 20x20.
            h, w = x.shape[-2:]
            mask = torch.zeros_like(x[0,0])
            pad = 4
            mask[pad:h-pad, pad:w-pad] = 1.0

            x_hat = hio(b[0], mask, n_iters=500)

        elif args.method == 'WF':
            x_hat = wirtinger_flow(b[0], n_iters=500)

        # Compute NMSE
        err = compute_nmse(x_hat, x_gt, align=True).item()
        nmse_list.append(err)

        if i < 5 and args.save_images:
            # Visualize
            x_aligned = align_phase(x_hat, x_gt)
            plt.figure(figsize=(10,4))
            plt.subplot(1,3,1)
            plt.title('Ground Truth')
            plt.imshow(torch.abs(x_gt).cpu(), cmap='gray')
            plt.subplot(1,3,2)
            plt.title('Measurement (Log)')
            plt.imshow(torch.log(1+b[0,0]).cpu(), cmap='gray')
            plt.subplot(1,3,3)
            plt.title(f'Reconstruction\nNMSE: {err:.4f}')
            plt.imshow(torch.abs(x_aligned).cpu(), cmap='gray')
            plt.savefig(f"result_{args.method}_{i}.png")
            plt.close()

    print(f"Average NMSE: {sum(nmse_list)/len(nmse_list):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--method', type=str, default='PGUN', choices=['PGUN', 'UNet', 'HIO', 'WF'])
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--save_images', action='store_true')
    args = parser.parse_args()

    evaluate(args)

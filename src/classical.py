import torch
import numpy as np
from src.utils import fft2d, ifft2d

def hio(b, support_mask, n_iters=500, beta=0.75, init=None):
    """
    Hybrid Input-Output (HIO) algorithm for phase retrieval.
    
    Args:
        b: [Batch, H, W] - measured magnitudes
        support_mask: [Batch, H, W] - binary mask (1 inside support, 0 outside)
        n_iters: Number of iterations
        beta: Relaxation parameter
        init: Initial guess (optional)
    """
    device = b.device
    if init is None:
        # Random phase initialization
        phase = torch.rand_like(b) * 2 * np.pi
        z = b * torch.exp(1j * phase)
        x = ifft2d(z)
    else:
        x = init

    for k in range(n_iters):
        # 1. Fourier domain constraint: enforce magnitude
        X = fft2d(x)
        phase_X = torch.angle(X)
        X_prime = b * torch.exp(1j * phase_X)
        x_prime = ifft2d(X_prime)

        # 2. Object domain constraint: HIO update
        # Assumes real, non-negative object (e.g., intensity images)
        x_prime_real = x_prime.real

        # Valid: inside support AND non-negative
        valid_mask = (support_mask > 0.5) & (x_prime_real >= 0)

        x_next = x.clone()
        
        # Accept values in valid set
        x_next.real[valid_mask] = x_prime_real[valid_mask]
        x_next.imag[valid_mask] = 0  # Enforce real

        # HIO update for invalid set: x = x - beta * x_prime
        not_valid = ~valid_mask
        x_next.real[not_valid] = x.real[not_valid] - beta * x_prime_real[not_valid]
        x_next.imag[not_valid] = x.imag[not_valid] - beta * x_prime.imag[not_valid]

        x = x_next

    return x

def wirtinger_flow(b, n_iters=500, init=None):
    """
    Wirtinger Flow algorithm for phase retrieval.
    Update: z_{k+1} = z_k - mu_k * grad L(z_k)
    where L(z) = 1/2m sum (|a_j* z|^2 - b_j^2)^2
    """
    device = b.device
    
    if init is None:
        # Spectral initialization using power method on Y = A^* diag(b^2) A
        z = torch.randn_like(b, dtype=torch.complex64)
        z = z / torch.norm(z)
        for _ in range(50):
            Az = fft2d(z)
            z_new = ifft2d(b**2 * Az)
            z = z_new / torch.norm(z_new)

        # Scale to match energy of measurements
        target_norm = torch.norm(b)
        z = z * target_norm
    else:
        z = init

    m = b.numel()

    for k in range(n_iters):
        # Adaptive step size: mu_k = min(1 - exp(-k/250), 0.4)
        mu = min(1.0 - np.exp(-k/250.0), 0.4)

        Az = fft2d(z)
        resid = torch.abs(Az)**2 - b**2

        # Gradient: grad = 1/m A^* (resid * Az)
        grad = ifft2d(resid * Az) / m

        z = z - mu * grad

    return z

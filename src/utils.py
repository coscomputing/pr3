import torch
import torch.fft
import numpy as np

def fft2d(x):
    """
    Computes the normalized 2D DFT of x.
    x: [Batch, C, H, W] or [Batch, H, W] (complex or real)
    Returns: complex tensor of same shape.
    """
    # Norm="ortho" makes the transform unitary, preserving energy.
    # This simplifies the adjoint to be just the inverse.
    return torch.fft.fft2(x, norm="ortho")

def ifft2d(y):
    """
    Computes the normalized 2D IDFT of y.
    """
    return torch.fft.ifft2(y, norm="ortho")

def measurement_operator(x):
    """
    Simulates the measurement process: |Ax|.
    """
    return torch.abs(fft2d(x))

def estimate_lipschitz_constant(input_shape, device='cpu', num_iters=100):
    """
    Estimates the Lipschitz constant L via power iteration of A*A.
    For normalized DFT (unitary operator), L = 1.
    
    Args:
        input_shape: Shape of the input signal
        device: Computation device
        num_iters: Number of power iterations
    
    Returns:
        L: Estimated spectral norm of A*A
    """
    u = torch.randn(input_shape, device=device, dtype=torch.complex64)
    u = u / torch.norm(u)

    for _ in range(num_iters):
        # Apply A*A via FFT
        Au = fft2d(u)
        v = ifft2d(Au)
        u = v / torch.norm(v)

    # Compute Rayleigh quotient
    Au = fft2d(u)
    v = ifft2d(Au)
    L = torch.norm(v) / torch.norm(u)
    return L.item()

def complex_to_real_imag_ch(x):
    """
    x: [B, C, H, W] complex or [B, H, W] complex
    returns: [B, 2*C, H, W] real
    """
    if x.dim() == 4:
        return torch.cat([x.real, x.imag], dim=1)
    else:
        return torch.stack([x.real, x.imag], dim=1)

def real_imag_ch_to_complex(x):
    """
    x: [B, 2*C, H, W] real
    returns: [B, C, H, W] complex
    """
    # Assume even channels
    c = x.shape[1] // 2
    return torch.complex(x[:, :c], x[:, c:])

def align_phase(x_hat, x_gt):
    """
    Aligns global phase of x_hat to match x_gt.
    x_hat, x_gt: complex tensors of same shape.
    Returns: x_hat_aligned
    """
    # Optimal phase is angle( <x_hat, x_gt> )
    dot = torch.sum(x_hat * torch.conj(x_gt))
    phase = torch.angle(dot)
    return x_hat * torch.exp(-1j * phase)

def compute_nmse(x_hat, x_gt, align=True):
    """
    Computes Normalized Mean Squared Error.
    NMSE = ||x_hat - x_gt||^2 / ||x_gt||^2
    
    If align=True, corrects for global phase ambiguity.
    Note: Full registration (shift, conjugate inversion) not implemented.
    """
    if align:
        x_hat = align_phase(x_hat, x_gt)

    diff = torch.norm(x_hat - x_gt)**2
    ref = torch.norm(x_gt)**2
    return diff / ref

def center_crop(img, size):
    """
    Center crop helper.
    """
    h, w = img.shape[-2:]
    th, tw = size, size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[..., i:i+th, j:j+tw]

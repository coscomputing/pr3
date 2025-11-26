import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from src.utils import complex_to_real_imag_ch, real_imag_ch_to_complex, fft2d, ifft2d

class ProximalResBlock(nn.Module):
    def __init__(self, in_channels=2, features=64, spectral_constraint=0.95):
        super().__init__()
        # 3-layer ResNet


        self.conv1 = spectral_norm(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, padding_mode='reflect', bias=True))
        self.in1 = nn.InstanceNorm2d(features)

        self.conv2 = spectral_norm(nn.Conv2d(features, features, kernel_size=3, padding=1, padding_mode='reflect', bias=True))
        self.in2 = nn.InstanceNorm2d(features)

        self.conv3 = spectral_norm(nn.Conv2d(features, in_channels, kernel_size=1, padding=0, bias=True))

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)

        self.spectral_constraint = spectral_constraint

    def forward(self, x):
        # x: [B, 2, H, W]
        identity = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)

        # Residual connection: P(v) = v + D(v)
        return identity + out

class PGUN(nn.Module):
    def __init__(self, K=10, L=1.0):
        super().__init__()
        self.K = K
        self.L = L

        # Learnable step sizes
        # Initialized to 0.8/L
        init_val = 0.8 / L
        init_param = torch.log(torch.exp(torch.tensor(init_val)) - 1.0) # Inverse softplus
        self.alpha_params = nn.Parameter(torch.full((K,), init_param))

        # Proximal operators
        self.prox_layers = nn.ModuleList([
            ProximalResBlock() for _ in range(K)
        ])

    def get_alphas(self):
        return F.softplus(self.alpha_params)

    def gradient_step(self, z, b, alpha):
        # Gradient of f(z) = 0.5 * || |Az| - b ||^2
        # grad f(z) = A* ( Az - b * Az/|Az| )
        #           = z - A* ( b * phase(Az) )  (since A*A=I)

        Az = fft2d(z)
        # Handle division by zero or small numbers for phase
        Az_abs = torch.abs(Az)
        Az_phase = torch.zeros_like(Az)
        mask = Az_abs > 1e-8
        Az_phase[mask] = Az[mask] / Az_abs[mask]

        # term = b * phase(Az)
        term = b * Az_phase

        # grad = z - A*(term)
        grad = z - ifft2d(term)

        # Update: z_new = z - alpha * grad
        z_new = z - alpha * grad
        return z_new

    def forward(self, z0, b):
        z = z0
        alphas = self.get_alphas()

        outputs = []
        for k in range(self.K):
            # 1. Gradient Descent Step
            z_gd = self.gradient_step(z, b, alphas[k])

            # 2. Proximal Step
            # Convert to real representation
            z_real = complex_to_real_imag_ch(z_gd)

            # Pass through CNN
            z_prox_real = self.prox_layers[k](z_real)

            # Convert back
            z = real_imag_ch_to_complex(z_prox_real)

            outputs.append(z)

        return z

class UNet(nn.Module):
    """
    Supervised U-Net baseline for phase retrieval.
    5-level encoder-decoder with channel multipliers {32, 64, 128, 256, 512}.
    Input: Measured Fourier magnitudes b (real-valued).
    Output: Reconstructed spatial-domain image x (real-valued).
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.enc5 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = conv_block(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = conv_block(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = conv_block(128 + 64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = conv_block(64 + 32, 32)

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # x: [B, 1, H, W]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)

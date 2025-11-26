"""
Quick test script to verify installation and run a minimal training example.

This script:
1. Checks that all dependencies are installed
2. Downloads MNIST
3. Runs 2 epochs of PGUN training
4. Evaluates the model

Usage:
    python test_installation.py
"""

import sys
import torch


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required = [
        ('torch', '2.1.0'),
        ('torchvision', None),
        ('numpy', None),
        ('scipy', None),
        ('matplotlib', None),
        ('tqdm', None),
        ('h5py', None),
    ]
    
    missing = []
    
    for package, min_version in required:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package} ({version})")
            
            if min_version and hasattr(module, '__version__'):
                from packaging import version as pkg_version
                if pkg_version.parse(module.__version__) < pkg_version.parse(min_version):
                    print(f"    WARNING: {package} version {version} < {min_version}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!\n")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("Checking CUDA...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠️  CUDA not available - training will use CPU (slower)")
    print()


def test_imports():
    """Test that all project modules can be imported"""
    print("Testing project imports...")
    
    try:
        from src.models import PGUN, UNet
        print("  ✓ src.models")
        
        from src.data import PhaseRetrievalDataset
        print("  ✓ src.data")
        
        from src.utils import fft2d, ifft2d, compute_nmse
        print("  ✓ src.utils")
        
        from src.classical import hio, wirtinger_flow
        print("  ✓ src.classical")
        
        from src.config import get_config
        print("  ✓ src.config")
        
        print("\n✓ All imports successful!\n")
        return True
    except Exception as e:
        print(f"\n✗ Import error: {e}\n")
        return False


def quick_train_test():
    """Run a quick training test on MNIST"""
    print("Running quick training test (2 epochs on MNIST)...")
    print("This will download MNIST (~10MB) on first run.\n")
    
    import torch
    from torch.utils.data import DataLoader
    from src.models import PGUN
    from src.data import PhaseRetrievalDataset
    from src.utils import measurement_operator
    from src.config import get_config
    
    # Config
    config = get_config('MNIST', 'PGUN')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small dataset for testing
    train_dataset = PhaseRetrievalDataset(
        root='./data',
        dataset_name='MNIST',
        split='train',
        download=True,
        augment=False
    )
    
    # Use only 100 samples for quick test
    train_dataset.data = torch.utils.data.Subset(train_dataset.data, range(100))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Model
    model = PGUN(K=3, L=1.0).to(device)  # Smaller model for test
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train for 2 epochs
    model.train()
    for epoch in range(2):
        total_loss = 0
        for i, x in enumerate(train_loader):
            x = x.to(device)
            
            # Generate measurements
            b = measurement_operator(x.to(torch.complex64))
            max_vals = b.amax(dim=(1,2,3), keepdim=True)
            b = b / (max_vals + 1e-8)
            
            # Forward
            z0 = torch.randn_like(x, dtype=torch.complex64)
            x_hat = model(z0, b)
            
            # Loss
            Ax_hat = measurement_operator(x_hat)
            loss = torch.mean((Ax_hat - b)**2)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i == 0:  # Print first batch
                print(f"Epoch {epoch+1}/2, Batch {i+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/2 - Average Loss: {avg_loss:.4f}\n")
    
    print("✓ Training test successful!\n")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("PGUN Installation Test")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Quick training test
    try:
        if quick_train_test():
            print("=" * 60)
            print("✅ ALL TESTS PASSED!")
            print("=" * 60)
            print()
            print("You're ready to start training:")
            print("  python main.py --dataset MNIST --model PGUN --epochs 10")
            print()
    except Exception as e:
        print(f"\n❌ Training test failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

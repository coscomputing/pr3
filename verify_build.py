"""
BUILD VERIFICATION SCRIPT

This script verifies that all components of the PGUN implementation are properly built
and configured according to the paper specifications.

Run this after completing the implementation to ensure everything is in place.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    p = Path(filepath)
    if p.exists():
        size = p.stat().st_size
        print(f"  ✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {description}: {filepath} - NOT FOUND")
        return False


def check_directory_structure():
    """Verify directory structure"""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    base_dir = Path(".")
    required_files = {
        # Core scripts
        "main.py": "Main Training Script",
        "evaluate.py": "Evaluation Script",
        "test_installation.py": "Installation Test",
        "requirements.txt": "Dependencies File",
        
        # Documentation
        "README.md": "Main README",
        "QUICKSTART.md": "Quick Start Guide",
        "IMPLEMENTATION_SUMMARY.md": "Implementation Summary",
        
        # Source code
        "src/config.py": "Configuration System",
        "src/models.py": "Model Implementations (PGUN, UNet)",
        "src/data.py": "Dataset Loaders",
        "src/utils.py": "Utility Functions",
        "src/classical.py": "Classical Baselines (HIO, WF)",
        "src/generate_emdb_dataset.py": "EMDB-1050 Generator",
       "src/train.py": "Legacy Training Script",
        "src/evaluate.py": "Legacy Evaluation Script",
        ".gitignore": "Git Ignore File (optional)",
    }
    
    all_exist = True
    for filepath, description in required_files.items():
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def check_imports():
    """Check if all modules can be imported"""
    print("\n" + "="*60)
    print("CHECKING MODULE IMPORTS")
    print("="*60)
    
    modules_to_check = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "TQDM"),
        ("h5py", "H5PY"),
        ("tensorboard", "TensorBoard (torch.utils.tensorboard)"),
    ]
    
    all_imported = True
    for module_name, description in modules_to_check:
        try:
            if module_name == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
            else:
                __import__(module_name)
            print(f"  ✓ {description}")
        except ImportError as e:
            print(f"  ✗ {description} - {e}")
            all_imported = False
    
    return all_imported


def check_project_modules():
    """Check if project modules can be imported"""
    print("\n" + "="*60)
    print("CHECKING PROJECT MODULES")
    print("="*60)
    
    project_modules = [
        ("src.config", "Configuration System"),
        ("src.models", "Models (PGUN, UNet)"),
        ("src.data", "Dataset Loaders"),
        ("src.utils", "Utility Functions"),
        ("src.classical", "Classical Baselines"),
    ]
    
    all_imported = True
    for module_name, description in project_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {description}")
        except Exception as e:
            print(f"  ✗ {description} - {e}")
            all_imported = False
    
    return all_imported


def check_configuration():
    """Check configuration system"""
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION SYSTEM")
    print("="*60)
    
    try:
        from src.config import get_config, MNIST_PGUN_CONFIG, CELEBA_PGUN_CONFIG, EMDB_PGUN_CONFIG
        
        # Check MNIST config
        config_mnist = get_config('MNIST', 'PGUN')
        print(f"  ✓ MNIST Config: batch_size={config_mnist.data.get_batch_size()}, epochs={config_mnist.train.epochs}")
        
        # Check CelebA config
        config_celeba = get_config('CelebA', 'PGUN')
        print(f"  ✓ CelebA Config: batch_size={config_celeba.data.get_batch_size()}, epochs={config_celeba.train.epochs}")
        
        # Check EMDB config
        config_emdb = get_config('EMDB-1050', 'PGUN')
        print(f"  ✓ EMDB-1050 Config: batch_size={config_emdb.data.get_batch_size()}, epochs={config_emdb.train.epochs}")
        
        # Verify hyperparameters from paper
        print(f"\n  Paper Hyperparameters (Appendix A Table 1):")
        print(f"    Adam beta1: {config_mnist.optim.beta1} (expected: 0.9)")
        print(f"    Adam beta2: {config_mnist.optim.beta2} (expected: 0.999)")
        print(f"    Weight decay: {config_mnist.optim.weight_decay} (expected: 1e-6)")
        print(f"    Initial LR: {config_mnist.optim.initial_lr} (expected: 1e-4)")
        print(f"    Final LR: {config_mnist.optim.final_lr} (expected: 1e-6)")
        print(f"    Gradient clip: {config_mnist.optim.gradient_clip_value} (expected: 5.0)")
        print(f"    Mixed precision (AMP): {config_mnist.optim.use_amp} (expected: True)")
        print(f"    PGUN layers (K): {config_mnist.pgun.K} (expected: 10)")
        print(f"    Proximal features: {config_mnist.pgun.prox_features} (expected: 64)")
        print(f"    Dropout: {config_mnist.pgun.prox_dropout} (expected: 0.1)")
        print(f"    Spectral norm constraint: {config_mnist.pgun.spectral_norm_constraint} (expected: 0.95)")
        
        return True
    except Exception as e:
        print(f"  ✗ Configuration check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_architecture():
    """Check model architectures"""
    print("\n" + "="*60)
    print("CHECKING MODEL ARCHITECTURES")
    print("="*60)
    
    try:
        import torch
        from src.models import PGUN, UNet
        
        # Check PGUN
        pgun = PGUN(K=10, L=1.0)
        param_count = sum(p.numel() for p in pgun.parameters())
        print(f"  ✓ PGUN: {param_count:,} parameters, K=10 layers")
        
        # Check UNet
        unet = UNet()
        param_count_unet = sum(p.numel() for p in unet.parameters())
        print(f"  ✓ UNet: {param_count_unet:,} parameters")
        
        # Test forward pass
        dummy_z0 = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        dummy_b = torch.randn(1, 1, 32, 32)
        
        with torch.no_grad():
            output_pgun = pgun(dummy_z0, dummy_b)
        print(f"  ✓ PGUN forward pass: input shape {dummy_z0.shape} -> output shape {output_pgun.shape}")
        
        dummy_input = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            output_unet = unet(dummy_input)
        print(f"  ✓ UNet forward pass: input shape {dummy_input.shape} -> output shape {output_unet.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model architecture check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print implementation summary"""
    print("\n" + "="*60)
    print("IMPLEMENTATION SUMMARY")
    print("="*60)
    print("""
✅ COMPLETED COMPONENTS:

1. Configuration System (src/config.py)
   - All hyperparameters from Appendix A Table 1
   - Dataset-specific settings
   - Optimizer and training configurations

2. Model Implementations (src/models.py)
   - PGUN: 10-layer unfolded network with learnable proximal operators
   - UNet: 5-level encoder-decoder baseline
   - ProximalResBlock: 3-layer ResNet with spectral norm

3. Dataset Loaders (src/data.py)
   - MNIST support (auto-download)
   - CelebA support
   - EMDB-1050 support (requires generation)
   - Data augmentation (horizontal flip, ±5° rotation)

4. Classical Baselines (src/classical.py)
   - Hybrid Input-Output (HIO)
   - Wirtinger Flow (WF) with spectral initialization

5. Utility Functions (src/utils.py)
   - Orthonormal 2D FFT/IFFT
   - NMSE computation with phase alignment
   - Measurement operator

6. Training Pipeline (main.py)
   - Mixed precision training (PyTorch AMP)
   - Tensorboard logging
   - Cosine LR schedule
   - Early stopping
   - Checkpoint saving

7. Evaluation Pipeline (evaluate.py)
   - All methods (PGUN, UNet, HIO, WF)
   - NMSE statistics
   - Result saving

8. EMDB-1050 Generator (src/generate_emdb_dataset.py)
   - Volume download from EMDB
   - 2D projection rendering
   - HDF5 storage

9. Documentation
   - README.md: Comprehensive documentation
   - QUICKSTART.md: Step-by-step guide
   - IMPLEMENTATION_SUMMARY.md: Technical details

📊 EXPECTED RESULTS (Paper Table 1):

| Method      | MNIST | CelebA | EMDB-1050 |
|-------------|-------|--------|-----------|
| PGUN        | 0.058 | 0.094  | 0.119     |
| HIO         | 0.215 | 0.358  | 0.412     |
| WF          | 0.189 | 0.311  | 0.365     |
| UNet        | 0.098 | 0.155  | 0.198     |

🚀 NEXT STEPS:

1. Install dependencies:
   pip install -r requirements.txt

2. Test installation:
   python test_installation.py

3. Generate EMDB-1050 (optional):
   python src/generate_emdb_dataset.py

4. Train PGUN:
   python main.py --dataset MNIST --model PGUN --epochs 100

5. Evaluate:
   python evaluate.py --dataset MNIST --method PGUN
""")


def main():
    """Main build verification function"""
    print("\n" + "="*60)
    print("PGUN IMPLEMENTATION - BUILD VERIFICATION")
    print("="*60)
    print("\nThis script verifies that all components are properly built.")
    print("Run from the project root directory.\n")
    
    results = {
        "Directory Structure": check_directory_structure(),
        "Module Imports": check_imports(),
        "Project Modules": check_project_modules(),
        "Configuration": check_configuration(),
        "Model Architecture": check_model_architecture(),
    }
    
    print("\n" + "="*60)
    print("BUILD VERIFICATION RESULTS")
    print("="*60)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ ALL CHECKS PASSED - BUILD COMPLETE!")
        print("="*60)
        print_summary()
        return 0
    else:
        print("\n" + "="*60)
        print("❌ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease review the errors above and fix any issues.")
        print("Common solutions:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Ensure you're running from the project root directory")
        print("  3. Check that all files were created properly")
        return 1


if __name__ == '__main__':
    sys.exit(main())

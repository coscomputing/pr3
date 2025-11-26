# Quick Start Guide - PGUN Implementation

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install torch>=2.1.0 torchvision numpy scipy matplotlib tqdm pillow h5py tensorboard
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python test_installation.py
```

This will:
- Check all dependencies
- Test CUDA availability
- Run a quick 2-epoch training test
- Download MNIST (first run only)

Expected output:
```
✅ ALL TESTS PASSED!
You're ready to start training:
  python main.py --dataset MNIST --model PGUN --epochs 10
```

---

## Quick Training Tests

### Test 1: PGUN on MNIST (10 epochs, ~5 minutes on GPU)

```bash
python main.py --dataset MNIST --model PGUN --epochs 10
```

**What happens:**
- Downloads MNIST automatically
- Trains PGUN for 10 epochs
- Saves checkpoints to `./checkpoints/MNIST/PGUN/`
- Logs to tensorboard: `./logs/MNIST/PGUN/`

**Check results:**
```bash
# View training logs
tensorboard --logdir ./logs

# Evaluate
python evaluate.py --dataset MNIST --method PGUN
```

Expected NMSE: ~0.1-0.15 (after just 10 epochs; paper reports 0.058 after 100 epochs)

---

### Test 2: Classical Baseline - HIO

```bash
python evaluate.py --dataset MNIST --method HIO --num_samples 10
```

**What happens:**
- Runs HIO algorithm on 10 MNIST test images
- No training required!
- Shows NMSE results

Expected NMSE: ~0.20-0.25 (paper reports 0.215)

---

## Full Paper Reproduction

### MNIST Results (Table 1)

```bash
# Train PGUN (100 epochs, ~1 hour on GPU)
python main.py --dataset MNIST --model PGUN --epochs 100

# Evaluate PGUN
python evaluate.py --dataset MNIST --method PGUN

# Compare with baselines (no training)
python evaluate.py --dataset MNIST --method HIO --num_samples 100
python evaluate.py --dataset MNIST --method WF --num_samples 100

# Train U-Net baseline
python main.py --dataset MNIST --model UNet --epochs 100
python evaluate.py --dataset MNIST --method UNet
```

**Expected Results:**
- PGUN: NMSE ≈ 0.058
- HIO: NMSE ≈ 0.215  
- WF: NMSE ≈ 0.189
- U-Net: NMSE ≈ 0.098

---

## Monitoring Training

### Tensorboard

While training is running, open a new terminal:

```bash
tensorboard --logdir ./logs
```

Then open browser to: http://localhost:6006

**You'll see:**
- Training loss curves
- Validation NMSE curves
- Learning rate schedule
- Per-batch and per-epoch metrics

### Checkpoints

Checkpoints are saved to: `./checkpoints/<dataset>/<model>/`

Files:
- `best_model.pth` - Best validation NMSE
- `checkpoint_epoch_N.pth` - Saved every 10 epochs

**Load a checkpoint:**
```bash
python evaluate.py --dataset MNIST --method PGUN \
    --checkpoint ./checkpoints/MNIST/PGUN/best_model.pth
```

---

## Working with Different Datasets

### CelebA

**Option 1: Automatic download (requires manual setup)**
```bash
python main.py --dataset CelebA --model PGUN --epochs 100
```

**Option 2: Manual download**
1. Download from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Place in `./data/celeba/`
3. Run training as above

**Note:** CelebA is ~1.4GB

### EMDB-1050 Cryo-EM Projections

**Generate dataset first:**
```bash
python src/generate_emdb_dataset.py --num_train 5000 --num_test 1000
```

This will:
1. Download EMD-1050 volume from EMDB (~50MB)
2. Generate 6,000 2D projections
3. Save to `./data/EMDB-1050/projections.h5` (~500MB)
4. Takes ~30 minutes to 1 hour

**Then train:**
```bash
python main.py --dataset EMDB-1050 --model PGUN --epochs 100
```

---

## Command-Line Options

### Training (main.py)

```bash
python main.py [options]

Required:
  --dataset {MNIST,CelebA,EMDB-1050}  # Dataset choice
  --model {PGUN,UNet}                 # Model choice

Optional:
  --epochs N          # Number of epochs (default: 100)
  --batch_size N      # Batch size (default: from config)
  --lr FLOAT          # Learning rate (default: 1e-4)
  --data_root PATH    # Data directory (default: ./data)
  --no_amp            # Disable mixed precision
  --seed N            # Random seed (default: 42)
```

### Evaluation (evaluate.py)

```bash
python evaluate.py [options]

Required:
  --dataset {MNIST,CelebA,EMDB-1050}
  --method {PGUN,UNet,HIO,WF}

Optional:
  --checkpoint PATH      # Model checkpoint (for PGUN/UNet)
  --num_samples N        # Limit test samples
  --save_images          # Save example reconstructions
  --output_dir PATH      # Results directory
  --batch_size N         # Evaluation batch size
```

---

## Troubleshooting

### Out of Memory

**Reduce batch size:**
```bash
python main.py --dataset MNIST --model PGUN --batch_size 8
```

**Or disable mixed precision:**
```bash
python main.py --dataset MNIST --model PGUN --no_amp
```

### Slow Training (CPU only)

If CUDA is not available:
- Training will be slower but still work
- Reduce batch size if needed
- Consider using fewer epochs for testing

### MNIST Download Fails

Manually download from: http://yann.lecun.com/exdb/mnist/

Place files in: `./data/MNIST/raw/`

### CelebA Download Requires Login

CelebA auto-download may fail. Download manually from official source and place in `./data/celeba/`.

---

## Example Output

### During Training:
```
Using device: cuda
Training PGUN on MNIST
Configuration: epochs=100, batch_size=32
Model parameters: 1,234,567

Starting training...
Epoch 1/100 - Average Loss: 0.0234, Val NMSE: 0.1234
Epoch 2/100 - Average Loss: 0.0198, Val NMSE: 0.1123
  * New best model saved (NMSE: 0.1123)
...
```

### After Evaluation:
```
Evaluating PGUN on MNIST
Device: cuda
Test samples: 10000

Results for PGUN on MNIST:
  Mean NMSE: 0.058123
  Std NMSE: 0.012456
  Median NMSE: 0.056789
  Min NMSE: 0.023456
  Max NMSE: 0.145678

Results saved to: ./results/MNIST/PGUN/results.txt
```

---

## What's Next?

1. ✅ **Verify installation**: `python test_installation.py`
2. ✅ **Quick test**: `python main.py --dataset MNIST --model PGUN --epochs 10`
3. ✅ **Full training**: `python main.py --dataset MNIST --model PGUN --epochs 100`
4. ✅ **Evaluation**: `python evaluate.py --dataset MNIST --method PGUN`
5. ✅ **Compare baselines**: Evaluate HIO, WF, UNet
6. 🎯 **Reproduce paper**: Run on all 3 datasets
7. 📊 **Analyze**: Review tensorboard logs

---

## Key Files

- `main.py` - **Start here** for training
- `evaluate.py` - Evaluation and metrics
- `src/config.py` - All hyperparameters
- `src/models.py` - PGUN architecture
- `README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details

---

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `IMPLEMENTATION_SUMMARY.md` for technical details  
3. Open an issue on GitHub
4. Contact: liueducati@gmail.com

---

**Happy training! 🚀**

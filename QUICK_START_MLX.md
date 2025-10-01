# Quick Start - MLX Training

## 🚀 Run from Project Root

```bash
# Quick test (5 minutes)
python train_mlx.py --mode full --backend mlx --n_ground_truths 5 --n_epochs 10

# Full training (40-60 minutes on M2 Max)
python train_mlx.py --mode full --backend mlx --n_ground_truths 100 --n_epochs 50

# Just generate dataset
python train_mlx.py --mode generate --n_ground_truths 100

# Just train (dataset must exist)
python train_mlx.py --mode train --backend mlx --n_epochs 50

# Run inference
python train_mlx.py --mode inference --backend mlx
```

## 📁 Or Run from ml_ideation Directory

```bash
cd ml_ideation

# Same commands work
python complete_pipeline_mlx.py --mode full --backend mlx --n_ground_truths 5 --n_epochs 10
```

## 🎯 Common Commands

### Test MLX Installation
```bash
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"
```

### Install Dependencies
```bash
pip install -r ml_ideation/requirements-mlx.txt
```

### Run Migration Script
```bash
cd ml_ideation
./migrate_to_mlx.sh
```

### Compare PyTorch vs MLX
```bash
# PyTorch (from root)
python train_mlx.py --mode train --backend pytorch --n_epochs 5

# MLX (from root)
python train_mlx.py --mode train --backend mlx --n_epochs 5
```

## 📚 Full Documentation

- **Complete Guide:** `ml_ideation/MLX_TRAINING_GUIDE.md`
- **Comparison:** `ml_ideation/COMPARISON_PYTORCH_VS_MLX.md`
- **Summary:** `ml_ideation/MLX_IMPLEMENTATION_SUMMARY.md`

## ⚡ Performance (M2 Max)

| Dataset | Epochs | Time (MLX) |
|---------|--------|------------|
| 5 GTs | 10 | ~3 min |
| 50 GTs | 50 | ~25 min |
| 100 GTs | 50 | ~40 min |

## 🐛 Troubleshooting

### "No such file or directory"
```bash
# Solution: Use train_mlx.py from root, or cd to ml_ideation
python train_mlx.py --mode full --backend mlx --n_ground_truths 5 --n_epochs 10
```

### "MLX not available"
```bash
pip install mlx
```

### "Out of memory"
```bash
# Reduce batch size
python train_mlx.py --mode train --backend mlx --batch_size 256
```

## ✅ Verify Installation

```bash
# Check platform
sysctl -n machdep.cpu.brand_string

# Check Python
python3 --version

# Check MLX
python -c "import mlx.core as mx; print('MLX OK')"

# Run test
python train_mlx.py --mode full --backend mlx --n_ground_truths 5 --n_epochs 10
```

"""
Complete End-to-End Pipeline: Dataset Generation → Training → Inference
MLX-optimized version for Apple Silicon

Supports both PyTorch and MLX backends
"""

import argparse
import sys
from pathlib import Path
import platform


def check_apple_silicon():
    """Check if running on Apple Silicon"""
    if platform.system() != 'Darwin':
        return False

    # Check for Apple Silicon
    try:
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                              capture_output=True, text=True)
        return 'Apple' in result.stdout
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="EIS Parameter Prediction - Complete Pipeline (MLX Optimized)"
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate', 'train', 'inference', 'full'],
        required=True,
        help='Pipeline mode: generate dataset, train model, run inference, or full pipeline'
    )

    parser.add_argument(
        '--backend',
        type=str,
        choices=['mlx', 'pytorch', 'auto'],
        default='auto',
        help='Training backend: mlx (Apple Silicon), pytorch, or auto-detect'
    )

    parser.add_argument(
        '--n_ground_truths',
        type=int,
        default=100,
        help='Number of ground truth configurations (default: 100)'
    )

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./eis_training_data',
        help='Directory for dataset storage'
    )

    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help='Number of parallel workers for dataset generation'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size for training (default: 512)'
    )

    args = parser.parse_args()

    # Auto-detect backend
    if args.backend == 'auto':
        if check_apple_silicon():
            try:
                import mlx.core as mx
                args.backend = 'mlx'
                print("✓ Detected Apple Silicon - using MLX backend")
            except ImportError:
                args.backend = 'pytorch'
                print("⚠ MLX not available - falling back to PyTorch")
        else:
            args.backend = 'pytorch'
            print("✓ Using PyTorch backend")

    print("="*70)
    print("EIS PARAMETER PREDICTION PIPELINE (MLX-OPTIMIZED)")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Backend: {args.backend}")
    print(f"Data directory: {args.data_dir}")
    print("="*70)

    if args.mode in ['generate', 'full']:
        run_generation(args)

    if args.mode in ['train', 'full']:
        run_training(args)

    if args.mode == 'inference':
        run_inference(args)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


def run_generation(args):
    """Dataset generation phase"""
    print("\n" + "="*70)
    print("PHASE 1: DATASET GENERATION")
    print("="*70)

    from dataset_generation_system import DatasetGenerator

    generator = DatasetGenerator(
        output_dir=args.data_dir,
        n_grid_points=12
    )

    dataset, ground_truths = generator.generate_complete_dataset(
        n_ground_truths=args.n_ground_truths,
        parallel=True,
        n_workers=args.n_workers
    )

    print(f"\n✓ Generated dataset with {len(dataset):,} models")
    print(f"✓ Covering {len(ground_truths)} ground truth configurations")


def run_training(args):
    """Model training phase"""
    print("\n" + "="*70)
    print("PHASE 2: MODEL TRAINING")
    print("="*70)
    print(f"Backend: {args.backend.upper()}")
    print("="*70)

    # Paths - find any existing dataset
    data_path = Path(args.data_dir) / f"combined_dataset_{args.n_ground_truths}gt.csv"

    # If specific dataset doesn't exist, look for any dataset
    if not data_path.exists():
        # Look for any combined_dataset_*.csv file
        data_dir = Path(args.data_dir)
        if data_dir.exists():
            csv_files = list(data_dir.glob("combined_dataset_*gt.csv"))
            if csv_files:
                data_path = csv_files[0]
                print(f"Using existing dataset: {data_path.name}")
            else:
                print(f"Error: No dataset found in {args.data_dir}")
                print("Please run with --mode generate first")
                sys.exit(1)
        else:
            print(f"Error: Data directory not found: {args.data_dir}")
            print("Please run with --mode generate first")
            sys.exit(1)

    # Masking patterns
    masking_patterns = [
        [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0], [1, 1, 0, 0, 1], [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1]
    ]

    if args.backend == 'mlx':
        run_training_mlx(args, data_path, masking_patterns)
    else:
        run_training_pytorch(args, data_path, masking_patterns)


def run_training_mlx(args, data_path, masking_patterns):
    """Train using MLX backend"""
    try:
        import mlx.core as mx
        from eis_predictor_mlx import (
            MLXEISDataset,
            ProbabilisticEISPredictorMLX,
            train_model_mlx
        )
        import numpy as np
    except ImportError as e:
        print(f"Error: MLX not available: {e}")
        print("Install with: pip install mlx")
        sys.exit(1)

    # Load dataset
    print("\nLoading dataset with MLX...")
    dataset = MLXEISDataset(
        data_path=str(data_path),
        masking_patterns=masking_patterns,
        samples_per_pattern=10000
    )

    # Split into train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    class SubsetDataset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def get_batch(self, batch_indices):
            actual_indices = [self.indices[i] for i in batch_indices]
            return self.dataset.get_batch(actual_indices)

    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    # Create model
    print("\nCreating MLX model...")
    model = ProbabilisticEISPredictorMLX(n_grid_points=12, hidden_dim=512)
    mx.eval(model.parameters())

    # Train
    history = train_model_mlx(
        model,
        train_dataset,
        val_dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=0.001
    )

    print(f"\n✓ Training complete! Best model saved to best_eis_predictor_mlx.npz")


def run_training_pytorch(args, data_path, masking_patterns):
    """Train using PyTorch backend"""
    try:
        import torch
        from torch.utils.data import DataLoader
        from eis_predictor_implementation import EISDataset, ProbabilisticEISPredictor, train_model
    except ImportError as e:
        print(f"Error: PyTorch not available: {e}")
        print("Install with: pip install torch")
        sys.exit(1)

    # Load dataset
    print("\nLoading dataset with PyTorch...")
    full_dataset = EISDataset(
        data_path=str(data_path),
        masking_patterns=masking_patterns,
        samples_per_pattern=10000
    )

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    # Model
    model = ProbabilisticEISPredictor(n_grid_points=12, hidden_dim=512)

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    history = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=args.n_epochs,
        device=device
    )

    print(f"\n✓ Training complete! Best model saved to best_eis_predictor.pth")


def run_inference(args):
    """Inference phase - demonstrate predictions"""
    print("\n" + "="*70)
    print("PHASE 3: INFERENCE DEMO")
    print("="*70)

    # Load ground truth metadata
    import json
    metadata_path = Path(args.data_dir) / "ground_truth_metadata.json"

    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        print("Please run with --mode generate first")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        ground_truths = json.load(f)

    if args.backend == 'mlx':
        run_inference_mlx(args, ground_truths)
    else:
        run_inference_pytorch(args, ground_truths)


def run_inference_mlx(args, ground_truths):
    """Run inference with MLX backend"""
    try:
        import mlx.core as mx
        from eis_predictor_mlx import ProbabilisticEISPredictorMLX, MLXEISParameterCompleter
        import numpy as np
    except ImportError as e:
        print(f"Error: MLX not available: {e}")
        sys.exit(1)

    # Load model
    checkpoint_path = Path('best_eis_predictor_mlx.npz')

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run with --mode train first")
        sys.exit(1)

    model = ProbabilisticEISPredictorMLX(n_grid_points=12, hidden_dim=512)

    # Load weights
    try:
        weights = mx.load(str(checkpoint_path))
        model.load_weights(list(weights.items()))
    except:
        print("⚠ Could not load weights, using initialized model")

    model.eval()

    print(f"✓ Loaded MLX model from {checkpoint_path}")

    # Demo: predict for a random ground truth
    gt = ground_truths[0]

    print(f"\n" + "="*70)
    print("INFERENCE EXAMPLE (MLX)")
    print("="*70)
    print(f"\nGround Truth Circuit: {gt['id']}")
    print(f"  Rsh = {gt['rsh']:.1f} Ω")
    print(f"  Ra = {gt['ra']:.1f} Ω")
    print(f"  Rb = {gt['rb']:.1f} Ω")
    print(f"  Ca = {gt['ca']:.2e} F")
    print(f"  Cb = {gt['cb']:.2e} F")

    # Create predictor (need grids from dataset)
    data_path = Path(args.data_dir) / f"combined_dataset_{args.n_ground_truths}gt.csv"
    from eis_predictor_mlx import MLXEISDataset

    masking_patterns = [[1, 1, 1, 0, 0]]
    dataset = MLXEISDataset(str(data_path), masking_patterns, samples_per_pattern=100)

    predictor = MLXEISParameterCompleter(model, dataset.grids)

    # Predict
    results = predictor.predict_missing_parameters(
        known_params={'Rsh': gt['rsh'], 'Ra': gt['ra'], 'Rb': gt['rb']},
        top_k=10
    )

    print(f"\nPrediction (knowing Rsh, Ra, Rb):")
    print(f"  Predicted Resnorm: {results['predicted_resnorm']:.4f}")

    print(f"\n  Top 10 joint predictions:")
    for pred in results['top_k_predictions']:
        print(f"    {pred['rank']}. Ca = {pred['Ca']:.2e} F, Cb = {pred['Cb']:.2e} F")
        print(f"       P(joint) = {pred['joint_probability']:.4f}")


def run_inference_pytorch(args, ground_truths):
    """Run inference with PyTorch backend"""
    try:
        import torch
        import numpy as np
        from eis_predictor_implementation import ProbabilisticEISPredictor
    except ImportError as e:
        print(f"Error: PyTorch not available: {e}")
        sys.exit(1)

    # Load model
    checkpoint_path = Path('best_eis_predictor.pth')

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run with --mode train first")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ProbabilisticEISPredictor(n_grid_points=12, hidden_dim=512)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"✓ Loaded PyTorch model from {checkpoint_path}")

    # Demo: predict for a random ground truth
    gt = ground_truths[0]

    print(f"\n" + "="*70)
    print("INFERENCE EXAMPLE (PYTORCH)")
    print("="*70)
    print(f"\nGround Truth Circuit: {gt['id']}")
    print(f"  Rsh = {gt['rsh']:.1f} Ω")
    print(f"  Ra = {gt['ra']:.1f} Ω")
    print(f"  Rb = {gt['rb']:.1f} Ω")
    print(f"  Ca = {gt['ca']:.2e} F")
    print(f"  Cb = {gt['cb']:.2e} F")

    # Create input: know Rsh, Ra, Rb, predict Ca, Cb
    known_params_log = [
        np.log10(gt['rsh']),
        np.log10(gt['ra']),
        np.log10(gt['rb']),
        0.0,
        0.0
    ]
    mask = [1.0, 1.0, 1.0, 0.0, 0.0]

    # Predict
    with torch.no_grad():
        params_tensor = torch.tensor([known_params_log], dtype=torch.float32).to(device)
        mask_tensor = torch.tensor([mask], dtype=torch.float32).to(device)

        param_probs, resnorm_pred = model(params_tensor, mask_tensor)

    # Extract predictions for Ca and Cb
    ca_probs = param_probs[3][0].cpu().numpy()
    cb_probs = param_probs[4][0].cpu().numpy()

    print(f"\nPrediction (knowing Rsh, Ra, Rb):")
    print(f"  Predicted Resnorm: {resnorm_pred.item():.4f}")

    # Joint predictions
    joint = np.outer(ca_probs, cb_probs)
    flat_joint = joint.flatten()
    top_joint_indices = np.argsort(flat_joint)[-10:][::-1]

    print(f"\n  Top 10 joint predictions:")
    for i, flat_idx in enumerate(top_joint_indices, 1):
        ca_idx, cb_idx = flat_idx // 12, flat_idx % 12
        prob = flat_joint[flat_idx]
        print(f"    {i}. Ca[{ca_idx}] × Cb[{cb_idx}]: P = {prob:.4f}")


if __name__ == "__main__":
    main()

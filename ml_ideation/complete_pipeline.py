"""
Complete End-to-End Pipeline: Dataset Generation → Training → Inference
Run this script to go from scratch to a fully trained model
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="EIS Parameter Prediction - Complete Pipeline"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate', 'train', 'inference', 'full'],
        required=True,
        help='Pipeline mode: generate dataset, train model, run inference, or full pipeline'
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
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory for model checkpoints'
    )
    
    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help='Number of parallel workers for dataset generation'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("EIS PARAMETER PREDICTION PIPELINE")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
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
    
    import torch
    from torch.utils.data import DataLoader
    from multi_gt_trainer import MultiGroundTruthDataset, AdaptiveEISPredictor, MultiGTTrainer
    
    # Paths
    data_path = Path(args.data_dir) / f"combined_dataset_{args.n_ground_truths}gt.csv"
    metadata_path = Path(args.data_dir) / "ground_truth_metadata.json"
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Please run with --mode generate first")
        sys.exit(1)
    
    # Masking patterns
    masking_patterns = [
        [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0], [1, 1, 0, 0, 1], [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1]
    ]
    
    # Load dataset
    full_dataset = MultiGroundTruthDataset(
        csv_path=str(data_path),
        metadata_path=str(metadata_path),
        masking_patterns=masking_patterns,
        augmentation_factor=3
    )
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=512, num_workers=4)
    
    # Model
    model = AdaptiveEISPredictor(n_grid_points=12, hidden_dim=512)
    
    # Train
    trainer = MultiGTTrainer(model)
    history = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.n_epochs,
        learning_rate=0.001,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print(f"\n✓ Training complete! Best model saved to {args.checkpoint_dir}")


def run_inference(args):
    """Inference phase - demonstrate predictions"""
    print("\n" + "="*70)
    print("PHASE 3: INFERENCE DEMO")
    print("="*70)
    
    import torch
    import json
    import numpy as np
    from multi_gt_trainer import AdaptiveEISPredictor
    
    # Load model
    checkpoint_path = Path(args.checkpoint_dir) / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run with --mode train first")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AdaptiveEISPredictor(n_grid_points=12, hidden_dim=512)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Load ground truth metadata
    metadata_path = Path(args.data_dir) / "ground_truth_metadata.json"
    with open(metadata_path, 'r') as f:
        ground_truths = json.load(f)
    
    # Demo: predict for a random ground truth
    gt = ground_truths[0]  # Use first ground truth as example
    
    print(f"\n" + "="*70)
    print("INFERENCE EXAMPLE")
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
        
        param_probs, resnorm_pred, uncertainty = model(params_tensor, mask_tensor)
    
    # Extract predictions for Ca and Cb
    ca_probs = param_probs[3][0].cpu().numpy()
    cb_probs = param_probs[4][0].cpu().numpy()
    
    print(f"\nPrediction (knowing Rsh, Ra, Rb):")
    print(f"  Predicted Resnorm: {resnorm_pred.item():.4f}")
    
    # Top predictions for Ca
    top_ca_indices = np.argsort(ca_probs)[-3:][::-1]
    print(f"\n  Top 5 joint predictions (Ca, Cb):")
    for i, flat_idx in enumerate(top_joint_indices, 1):
        ca_idx, cb_idx = flat_idx // 12, flat_idx % 12
        prob = flat_joint[flat_idx]
        print(f"    {i}. Ca[{ca_idx}] × Cb[{cb_idx}]: P = {prob:.4f}")
    
    print(f"\n  Uncertainty estimates:")
    uncert = uncertainty[0].cpu().numpy()
    param_names = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb']
    for i, name in enumerate(param_names):
        if mask[i] == 0:  # Only show uncertainty for predicted params
            print(f"    {name}: {uncert[i]:.4f}")


if __name__ == "__main__":
    main()
3 Ca predictions:")
    for i, idx in enumerate(top_ca_indices, 1):
        print(f"    {i}. Index {idx}: P = {ca_probs[idx]:.4f}")
    
    # Top predictions for Cb
    top_cb_indices = np.argsort(cb_probs)[-3:][::-1]
    print(f"\n  Top 3 Cb predictions:")
    for i, idx in enumerate(top_cb_indices, 1):
        print(f"    {i}. Index {idx}: P = {cb_probs[idx]:.4f}")
    
    # Joint predictions
    joint = np.outer(ca_probs, cb_probs)
    flat_joint = joint.flatten()
    top_joint_indices = np.argsort(flat_joint)[-5:][::-1]
    
    print(f"\n  Top 
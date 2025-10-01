"""
EIS Parameter Predictor with Probabilistic Inference - MLX Implementation
Optimized for Apple Silicon using MLX framework

Advantages over PyTorch:
- Unified memory model (no CPU<->GPU transfers)
- Native Apple Silicon optimization
- Lazy computation for better performance
- Simpler API with automatic differentiation
"""

import numpy as np
import pandas as pd
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
from tqdm import tqdm


class MLXEISDataset:
    """Dataset for EIS parameter prediction with masking - MLX optimized"""

    def __init__(self, data_path: str, masking_patterns: List[List[int]],
                 samples_per_pattern: int = 10000):
        """
        Args:
            data_path: Path to CSV file with columns [Model ID, Resnorm, Rsh, Ra, Rb, Ca, Cb]
            masking_patterns: List of binary masks [1,1,1,0,0] indicating known/unknown params
            samples_per_pattern: Number of samples to generate per masking pattern
        """
        print("Loading dataset...")
        self.data = pd.read_csv(data_path)

        # Filter out reference circuits
        if 'Ground Truth ID' in self.data.columns:
            self.data = self.data[~self.data['Model ID'].str.contains('reference_')]

        # Extract grid indices from Model IDs
        self.data['indices'] = self.data['Model ID'].apply(self._extract_indices)
        self.data = self.data.dropna(subset=['indices'])

        # Build grid mappings
        self.build_grid_mappings()

        # Generate training samples with various masks
        print(f"Generating masked samples with {len(masking_patterns)} patterns...")
        self.samples = self._generate_masked_samples(masking_patterns, samples_per_pattern)
        print(f"Generated {len(self.samples)} training samples")

    def _extract_indices(self, model_id: str) -> List[int]:
        """Extract grid indices from model_id"""
        import re
        # Handle both formats: model_01_02_03_04_05 and model_12_01_02_03_04_05
        match = re.search(r'model_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)(?:_(\d+))?', model_id)
        if match:
            if match.group(6):  # 6-index format
                return [int(match.group(i)) - 1 for i in range(2, 7)]
            else:  # 5-index format
                return [int(match.group(i)) - 1 for i in range(1, 6)]
        return None

    def build_grid_mappings(self):
        """Build logarithmically-spaced grid for each parameter"""
        self.param_names = ['Rsh (Ω)', 'Ra (Ω)', 'Rb (Ω)', 'Ca (F)', 'Cb (F)']
        self.grids = {}

        for param in self.param_names:
            values = self.data[param].values
            values = values[values > 0]
            min_val = np.min(values)
            max_val = np.max(values)
            self.grids[param] = np.logspace(np.log10(min_val), np.log10(max_val), 12)

    def _generate_masked_samples(self, patterns: List[List[int]], n_per_pattern: int):
        """Generate training samples with various masking patterns"""
        samples = []

        for pattern in patterns:
            sample_data = self.data.sample(n=min(n_per_pattern, len(self.data)), replace=True)

            for _, row in sample_data.iterrows():
                params_log = np.array([
                    np.log10(row['Rsh (Ω)']),
                    np.log10(row['Ra (Ω)']),
                    np.log10(row['Rb (Ω)']),
                    np.log10(row['Ca (F)']),
                    np.log10(row['Cb (F)'])
                ], dtype=np.float32)

                masked_params = params_log * np.array(pattern, dtype=np.float32)

                samples.append({
                    'params_log': masked_params,
                    'mask': np.array(pattern, dtype=np.float32),
                    'indices': np.array(row['indices'], dtype=np.int32),
                    'resnorm': np.float32(row['Resnorm'])
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def get_batch(self, indices):
        """Get batch of samples as MLX arrays"""
        batch_params = []
        batch_masks = []
        batch_indices = []
        batch_resnorms = []

        for idx in indices:
            sample = self.samples[idx]
            batch_params.append(sample['params_log'])
            batch_masks.append(sample['mask'])
            batch_indices.append(sample['indices'])
            batch_resnorms.append(sample['resnorm'])

        return (
            mx.array(np.array(batch_params)),
            mx.array(np.array(batch_masks)),
            mx.array(np.array(batch_indices)),
            mx.array(np.array(batch_resnorms))
        )


class ProbabilisticEISPredictorMLX(nn.Module):
    """Neural network for probabilistic EIS parameter prediction - MLX implementation"""

    def __init__(self, n_grid_points: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.n_grid = n_grid_points

        # Encoder: processes masked parameters
        self.encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),  # 5 params + 5 mask indicators
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 128)
        )

        # Parameter prediction heads (classification over grid points)
        self.param_heads = [
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, n_grid_points)
            ) for _ in range(5)
        ]

        # Resnorm prediction head
        self.resnorm_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def __call__(self, params_log, mask):
        """
        Forward pass

        Args:
            params_log: (batch, 5) - log10 parameters, 0 for masked
            mask: (batch, 5) - 1 for known, 0 for unknown

        Returns:
            param_logits: List of 5 arrays (batch, n_grid) - logits for each parameter
            resnorm_pred: (batch, 1) - predicted resnorm values
        """
        # Concatenate parameters and mask
        x = mx.concatenate([params_log, mask], axis=1)

        # Encode
        features = self.encoder(x)

        # Predict logits for each parameter
        param_logits = [head(features) for head in self.param_heads]

        # Predict resnorm
        resnorm_pred = self.resnorm_head(features)

        return param_logits, resnorm_pred


def softmax(x, axis=-1):
    """Softmax function for MLX"""
    exp_x = mx.exp(x - mx.max(x, axis=axis, keepdims=True))
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(logits, targets):
    """Cross entropy loss for MLX"""
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    batch_size = targets.shape[0]
    target_log_probs = log_probs[mx.arange(batch_size), targets]
    return -mx.mean(target_log_probs)


def loss_fn(model, params_log, mask, indices, resnorm):
    """
    Combined loss function for parameter classification and resnorm regression

    Args:
        model: The neural network model
        params_log: Masked parameters in log space
        mask: Binary mask indicating known parameters
        indices: True grid indices for all parameters
        resnorm: True resnorm values

    Returns:
        total_loss: Combined classification + regression loss
    """
    # Forward pass
    param_logits, resnorm_pred = model(params_log, mask)

    # Classification loss for each parameter (only for masked ones)
    ce_loss = mx.array(0.0)
    n_masked = 0

    for i in range(5):
        # Find which samples have this parameter masked (unknown)
        unknown_mask = (mask[:, i] == 0)
        n_unknown = mx.sum(unknown_mask)

        if n_unknown > 0:
            # Get indices where mask is 0 (unknown parameters)
            # Convert to int for indexing
            unknown_mask_int = unknown_mask.astype(mx.int32)

            # Create all indices and filter by mask
            all_indices = mx.arange(mask.shape[0])
            unknown_indices = all_indices * unknown_mask_int

            # Filter out zeros (masked positions)
            unknown_indices = unknown_indices[unknown_indices != 0] if mx.sum(unknown_mask_int) < mask.shape[0] else unknown_indices

            # Alternative: use list comprehension with numpy conversion for now
            import numpy as np
            mask_np = np.array(unknown_mask)
            unknown_idx = np.where(mask_np)[0]
            unknown_indices = mx.array(unknown_idx)

            # Extract logits and targets for unknown parameters using take
            unknown_logits = mx.take(param_logits[i], unknown_indices, axis=0)
            unknown_targets = mx.take(indices[:, i], unknown_indices, axis=0)

            # Compute cross entropy
            ce_loss = ce_loss + cross_entropy_loss(unknown_logits, unknown_targets)
            n_masked += 1

    # Average classification loss across masked parameters
    if n_masked > 0:
        ce_loss = ce_loss / n_masked

    # Resnorm regression loss (MSE)
    resnorm_loss = mx.mean((resnorm_pred.squeeze() - resnorm) ** 2)

    # Combined loss
    total_loss = ce_loss + 0.5 * resnorm_loss

    return total_loss


def compute_accuracy(param_logits, mask, indices):
    """Compute classification accuracy for masked parameters"""
    import numpy as np
    correct = 0
    total = 0

    for i in range(5):
        unknown_mask = (mask[:, i] == 0)
        n_unknown = mx.sum(unknown_mask)

        if n_unknown > 0:
            # Get indices where mask is 0 (unknown parameters)
            # Use numpy for boolean indexing, then convert back to MLX
            mask_np = np.array(unknown_mask)
            unknown_idx = np.where(mask_np)[0]
            unknown_indices = mx.array(unknown_idx)

            # Extract logits and targets using take
            unknown_logits = mx.take(param_logits[i], unknown_indices, axis=0)
            unknown_targets = mx.take(indices[:, i], unknown_indices, axis=0)

            preds = mx.argmax(unknown_logits, axis=1)
            correct += mx.sum(preds == unknown_targets)
            total += n_unknown

    return float(correct) / float(total) if total > 0 else 0.0


def batch_iterate(batch_size, dataset, shuffle=True):
    """Generate batches of data"""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(dataset), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield dataset.get_batch(batch_indices)


def train_model_mlx(model, train_dataset, val_dataset,
                    n_epochs: int = 50,
                    batch_size: int = 512,
                    learning_rate: float = 0.001):
    """
    Train the probabilistic predictor using MLX

    Args:
        model: ProbabilisticEISPredictorMLX instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate

    Returns:
        history: Dictionary containing training metrics
    """

    # Initialize optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)

    # Learning rate scheduler (cosine decay)
    lr_schedule = optim.cosine_decay(learning_rate, n_epochs * (len(train_dataset) // batch_size))

    # Get loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n" + "="*70)
    print("TRAINING ON APPLE SILICON WITH MLX")
    print("="*70)
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print("="*70 + "\n")

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_losses = []
        train_accs = []

        # Progress bar for batches
        n_batches = len(train_dataset) // batch_size
        pbar = tqdm(batch_iterate(batch_size, train_dataset, shuffle=True),
                    total=n_batches, desc=f"Epoch {epoch+1}/{n_epochs}")

        for params_log, mask, indices, resnorm in pbar:
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, params_log, mask, indices, resnorm)

            # Update model parameters
            optimizer.update(model, grads)

            # Force evaluation (MLX lazy computation)
            mx.eval(model.parameters(), optimizer.state)

            # Compute accuracy
            param_logits, _ = model(params_log, mask)
            acc = compute_accuracy(param_logits, mask, indices)

            train_losses.append(float(loss))
            train_accs.append(acc)

            # Update progress bar
            pbar.set_postfix({'loss': f'{float(loss):.4f}', 'acc': f'{acc:.4f}'})

        # Validation phase
        model.eval()
        val_losses = []
        val_accs = []

        for params_log, mask, indices, resnorm in batch_iterate(batch_size, val_dataset, shuffle=False):
            # Forward pass only (no gradients)
            loss = loss_fn(model, params_log, mask, indices, resnorm)
            param_logits, _ = model(params_log, mask)
            acc = compute_accuracy(param_logits, mask, indices)

            val_losses.append(float(loss))
            val_accs.append(acc)

        # Compute epoch statistics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = np.mean(train_accs)
        val_acc = np.mean(val_accs)

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model weights
            model_weights = model.parameters()
            mx.savez('best_eis_predictor_mlx.npz', **dict(tree_flatten(model_weights)))
            print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")

        print()

    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: best_eis_predictor_mlx.npz")

    return history


def tree_flatten(tree):
    """Flatten a nested dictionary/tree structure"""
    flat = []

    def _flatten(t, prefix=""):
        if isinstance(t, dict):
            for k, v in t.items():
                _flatten(v, prefix + k + ".")
        elif isinstance(t, (list, tuple)):
            for i, v in enumerate(t):
                _flatten(v, prefix + str(i) + ".")
        else:
            flat.append((prefix.rstrip("."), t))

    _flatten(tree)
    return flat


class MLXEISParameterCompleter:
    """High-level interface for parameter completion with MLX"""

    def __init__(self, model: ProbabilisticEISPredictorMLX, grids: Dict[str, np.ndarray]):
        self.model = model
        self.model.eval()
        self.grids = grids
        self.param_names = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb']

    def predict_missing_parameters(
        self,
        known_params: Dict[str, float],
        top_k: int = 10
    ) -> Dict:
        """
        Predict missing parameters with probability distributions

        Args:
            known_params: Dict like {'Rsh': 460, 'Ra': 4820, 'Rb': 2210}
            top_k: Number of top predictions to return

        Returns:
            results: Dict containing predictions, probabilities, and confidence metrics
        """
        # Prepare input
        params_log = []
        mask = []
        missing_indices = []

        for i, param_name in enumerate(self.param_names):
            if param_name in known_params:
                params_log.append(np.log10(known_params[param_name]))
                mask.append(1.0)
            else:
                params_log.append(0.0)
                mask.append(0.0)
                missing_indices.append(i)

        # Convert to MLX arrays
        params_tensor = mx.array([params_log], dtype=mx.float32)
        mask_tensor = mx.array([mask], dtype=mx.float32)

        # Predict
        param_logits, resnorm_pred = self.model(params_tensor, mask_tensor)

        # Convert logits to probabilities
        param_probs = [softmax(logits, axis=-1) for logits in param_logits]

        # Extract probabilities for missing parameters
        missing_probs = {}
        for idx in missing_indices:
            probs = np.array(param_probs[idx][0])
            missing_probs[self.param_names[idx]] = probs

        results = {
            'known_params': known_params,
            'missing_params': [self.param_names[i] for i in missing_indices],
            'predicted_resnorm': float(resnorm_pred[0, 0]),
            'marginal_distributions': missing_probs,
            'top_k_predictions': []
        }

        # If 2 parameters are missing, compute joint distribution
        if len(missing_indices) == 2:
            idx1, idx2 = missing_indices
            probs1 = np.array(param_probs[idx1][0])
            probs2 = np.array(param_probs[idx2][0])

            # Joint distribution
            joint = np.outer(probs1, probs2)
            results['joint_distribution'] = joint

            # Get top-k combinations
            flat_joint = joint.flatten()
            top_indices = np.argsort(flat_joint)[-top_k:][::-1]

            param1_name = self.param_names[idx1]
            param2_name = self.param_names[idx2]
            grid1 = self.grids[param1_name + ' (Ω)' if idx1 < 3 else param1_name + ' (F)']
            grid2 = self.grids[param2_name + ' (Ω)' if idx2 < 3 else param2_name + ' (F)']

            for rank, flat_idx in enumerate(top_indices, 1):
                i, j = flat_idx // 12, flat_idx % 12

                prediction = {
                    'rank': rank,
                    param1_name: grid1[i],
                    param2_name: grid2[j],
                    'joint_probability': float(flat_joint[flat_idx]),
                    'marginal_prob_1': float(probs1[i]),
                    'marginal_prob_2': float(probs2[j]),
                    'grid_indices': (int(i), int(j))
                }

                results['top_k_predictions'].append(prediction)

        return results


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("EIS PARAMETER PREDICTION WITH MLX (APPLE SILICON OPTIMIZED)")
    print("="*70)

    # Check if MLX is available
    try:
        test_array = mx.array([1.0, 2.0, 3.0])
        print(f"✓ MLX initialized successfully")
        print(f"✓ Running on Apple Silicon")
    except Exception as e:
        print(f"✗ MLX initialization failed: {e}")
        exit(1)

    # Define masking patterns for training
    masking_patterns = [
        [1, 1, 1, 0, 0],  # Know Rsh, Ra, Rb → Predict Ca, Cb
        [1, 1, 0, 1, 0],  # Know Rsh, Ra, Ca → Predict Rb, Cb
        [1, 0, 1, 1, 0],  # Know Rsh, Rb, Ca → Predict Ra, Cb
        [0, 1, 1, 1, 0],  # Know Ra, Rb, Ca → Predict Rsh, Cb
        [1, 1, 0, 0, 1],  # Know Rsh, Ra, Cb → Predict Rb, Ca
        [1, 0, 1, 0, 1],  # Know Rsh, Rb, Cb → Predict Ra, Ca
        [0, 1, 1, 0, 1],  # Know Ra, Rb, Cb → Predict Rsh, Ca
        [1, 0, 0, 1, 1],  # Know Rsh, Ca, Cb → Predict Ra, Rb
        [0, 1, 0, 1, 1],  # Know Ra, Ca, Cb → Predict Rsh, Rb
        [0, 0, 1, 1, 1],  # Know Rb, Ca, Cb → Predict Rsh, Ra
    ]

    # Load dataset
    print("\nLoading dataset...")
    dataset = MLXEISDataset(
        'eis_training_data/combined_dataset_100gt.csv',
        masking_patterns=masking_patterns,
        samples_per_pattern=5000
    )

    # Split into train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    # Create simple split (MLX doesn't have random_split, so we'll do it manually)
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
    print("\nCreating model...")
    model = ProbabilisticEISPredictorMLX(n_grid_points=12, hidden_dim=256)
    mx.eval(model.parameters())

    # Count parameters
    n_params = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Model parameters: {n_params:,}")

    # Train model
    print("\nStarting training...")
    history = train_model_mlx(
        model,
        train_dataset,
        val_dataset,
        n_epochs=50,
        batch_size=512,
        learning_rate=0.001
    )

    print("\n✓ Training complete!")
    print("Use the trained model for inference with MLXEISParameterCompleter")

"""
EIS Parameter Predictor with Probabilistic Inference
Given 3 known circuit parameters, predicts 2 missing parameters with probabilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class EISDataset(Dataset):
    """Dataset for EIS parameter prediction with masking"""
    
    def __init__(self, data_path: str, masking_patterns: List[List[int]], 
                 samples_per_pattern: int = 10000):
        """
        Args:
            data_path: Path to CSV file with columns [Model ID, Resnorm, Rsh, Ra, Rb, Ca, Cb]
            masking_patterns: List of binary masks [1,1,1,0,0] indicating known/unknown params
            samples_per_pattern: Number of samples to generate per masking pattern
        """
        self.data = pd.read_csv(data_path)
        # Remove reference/ground truth circuits
        self.data = self.data[~self.data['Model ID'].str.contains('reference', na=False)]
        
        # Extract grid indices from Model IDs
        self.data['indices'] = self.data['Model ID'].apply(self._extract_indices)

        # Drop rows where indices couldn't be extracted
        self.data = self.data.dropna(subset=['indices'])

        print(f"Loaded {len(self.data)} valid data points")

        # Build grid mappings (log-spaced parameter values)
        self.build_grid_mappings()
        
        # Generate training samples with various masks
        self.samples = self._generate_masked_samples(masking_patterns, samples_per_pattern)
        
    def _extract_indices(self, model_id: str) -> List[int]:
        """Extract grid indices from model_id like 'model_01_02_03_04_05'"""
        import re
        # Try 5-index format first (model_01_02_03_04_05)
        match = re.search(r'model_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$', model_id)
        if match:
            return [int(match.group(i)) - 1 for i in range(1, 6)]  # Convert to 0-indexed

        # Try 6-index format (model_12_01_02_03_04_05)
        match = re.search(r'model_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)', model_id)
        if match:
            # Skip first index, return indices 2-6
            return [int(match.group(i)) - 1 for i in range(2, 7)]  # Convert to 0-indexed

        return None
    
    def build_grid_mappings(self):
        """Build logarithmically-spaced grid for each parameter"""
        # Estimate ranges from data
        self.param_names = ['Rsh (Ω)', 'Ra (Ω)', 'Rb (Ω)', 'Ca (F)', 'Cb (F)']
        self.grids = {}
        
        for param in self.param_names:
            values = self.data[param].values
            values = values[values > 0]  # Remove any zeros
            min_val = np.min(values)
            max_val = np.max(values)
            
            # Create 12-point log-spaced grid
            self.grids[param] = np.logspace(np.log10(min_val), np.log10(max_val), 12)
    
    def _generate_masked_samples(self, patterns: List[List[int]], n_per_pattern: int):
        """Generate training samples with various masking patterns"""
        samples = []
        
        for pattern in patterns:
            # Sample random rows
            sample_data = self.data.sample(n=min(n_per_pattern, len(self.data)), replace=True)
            
            for _, row in sample_data.iterrows():
                # Extract parameters in log-space
                params_log = np.array([
                    np.log10(row['Rsh (Ω)']),
                    np.log10(row['Ra (Ω)']),
                    np.log10(row['Rb (Ω)']),
                    np.log10(row['Ca (F)']),
                    np.log10(row['Cb (F)'])
                ])
                
                # Apply mask
                masked_params = params_log * np.array(pattern)
                
                samples.append({
                    'params_log': masked_params,
                    'mask': np.array(pattern, dtype=np.float32),
                    'indices': np.array(row['indices'], dtype=np.int64),
                    'resnorm': row['Resnorm']
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['params_log'], dtype=torch.float32),
            torch.tensor(sample['mask'], dtype=torch.float32),
            torch.tensor(sample['indices'], dtype=torch.long),
            torch.tensor(sample['resnorm'], dtype=torch.float32)
        )


class ProbabilisticEISPredictor(nn.Module):
    """Neural network for probabilistic EIS parameter prediction"""
    
    def __init__(self, n_grid_points: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.n_grid = n_grid_points
        
        # Encoder: processes masked parameters
        self.encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),  # 5 params + 5 mask indicators
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 128)
        )
        
        # Parameter prediction heads (classification over grid points)
        self.param_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, n_grid_points)
            ) for _ in range(5)
        ])
        
        # Resnorm prediction head
        self.resnorm_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, params_log, mask):
        """
        Args:
            params_log: (batch, 5) - log10 parameters, 0 for masked
            mask: (batch, 5) - 1 for known, 0 for unknown
        
        Returns:
            param_probs: List of 5 tensors (batch, n_grid) - probability distributions
            resnorm_pred: (batch, 1) - predicted resnorm values
        """
        # Concatenate parameters and mask
        x = torch.cat([params_log, mask], dim=1)
        
        # Encode
        features = self.encoder(x)
        
        # Predict probability distributions for each parameter
        param_logits = [head(features) for head in self.param_heads]
        param_probs = [F.softmax(logits, dim=-1) for logits in param_logits]
        
        # Predict resnorm
        resnorm_pred = self.resnorm_head(features)
        
        return param_probs, resnorm_pred


def train_model(model, train_loader, val_loader, n_epochs: int = 50, 
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the probabilistic predictor"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for params_log, mask, indices, resnorm in train_loader:
            params_log = params_log.to(device)
            mask = mask.to(device)
            indices = indices.to(device)
            resnorm = resnorm.to(device)
            
            # Forward pass
            param_probs, resnorm_pred = model(params_log, mask)
            
            # Loss: weighted combination of classification and regression
            ce_loss = 0
            correct = 0
            total = 0
            
            for i in range(5):
                # Only compute loss for masked (unknown) parameters
                unknown_mask = (mask[:, i] == 0)
                if unknown_mask.sum() > 0:
                    # Classification loss
                    ce_loss += F.cross_entropy(
                        param_probs[i][unknown_mask],
                        indices[:, i][unknown_mask]
                    )
                    
                    # Accuracy
                    preds = param_probs[i][unknown_mask].argmax(dim=1)
                    correct += (preds == indices[:, i][unknown_mask]).sum().item()
                    total += unknown_mask.sum().item()
            
            # Resnorm regression loss
            resnorm_loss = F.mse_loss(resnorm_pred.squeeze(), resnorm)
            
            # Combined loss
            loss = ce_loss + 0.5 * resnorm_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_correct += correct
            train_total += total
        
        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for params_log, mask, indices, resnorm in val_loader:
                params_log = params_log.to(device)
                mask = mask.to(device)
                indices = indices.to(device)
                resnorm = resnorm.to(device)
                
                param_probs, resnorm_pred = model(params_log, mask)
                
                ce_loss = 0
                for i in range(5):
                    unknown_mask = (mask[:, i] == 0)
                    if unknown_mask.sum() > 0:
                        ce_loss += F.cross_entropy(
                            param_probs[i][unknown_mask],
                            indices[:, i][unknown_mask]
                        )
                        preds = param_probs[i][unknown_mask].argmax(dim=1)
                        val_correct += (preds == indices[:, i][unknown_mask]).sum().item()
                        val_total += unknown_mask.sum().item()
                
                resnorm_loss = F.mse_loss(resnorm_pred.squeeze(), resnorm)
                loss = ce_loss + 0.5 * resnorm_loss
                val_losses.append(loss.item())
        
        # Statistics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_eis_predictor.pth')
            print(f"  ✓ New best model saved!")
    
    return history


class EISParameterCompleter:
    """High-level interface for parameter completion with probability distributions"""
    
    def __init__(self, model: ProbabilisticEISPredictor, grids: Dict[str, np.ndarray],
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.grids = grids
        self.device = device
        self.param_names = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb']
    
    def predict_missing_parameters(
        self, 
        known_params: Dict[str, float],
        top_k: int = 10,
        return_joint: bool = True
    ) -> Dict:
        """
        Predict missing parameters with probability distributions
        
        Args:
            known_params: Dict like {'Rsh': 460, 'Ra': 4820, 'Rb': 2210}
            top_k: Number of top predictions to return
            return_joint: Whether to compute joint distribution for 2 missing params
        
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
        
        # Convert to tensors
        params_tensor = torch.tensor([params_log], dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor([mask], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            param_probs, resnorm_pred = self.model(params_tensor, mask_tensor)
        
        # Extract probabilities for missing parameters
        missing_probs = {}
        for idx in missing_indices:
            probs = param_probs[idx][0].cpu().numpy()
            missing_probs[self.param_names[idx]] = probs
        
        results = {
            'known_params': known_params,
            'missing_params': [self.param_names[i] for i in missing_indices],
            'predicted_resnorm': resnorm_pred.item(),
            'marginal_distributions': missing_probs,
            'top_k_predictions': []
        }
        
        # If 2 parameters are missing, compute joint distribution
        if len(missing_indices) == 2 and return_joint:
            idx1, idx2 = missing_indices
            probs1 = param_probs[idx1][0].cpu().numpy()
            probs2 = param_probs[idx2][0].cpu().numpy()
            
            # Joint distribution (independence assumption)
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
                    'joint_probability': flat_joint[flat_idx],
                    'marginal_prob_1': probs1[i],
                    'marginal_prob_2': probs2[j],
                    'grid_indices': (i, j),
                    'confidence_score': self._compute_confidence(probs1, probs2, i, j)
                }
                
                results['top_k_predictions'].append(prediction)
        
        # Single missing parameter case
        elif len(missing_indices) == 1:
            idx = missing_indices[0]
            probs = param_probs[idx][0].cpu().numpy()
            param_name = self.param_names[idx]
            grid = self.grids[param_name + ' (Ω)' if idx < 3 else param_name + ' (F)']
            
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            for rank, i in enumerate(top_indices, 1):
                prediction = {
                    'rank': rank,
                    param_name: grid[i],
                    'probability': probs[i],
                    'grid_index': i,
                    'confidence_score': probs[i]
                }
                results['top_k_predictions'].append(prediction)
        
        # Compute uncertainty metrics
        results['uncertainty'] = self._compute_uncertainty(missing_probs)
        
        return results
    
    def _compute_confidence(self, probs1, probs2, i, j):
        """Compute confidence score for a joint prediction"""
        # Factors: joint probability, marginal certainty, entropy
        joint_prob = probs1[i] * probs2[j]
        entropy1 = entropy(probs1)
        entropy2 = entropy(probs2)
        max_entropy = np.log(12)  # Maximum entropy for uniform distribution
        
        # Normalized confidence (higher is better)
        certainty1 = 1 - (entropy1 / max_entropy)
        certainty2 = 1 - (entropy2 / max_entropy)
        
        confidence = joint_prob * np.sqrt(certainty1 * certainty2)
        return confidence
    
    def _compute_uncertainty(self, missing_probs):
        """Compute uncertainty metrics for missing parameters"""
        uncertainties = {}
        for param_name, probs in missing_probs.items():
            uncertainties[param_name] = {
                'entropy': entropy(probs),
                'max_probability': np.max(probs),
                'top_5_mass': np.sum(np.sort(probs)[-5:]),
                'normalized_entropy': entropy(probs) / np.log(12)
            }
        return uncertainties
    
    def visualize_predictions(self, results: Dict, save_path: Optional[str] = None):
        """Visualize prediction results"""
        if len(results['missing_params']) == 2:
            self._plot_joint_distribution(results, save_path)
        else:
            self._plot_marginal_distribution(results, save_path)
    
    def _plot_joint_distribution(self, results: Dict, save_path: Optional[str]):
        """Plot heatmap of joint distribution for 2 missing parameters"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        param1, param2 = results['missing_params']
        joint = results['joint_distribution']
        
        # Heatmap
        ax = axes[0]
        sns.heatmap(joint, cmap='viridis', ax=ax, cbar_kws={'label': 'Probability'})
        ax.set_xlabel(f'{param2} Grid Index')
        ax.set_ylabel(f'{param1} Grid Index')
        ax.set_title(f'Joint Distribution: P({param1}, {param2} | known params)')
        
        # Top predictions
        ax = axes[1]
        top_preds = results['top_k_predictions'][:10]
        ranks = [p['rank'] for p in top_preds]
        probs = [p['joint_probability'] for p in top_preds]
        
        bars = ax.barh(ranks, probs, color='steelblue')
        ax.set_xlabel('Joint Probability')
        ax.set_ylabel('Rank')
        ax.set_title('Top 10 Predictions')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (rank, prob) in enumerate(zip(ranks, probs)):
            pred = top_preds[i]
            label = f"{pred[param1]:.2e}, {pred[param2]:.2e}"
            ax.text(prob, rank, f' {label}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_marginal_distribution(self, results: Dict, save_path: Optional[str]):
        """Plot bar chart for single missing parameter"""
        param_name = results['missing_params'][0]
        probs = results['marginal_distributions'][param_name]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.arange(12)
        ax.bar(indices, probs, color='steelblue', alpha=0.7)
        ax.set_xlabel('Grid Index')
        ax.set_ylabel('Probability')
        ax.set_title(f'Probability Distribution for {param_name}')
        ax.set_xticks(indices)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
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
    
    # Load and prepare data
    print("Loading dataset...")
    dataset = EISDataset(
        'impedance_data_20250930 1.csv',
        masking_patterns=masking_patterns,
        samples_per_pattern=5000
    )
    
    # Split into train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # Create and train model
    print("Creating model...")
    model = ProbabilisticEISPredictor(n_grid_points=12, hidden_dim=256)
    
    print("Training model...")
    history = train_model(model, train_loader, val_loader, n_epochs=50)
    
    # Create predictor interface
    predictor = EISParameterCompleter(model, dataset.grids)
    
    # Example: Predict Ca and Cb given Rsh, Ra, Rb
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION: Given Rsh=460, Ra=4820, Rb=2210")
    print("="*70)
    
    results = predictor.predict_missing_parameters(
        known_params={'Rsh': 460, 'Ra': 4820, 'Rb': 2210},
        top_k=10
    )
    
    print(f"\nPredicted Resnorm: {results['predicted_resnorm']:.4f}")
    print(f"Missing Parameters: {results['missing_params']}")
    
    print("\nTop 10 Predictions:")
    print("-" * 70)
    for pred in results['top_k_predictions']:
        print(f"\nRank {pred['rank']}:")
        print(f"  {results['missing_params'][0]} = {pred[results['missing_params'][0]]:.4e}")
        print(f"  {results['missing_params'][1]} = {pred[results['missing_params'][1]]:.4e}")
        print(f"  Joint Probability: {pred['joint_probability']:.6f}")
        print(f"  Confidence Score: {pred['confidence_score']:.6f}")
    
    print("\nUncertainty Metrics:")
    print("-" * 70)
    for param, metrics in results['uncertainty'].items():
        print(f"\n{param}:")
        print(f"  Entropy: {metrics['entropy']:.4f}")
        print(f"  Max Probability: {metrics['max_probability']:.4f}")
        print(f"  Top-5 Probability Mass: {metrics['top_5_mass']:.4f}")
    
    # Visualize
    predictor.visualize_predictions(results, save_path='prediction_results.png')
# Machine Learning Strategy for EIS Parameter Prediction with Probabilistic Inference

## Dataset Analysis Summary

### Ground Truth Circuit
- **Rsh**: 460 Ω
- **Ra**: 4820 Ω  
- **Rb**: 2210 Ω
- **Ca**: 3.7 µF
- **Cb**: 3.4 µF

### Grid Structure
- **Total Models**: 125,280 (12^5 combinations)
- **Index Structure**: `[12, i2, i3, i4, i5, i6]` where each index ∈ [1,12]
- **Parameters**: 5 circuit parameters mapped to 12 logarithmically-spaced grid points each
- **Resnorm Range**: 4.07 (best) to 34.58 (worst) based on MAE

## Core ML Task: Probabilistic Parameter Completion

### Problem Statement
**Given**: 3 known parameters (e.g., Rsh, Ra, Rb)  
**Predict**: 2 missing parameters (e.g., Ca, Cb) with probability distribution

### Why This Works Well

1. **Structured Search Space**: 12^2 = 144 combinations for 2 missing parameters
2. **Physical Constraints**: Circuit parameters have correlations (e.g., RC time constants)
3. **Resnorm as Fitness**: Guides probability estimation toward optimal combinations

## Implementation Approach

### Method 1: Direct Probabilistic Neural Network (Recommended)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticParameterPredictor(nn.Module):
    def __init__(self, n_grid_points=12):
        super().__init__()
        self.n_grid = n_grid_points
        
        # Encoder for known parameters
        self.encoder = nn.Sequential(
            nn.Linear(5 + 5, 256),  # 5 params + 5 mask indicators
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Separate heads for each parameter (treating as classification)
        self.param_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_grid_points)  # Logits for 12 grid points
            ) for _ in range(5)
        ])
        
        # Resnorm prediction head
        self.resnorm_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, params_log, mask):
        """
        params_log: (batch, 5) - log10 of parameters, 0 for masked
        mask: (batch, 5) - 1 for known, 0 for unknown
        """
        # Concatenate params and mask
        x = torch.cat([params_log, mask], dim=1)
        
        # Encode
        features = self.encoder(x)
        
        # Predict logits for each parameter
        param_logits = [head(features) for head in self.param_heads]
        param_probs = [F.softmax(logits, dim=-1) for logits in param_logits]
        
        # Predict resnorm
        resnorm_pred = self.resnorm_head(features)
        
        return param_probs, resnorm_pred

# Training
def train_probabilistic_model(model, train_loader, optimizer, device):
    model.train()
    for batch in train_loader:
        params_log, indices, resnorm, mask = batch
        params_log = params_log.to(device)
        indices = indices.to(device)
        resnorm = resnorm.to(device)
        mask = mask.to(device)
        
        # Mask some parameters during training
        masked_params = params_log * mask
        
        # Forward pass
        param_probs, resnorm_pred = model(masked_params, mask)
        
        # Classification loss for each parameter (only for masked ones)
        ce_loss = 0
        for i in range(5):
            if torch.any(mask[:, i] == 0):  # If this param was masked
                ce_loss += F.cross_entropy(
                    param_probs[i][mask[:, i] == 0],
                    indices[:, i][mask[:, i] == 0]
                )
        
        # Resnorm regression loss
        resnorm_loss = F.mse_loss(resnorm_pred.squeeze(), resnorm)
        
        # Combined loss
        loss = ce_loss + 0.5 * resnorm_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item()
```

### Method 2: Conditional Probability with Joint Distribution

```python
class JointProbabilityPredictor(nn.Module):
    """
    Predicts P(param_i, param_j | other_params) for missing parameters
    Accounts for correlations between missing parameters
    """
    def __init__(self, n_grid=12):
        super().__init__()
        self.n_grid = n_grid
        
        self.encoder = nn.Sequential(
            nn.Linear(5 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Joint distribution predictor for pairs
        self.joint_predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_grid * n_grid)  # 144 combinations for 2 params
        )
    
    def forward(self, params_log, mask):
        features = self.encoder(torch.cat([params_log, mask], dim=1))
        
        # Get joint probability distribution
        joint_logits = self.joint_predictor(features)
        joint_logits = joint_logits.view(-1, self.n_grid, self.n_grid)
        joint_probs = F.softmax(joint_logits.view(-1, self.n_grid * self.n_grid), dim=-1)
        joint_probs = joint_probs.view(-1, self.n_grid, self.n_grid)
        
        return joint_probs

def predict_missing_pair(model, known_params, known_mask, grid_mapping):
    """
    Given 3 known parameters, predict distribution over 2 missing parameters
    
    Returns:
        joint_probs: (12, 12) probability matrix
        top_k_predictions: List of (param1_val, param2_val, probability)
    """
    model.eval()
    with torch.no_grad():
        # Prepare input
        params_tensor = torch.tensor(known_params).float().unsqueeze(0)
        mask_tensor = torch.tensor(known_mask).float().unsqueeze(0)
        
        # Get joint probability distribution
        joint_probs = model(params_tensor, mask_tensor)
        joint_probs = joint_probs.squeeze(0).cpu().numpy()
        
        # Get top-k predictions
        flat_probs = joint_probs.flatten()
        top_k_indices = np.argsort(flat_probs)[-10:][::-1]
        
        predictions = []
        for idx in top_k_indices:
            i, j = idx // 12, idx % 12
            param1_val = grid_mapping[0][i]  # Map index to actual value
            param2_val = grid_mapping[1][j]
            prob = flat_probs[idx]
            predictions.append((param1_val, param2_val, prob))
        
        return joint_probs, predictions

# Example usage
known_params = [np.log10(460), np.log10(4820), np.log10(2210), 0, 0]  # Rsh, Ra, Rb known
known_mask = [1, 1, 1, 0, 0]  # First 3 known, last 2 unknown

joint_probs, top_predictions = predict_missing_pair(
    model, known_params, known_mask, grid_mapping
)

print("Top 10 predictions for (Ca, Cb):")
for ca, cb, prob in top_predictions:
    print(f"Ca={ca:.2e} F, Cb={cb:.2e} F, P={prob:.4f}")
```

### Method 3: Ensemble with Uncertainty Quantification

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy

class EnsembleParameterPredictor:
    def __init__(self, n_estimators=100, n_grid=12):
        self.n_grid = n_grid
        # Separate classifier for each parameter
        self.models = {
            'rsh': RandomForestClassifier(n_estimators=n_estimators, max_depth=10),
            'ra': RandomForestClassifier(n_estimators=n_estimators, max_depth=10),
            'rb': RandomForestClassifier(n_estimators=n_estimators, max_depth=10),
            'ca': RandomForestClassifier(n_estimators=n_estimators, max_depth=10),
            'cb': RandomForestClassifier(n_estimators=n_estimators, max_depth=10)
        }
        
    def train(self, X, y_indices, missing_pattern):
        """
        Train with specific missing pattern
        X: Known parameters (with 0 for missing)
        y_indices: True grid indices for all parameters
        missing_pattern: Which parameters to predict
        """
        for i, param_name in enumerate(['rsh', 'ra', 'rb', 'ca', 'cb']):
            if missing_pattern[i]:  # This parameter should be predicted
                self.models[param_name].fit(X, y_indices[:, i])
    
    def predict_with_probability(self, known_params, missing_mask):
        """
        Predict missing parameters with probability
        
        Returns:
            predictions: Dict of {param_name: (index, probability_distribution)}
        """
        predictions = {}
        X_input = np.array(known_params).reshape(1, -1)
        
        for i, param_name in enumerate(['rsh', 'ra', 'rb', 'ca', 'cb']):
            if missing_mask[i] == 0:  # This parameter is missing
                # Get probability distribution from forest
                proba = self.models[param_name].predict_proba(X_input)[0]
                
                # Reshape to grid size
                full_proba = np.zeros(self.n_grid)
                classes = self.models[param_name].classes_
                full_proba[classes] = proba
                
                # Normalize
                full_proba /= full_proba.sum()
                
                predictions[param_name] = full_proba
        
        return predictions
    
    def predict_joint_distribution(self, known_params, missing_params):
        """
        Predict joint distribution for 2 missing parameters
        Accounts for independence assumption or learned correlations
        """
        predictions = self.predict_with_probability(known_params, missing_params)
        
        if len(predictions) == 2:
            # Get the two parameter names
            param_names = list(predictions.keys())
            proba1 = predictions[param_names[0]]
            proba2 = predictions[param_names[1]]
            
            # Joint distribution (assuming independence)
            joint = np.outer(proba1, proba2)
            
            # Could refine with learned correlations
            # joint = self._apply_correlation_correction(joint, param_names)
            
            return joint, param_names
        
        return None, None
    
    def get_uncertainty(self, predictions):
        """
        Quantify prediction uncertainty using entropy
        High entropy = high uncertainty
        """
        uncertainties = {}
        for param_name, proba in predictions.items():
            uncertainties[param_name] = entropy(proba)
        return uncertainties

# Example: Predict Ca and Cb given Rsh, Ra, Rb
ensemble = EnsembleParameterPredictor(n_estimators=200, n_grid=12)

# Train on full dataset with various masking patterns
# ... training code ...

# Predict
known = [np.log10(460), np.log10(4820), np.log10(2210), 0, 0]
mask = [1, 1, 1, 0, 0]

predictions = ensemble.predict_with_probability(known, mask)
uncertainties = ensemble.get_uncertainty(predictions)

print("Ca predictions:")
for idx, prob in enumerate(predictions['ca']):
    if prob > 0.01:  # Only show meaningful probabilities
        print(f"  Index {idx}: {prob:.4f}")
print(f"Uncertainty (entropy): {uncertainties['ca']:.3f}")

print("\nCb predictions:")
for idx, prob in enumerate(predictions['cb']):
    if prob > 0.01:
        print(f"  Index {idx}: {prob:.4f}")
print(f"Uncertainty (entropy): {uncertainties['cb']:.3f}")

# Get joint distribution
joint_dist, param_names = ensemble.predict_joint_distribution(known, mask)
print(f"\nJoint distribution for {param_names}:")
print(f"Shape: {joint_dist.shape}")

# Top 5 combinations
flat_joint = joint_dist.flatten()
top_indices = np.argsort(flat_joint)[-5:][::-1]
for idx in top_indices:
    i, j = idx // 12, idx % 12
    print(f"  ({param_names[0]}[{i}], {param_names[1]}[{j}]): P={flat_joint[idx]:.4f}")
```

## Practical Workflow

### Step 1: Data Preparation with Multiple Masking Patterns

```python
def generate_training_data_with_masks(data, n_samples_per_pattern=10000):
    """
    Generate training data with various masking patterns
    """
    # Define masking patterns (1=keep, 0=mask)
    patterns = [
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
    
    training_samples = []
    for pattern in patterns:
        # Sample random models
        samples = data.sample(n_samples_per_pattern)
        
        for _, row in samples.iterrows():
            params_log = [
                np.log10(row['Rsh (Ω)']),
                np.log10(row['Ra (Ω)']),
                np.log10(row['Rb (Ω)']),
                np.log10(row['Ca (F)']),
                np.log10(row['Cb (F)'])
            ]
            
            # Apply mask
            masked_params = [p * m for p, m in zip(params_log, pattern)]
            
            training_samples.append({
                'params': masked_params,
                'mask': pattern,
                'indices': extract_indices(row['Model ID']),
                'resnorm': row['Resnorm']
            })
    
    return training_samples
```

### Step 2: Inference with Confidence Scores

```python
def predict_missing_parameters_with_confidence(
    model, 
    known_params,  # e.g., {'Rsh': 460, 'Ra': 4820, 'Rb': 2210}
    grid_definitions,
    top_k=10
):
    """
    Complete prediction workflow with confidence metrics
    """
    # Prepare input
    param_order = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb']
    params_log = []
    mask = []
    
    for param_name in param_order:
        if param_name in known_params:
            params_log.append(np.log10(known_params[param_name]))
            mask.append(1)
        else:
            params_log.append(0)
            mask.append(0)
    
    # Get predictions
    param_probs, resnorm_pred = model(
        torch.tensor([params_log]).float(),
        torch.tensor([mask]).float()
    )
    
    # Extract missing parameter names
    missing_params = [param_order[i] for i, m in enumerate(mask) if m == 0]
    
    if len(missing_params) == 2:
        # Joint distribution for 2 missing parameters
        idx1 = param_order.index(missing_params[0])
        idx2 = param_order.index(missing_params[1])
        
        probs1 = param_probs[idx1].detach().numpy()[0]
        probs2 = param_probs[idx2].detach().numpy()[0]
        
        # Compute joint (can be refined with correlation model)
        joint_probs = np.outer(probs1, probs2)
        
        # Get top-k combinations
        flat_probs = joint_probs.flatten()
        top_indices = np.argsort(flat_probs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            i, j = idx // 12, idx % 12
            
            # Map indices to actual parameter values
            val1 = grid_definitions[missing_params[0]][i]
            val2 = grid_definitions[missing_params[1]][j]
            prob = flat_probs[idx]
            
            # Confidence metrics
            confidence = {
                'joint_probability': prob,
                'marginal_prob_1': probs1[i],
                'marginal_prob_2': probs2[j],
                'predicted_resnorm': resnorm_pred.item(),
                'entropy_1': entropy(probs1),
                'entropy_2': entropy(probs2)
            }
            
            results.append({
                missing_params[0]: val1,
                missing_params[1]: val2,
                'confidence': confidence
            })
        
        return results
    
    return None

# Usage example
grid_defs = {
    'Ca': np.logspace(-7, -4.3, 12),  # 0.1 to 50 µF
    'Cb': np.logspace(-6, -4.3, 12)   # 1 to 50 µF
}

predictions = predict_missing_parameters_with_confidence(
    model,
    known_params={'Rsh': 460, 'Ra': 4820, 'Rb': 2210},
    grid_definitions=grid_defs,
    top_k=10
)

print("Top 10 predictions for missing Ca and Cb:")
print("="*70)
for i, pred in enumerate(predictions, 1):
    print(f"\n{i}. Ca = {pred['Ca']:.2e} F, Cb = {pred['Cb']:.2e} F")
    print(f"   Joint Probability: {pred['confidence']['joint_probability']:.4f}")
    print(f"   Marginal Ca: {pred['confidence']['marginal_prob_1']:.4f}")
    print(f"   Marginal Cb: {pred['confidence']['marginal_prob_2']:.4f}")
    print(f"   Predicted Resnorm: {pred['confidence']['predicted_resnorm']:.3f}")
    print(f"   Uncertainty (Ca): {pred['confidence']['entropy_1']:.3f}")
    print(f"   Uncertainty (Cb): {pred['confidence']['entropy_2']:.3f}")
```

## Key Advantages of This Approach

1. **Principled Probability**: Not just point estimates, but full distributions
2. **Uncertainty Quantification**: Entropy and variance metrics show confidence
3. **Top-K Predictions**: Explore multiple plausible parameter combinations
4. **Joint Distributions**: Capture correlations between missing parameters
5. **Flexible**: Works with any combination of 3 known / 2 unknown parameters
6. **Interpretable**: Grid-based predictions map to actual parameter values

## Next Steps

1. Train model on your 125k dataset with multiple masking patterns
2. Validate on held-out circuits with known ground truth
3. Deploy as parameter suggestion system for new measurements
4. Integrate with optimization loop for refinement
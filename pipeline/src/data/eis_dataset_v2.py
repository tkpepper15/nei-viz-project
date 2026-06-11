#!/usr/bin/env python3
"""
EIS Dataset V2 - Properly normalized for all 12 parameters including Bertrand coefficients.

Fixes the normalization mismatch for N1, N0, D1 by computing dataset-wide
statistics for all parameters in log10 space.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


class EISDatasetV2(Dataset):
    """
    EIS Dataset with proper normalization for all 12 parameters.

    Parameters are normalized to ~N(0,1) in log10 space:
    - Base (5): Ra, Rb, Ca, Cb, Rsh
    - Derived (4): TER, TEC, tau_a, tau_b
    - Bertrand (3): N1, N0, D1

    This ensures all parameters have similar scale for training.
    """

    PARAM_NAMES = ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh', 'TER', 'TEC', 'tau_a', 'tau_b', 'N1', 'N0', 'D1']
    BASE_PARAMS = ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh']
    DERIVED_PARAMS = ['TER', 'TEC', 'tau_a', 'tau_b']
    BERTRAND_PARAMS = ['N1', 'N0', 'D1']

    def __init__(self, csv_path: str, metadata_path: str, precompute_stats: bool = True):
        """
        Args:
            csv_path: Path to CSV file with EIS data
            metadata_path: Path to metadata JSON
            precompute_stats: If True, compute normalization stats from data
        """
        self.df = pd.read_csv(csv_path)

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.sample_ids = self.df['sample_id'].unique()

        # Compute all 12 parameter statistics
        self._compute_all_stats()

        if precompute_stats:
            self._print_stats()

    def _compute_bertrand_coefficients(self, Ra, Rb, Ca, Cb, Rsh):
        """Compute Bertrand system theory coefficients."""
        tau_a = Ra * Ca
        tau_b = Rb * Cb

        # N1 = 1/Ca + 1/Cb (inverse capacitance sum)
        N1 = 1.0 / Ca + 1.0 / Cb

        # N0 = (1/Ra + 1/Rb) / (Ca * Cb) (mixed conductance-capacitance)
        N0 = (1.0 / Ra + 1.0 / Rb) / (Ca * Cb)

        # D1 = N1/Rsh + 1/tau_a + 1/tau_b (damping coefficient)
        D1 = N1 / Rsh + 1.0 / tau_a + 1.0 / tau_b

        return N1, N0, D1

    def _compute_all_stats(self):
        """Compute normalization statistics for all 12 parameters."""
        # Get unique samples
        param_df = self.df.groupby('sample_id').first()[self.BASE_PARAMS]

        # Base parameters (already in dataframe)
        Ra = param_df['Ra'].values
        Rb = param_df['Rb'].values
        Ca = param_df['Ca'].values
        Cb = param_df['Cb'].values
        Rsh = param_df['Rsh'].values

        # Derived parameters
        TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb)
        TEC = (Ca * Cb) / (Ca + Cb)
        tau_a = Ra * Ca
        tau_b = Rb * Cb

        # Bertrand coefficients
        N1, N0, D1 = self._compute_bertrand_coefficients(Ra, Rb, Ca, Cb, Rsh)

        # Stack all parameters
        all_params = np.column_stack([
            Ra, Rb, Ca, Cb, Rsh,  # Base
            TER, TEC, tau_a, tau_b,  # Derived
            N1, N0, D1  # Bertrand
        ])

        # Compute log10 statistics
        all_params_log = np.log10(all_params + 1e-30)  # Avoid log(0)

        self.param_log_mean = all_params_log.mean(axis=0)  # (12,)
        self.param_log_std = all_params_log.std(axis=0)    # (12,)

        # Ensure no zero std
        self.param_log_std = np.maximum(self.param_log_std, 0.01)

        # Store as tensors
        self.param_log_mean_tensor = torch.FloatTensor(self.param_log_mean)
        self.param_log_std_tensor = torch.FloatTensor(self.param_log_std)

        # Also store base-only stats for compatibility
        self.base_log_mean = self.param_log_mean[:5]
        self.base_log_std = self.param_log_std[:5]

    def _print_stats(self):
        """Print normalization statistics."""
        print(f"\nDataset: {len(self.sample_ids):,} samples")
        print(f"\nNormalization statistics (log10 space):")
        print(f"{'Parameter':<10} {'Mean':>10} {'Std':>10} {'Physical Range'}")
        print("-" * 50)

        for i, name in enumerate(self.PARAM_NAMES):
            mean = self.param_log_mean[i]
            std = self.param_log_std[i]
            phys_low = 10 ** (mean - 2*std)
            phys_high = 10 ** (mean + 2*std)
            print(f"{name:<10} {mean:>10.3f} {std:>10.3f} [{phys_low:.2e}, {phys_high:.2e}]")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        sample_df = self.df[self.df['sample_id'] == sample_id].sort_values('frequency')

        # Impedance data
        Z_real = sample_df['Z_real'].values
        Z_imag = sample_df['Z_imag'].values
        Z_mag = sample_df['Z_mag'].values
        phase = sample_df['phase'].values
        frequencies = sample_df['frequency'].values

        # Impedance features (log-transformed)
        log_Z_real = np.sign(Z_real) * np.log10(np.abs(Z_real) + 1e-12)
        log_Z_imag = np.sign(Z_imag) * np.log10(np.abs(Z_imag) + 1e-12)
        log_Z_mag = np.log10(Z_mag + 1e-12)

        impedance_features = np.stack([log_Z_real, log_Z_imag, log_Z_mag, phase], axis=1)
        log_freq = np.log10(frequencies).reshape(-1, 1)

        # Base parameters
        Ra = sample_df['Ra'].iloc[0]
        Rb = sample_df['Rb'].iloc[0]
        Ca = sample_df['Ca'].iloc[0]
        Cb = sample_df['Cb'].iloc[0]
        Rsh = sample_df['Rsh'].iloc[0]

        # Derived parameters
        TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb)
        TEC = (Ca * Cb) / (Ca + Cb)
        tau_a = Ra * Ca
        tau_b = Rb * Cb

        # Bertrand coefficients
        N1, N0, D1 = self._compute_bertrand_coefficients(Ra, Rb, Ca, Cb, Rsh)

        # All 12 parameters in physical space
        params_phys = np.array([Ra, Rb, Ca, Cb, Rsh, TER, TEC, tau_a, tau_b, N1, N0, D1])

        # Normalize all 12 parameters to ~N(0,1) in log10 space
        params_log = np.log10(params_phys + 1e-30)
        params_norm = (params_log - self.param_log_mean) / self.param_log_std

        return {
            'impedance_features': torch.FloatTensor(impedance_features),
            'log_freq': torch.FloatTensor(log_freq),
            'params_norm': torch.FloatTensor(params_norm),  # All 12 normalized
            'params_phys': torch.FloatTensor(params_phys),  # All 12 physical
            'params_log': torch.FloatTensor(params_log),    # All 12 log10
            # Individual derived for loss computation
            'ter': torch.FloatTensor([params_norm[5]]),     # Normalized
            'tec': torch.FloatTensor([params_norm[6]]),
            'tau_a': torch.FloatTensor([params_norm[7]]),
            'tau_b': torch.FloatTensor([params_norm[8]]),
            # Complex impedance
            'Z_complex': torch.complex(torch.FloatTensor(Z_real), torch.FloatTensor(Z_imag)),
            'frequencies': torch.FloatTensor(frequencies),
            # Normalization stats (all 12)
            'param_log_mean': self.param_log_mean_tensor.clone(),
            'param_log_std': self.param_log_std_tensor.clone(),
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, val, test dataloaders with proper collation.

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv, metadata.json
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    train_dataset = EISDatasetV2(
        str(data_dir / 'train.csv'),
        str(data_dir / 'metadata.json')
    )
    val_dataset = EISDatasetV2(
        str(data_dir / 'val.csv'),
        str(data_dir / 'metadata.json'),
        precompute_stats=False
    )
    # Use train stats for val/test
    val_dataset.param_log_mean = train_dataset.param_log_mean
    val_dataset.param_log_std = train_dataset.param_log_std
    val_dataset.param_log_mean_tensor = train_dataset.param_log_mean_tensor
    val_dataset.param_log_std_tensor = train_dataset.param_log_std_tensor

    test_dataset = EISDatasetV2(
        str(data_dir / 'test.csv'),
        str(data_dir / 'metadata.json'),
        precompute_stats=False
    )
    test_dataset.param_log_mean = train_dataset.param_log_mean
    test_dataset.param_log_std = train_dataset.param_log_std
    test_dataset.param_log_mean_tensor = train_dataset.param_log_mean_tensor
    test_dataset.param_log_std_tensor = train_dataset.param_log_std_tensor

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_eis_batch
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_eis_batch
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_eis_batch
    )

    return train_loader, val_loader, test_loader


def collate_eis_batch(batch):
    """
    Collate function for variable-length EIS sequences.

    Pads sequences to max length in batch.
    """
    max_len = max(item['impedance_features'].shape[0] for item in batch)
    batch_size = len(batch)

    # Variable length tensors (pad to max_len)
    impedance_features = torch.zeros(batch_size, max_len, 4)
    log_freq = torch.zeros(batch_size, max_len, 1)
    Z_complex = torch.zeros(batch_size, max_len, dtype=torch.complex64)
    frequencies = torch.zeros(batch_size, max_len)

    # Fixed size tensors
    params_norm = torch.stack([item['params_norm'] for item in batch])
    params_phys = torch.stack([item['params_phys'] for item in batch])
    params_log = torch.stack([item['params_log'] for item in batch])
    ter = torch.stack([item['ter'] for item in batch])
    tec = torch.stack([item['tec'] for item in batch])
    tau_a = torch.stack([item['tau_a'] for item in batch])
    tau_b = torch.stack([item['tau_b'] for item in batch])

    # Normalization stats (expand to batch size)
    param_log_mean = batch[0]['param_log_mean'].unsqueeze(0).expand(batch_size, -1)
    param_log_std = batch[0]['param_log_std'].unsqueeze(0).expand(batch_size, -1)

    # Pad variable length sequences
    for i, item in enumerate(batch):
        seq_len = item['impedance_features'].shape[0]
        impedance_features[i, :seq_len] = item['impedance_features']
        log_freq[i, :seq_len] = item['log_freq']
        Z_complex[i, :seq_len] = item['Z_complex']
        frequencies[i, :seq_len] = item['frequencies']

    return {
        'impedance_features': impedance_features,
        'log_freq': log_freq,
        'params_norm': params_norm,
        'params_phys': params_phys,
        'params_log': params_log,
        'ter': ter,
        'tec': tec,
        'tau_a': tau_a,
        'tau_b': tau_b,
        'Z_complex': Z_complex,
        'frequencies': frequencies,
        'param_log_mean': param_log_mean,
        'param_log_std': param_log_std,
    }


if __name__ == '__main__':
    # Test dataset
    print("Testing EISDatasetV2...")

    dataset = EISDatasetV2(
        'data/physics_constrained_corrected/train.csv',
        'data/physics_constrained_corrected/metadata.json'
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single item
    item = dataset[0]
    print(f"\nSample item:")
    for key, val in item.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    # Test dataloader
    train_loader, val_loader, _ = create_dataloaders(
        'data/physics_constrained_corrected',
        batch_size=4
    )

    batch = next(iter(train_loader))
    print(f"\nBatch:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}")

    # Verify normalization
    print(f"\nNormalization check (should be ~0 mean, ~1 std):")
    params_norm = batch['params_norm']
    print(f"  Mean: {params_norm.mean(dim=0).numpy()}")
    print(f"  Std:  {params_norm.std(dim=0).numpy()}")

    print("\nDataset tests passed!")

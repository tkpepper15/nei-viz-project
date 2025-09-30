"""
Automated Dataset Generation System for Multi-Ground-Truth EIS Training
Generates comprehensive datasets across diverse circuit configurations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
import hashlib


@dataclass
class GroundTruthConfig:
    """Configuration for a ground truth circuit"""
    id: str
    rsh: float  # Ohms
    ra: float   # Ohms
    rb: float   # Ohms
    ca: float   # Farads
    cb: float   # Farads
    
    def to_dict(self):
        return asdict(self)
    
    def get_hash(self):
        """Generate unique hash for this configuration"""
        config_str = f"{self.rsh:.6e}_{self.ra:.6e}_{self.rb:.6e}_{self.ca:.6e}_{self.cb:.6e}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class GroundTruthSampler:
    """Generates diverse ground truth configurations using various sampling strategies"""
    
    def __init__(self, param_ranges: Dict[str, Tuple[float, float]], n_frequencies: int = 50):
        """
        Args:
            param_ranges: Dict with keys 'rsh', 'ra', 'rb', 'ca', 'cb' and (min, max) tuples
            n_frequencies: Number of frequency points for EIS computation
        """
        self.param_ranges = param_ranges
        self.n_frequencies = n_frequencies
        
    def generate_latin_hypercube_samples(self, n_samples: int) -> List[GroundTruthConfig]:
        """
        Latin Hypercube Sampling for optimal parameter space coverage
        Ensures no two samples share the same projection on any parameter axis
        """
        from scipy.stats import qmc
        
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=5, seed=42)
        samples = sampler.random(n=n_samples)
        
        # Transform from [0,1] to log-space parameter ranges
        configs = []
        param_names = ['rsh', 'ra', 'rb', 'ca', 'cb']
        
        for i, sample in enumerate(samples):
            params = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.param_ranges[param_name]
                # Log-space interpolation
                log_val = np.log10(min_val) + sample[j] * (np.log10(max_val) - np.log10(min_val))
                params[param_name] = 10 ** log_val
            
            config = GroundTruthConfig(
                id=f"lhs_{i:04d}",
                rsh=params['rsh'],
                ra=params['ra'],
                rb=params['rb'],
                ca=params['ca'],
                cb=params['cb']
            )
            configs.append(config)
        
        return configs
    
    def generate_sobol_samples(self, n_samples: int) -> List[GroundTruthConfig]:
        """
        Sobol Sequence for quasi-random low-discrepancy sampling
        Better uniformity than random sampling
        """
        from scipy.stats import qmc
        
        sampler = qmc.Sobol(d=5, scramble=True, seed=42)
        samples = sampler.random(n=n_samples)
        
        configs = []
        param_names = ['rsh', 'ra', 'rb', 'ca', 'cb']
        
        for i, sample in enumerate(samples):
            params = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.param_ranges[param_name]
                log_val = np.log10(min_val) + sample[j] * (np.log10(max_val) - np.log10(min_val))
                params[param_name] = 10 ** log_val
            
            config = GroundTruthConfig(
                id=f"sobol_{i:04d}",
                rsh=params['rsh'],
                ra=params['ra'],
                rb=params['rb'],
                ca=params['ca'],
                cb=params['cb']
            )
            configs.append(config)
        
        return configs
    
    def generate_grid_samples(self, n_per_dim: int = 3) -> List[GroundTruthConfig]:
        """
        Regular grid sampling for systematic coverage
        n_per_dim=3 gives 3^5 = 243 samples
        """
        param_names = ['rsh', 'ra', 'rb', 'ca', 'cb']
        
        # Create log-spaced grid points for each parameter
        grids = {}
        for param_name in param_names:
            min_val, max_val = self.param_ranges[param_name]
            grids[param_name] = np.logspace(np.log10(min_val), np.log10(max_val), n_per_dim)
        
        # Generate all combinations
        configs = []
        idx = 0
        
        for rsh in grids['rsh']:
            for ra in grids['ra']:
                for rb in grids['rb']:
                    for ca in grids['ca']:
                        for cb in grids['cb']:
                            config = GroundTruthConfig(
                                id=f"grid_{idx:04d}",
                                rsh=rsh, ra=ra, rb=rb, ca=ca, cb=cb
                            )
                            configs.append(config)
                            idx += 1
        
        return configs
    
    def generate_biologically_relevant_samples(self, n_samples: int) -> List[GroundTruthConfig]:
        """
        Sample from distributions typical for RPE tissue
        Based on literature values for retinal pigment epithelium
        """
        configs = []
        
        # RPE-specific parameter distributions (example ranges)
        rsh_dist = np.random.lognormal(np.log(500), 0.5, n_samples)  # Mean ~500 Ω
        ra_dist = np.random.lognormal(np.log(3000), 0.6, n_samples)  # Mean ~3000 Ω
        rb_dist = np.random.lognormal(np.log(2000), 0.6, n_samples)  # Mean ~2000 Ω
        ca_dist = np.random.lognormal(np.log(3e-6), 0.4, n_samples)  # Mean ~3 µF
        cb_dist = np.random.lognormal(np.log(3e-6), 0.4, n_samples)  # Mean ~3 µF
        
        for i in range(n_samples):
            # Clamp to valid ranges
            rsh = np.clip(rsh_dist[i], *self.param_ranges['rsh'])
            ra = np.clip(ra_dist[i], *self.param_ranges['ra'])
            rb = np.clip(rb_dist[i], *self.param_ranges['rb'])
            ca = np.clip(ca_dist[i], *self.param_ranges['ca'])
            cb = np.clip(cb_dist[i], *self.param_ranges['cb'])
            
            config = GroundTruthConfig(
                id=f"bio_{i:04d}",
                rsh=rsh, ra=ra, rb=rb, ca=ca, cb=cb
            )
            configs.append(config)
        
        return configs
    
    def generate_edge_case_samples(self) -> List[GroundTruthConfig]:
        """
        Generate edge cases at parameter space boundaries
        Tests model robustness at extremes
        """
        configs = []
        param_names = ['rsh', 'ra', 'rb', 'ca', 'cb']
        
        # Corner samples (all min/max combinations)
        for i in range(2**5):
            params = {}
            binary = format(i, '05b')
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.param_ranges[param_name]
                params[param_name] = max_val if binary[j] == '1' else min_val
            
            config = GroundTruthConfig(
                id=f"edge_{i:03d}",
                **params
            )
            configs.append(config)
        
        return configs
    
    def generate_combined_dataset(self, n_total: int = 100) -> List[GroundTruthConfig]:
        """
        Generate diverse dataset combining multiple sampling strategies
        
        Recommended distribution for n_total=100:
        - 40% Latin Hypercube (optimal coverage)
        - 30% Sobol (quasi-random)
        - 15% Biologically relevant (RPE-specific)
        - 10% Grid (systematic)
        - 5% Edge cases (robustness)
        """
        configs = []
        
        # Calculate sample sizes for each strategy
        n_lhs = int(0.40 * n_total)
        n_sobol = int(0.30 * n_total)
        n_bio = int(0.15 * n_total)
        n_grid_per_dim = 2  # 2^5 = 32 samples
        n_edge = 32  # All corners
        
        print(f"Generating {n_total} ground truth configurations...")
        print(f"  Latin Hypercube: {n_lhs}")
        print(f"  Sobol Sequence: {n_sobol}")
        print(f"  Biologically Relevant: {n_bio}")
        print(f"  Grid Samples: {n_grid_per_dim**5}")
        print(f"  Edge Cases: {n_edge}")
        
        # Generate samples
        configs.extend(self.generate_latin_hypercube_samples(n_lhs))
        configs.extend(self.generate_sobol_samples(n_sobol))
        configs.extend(self.generate_biologically_relevant_samples(n_bio))
        configs.extend(self.generate_grid_samples(n_per_dim=n_grid_per_dim))
        configs.extend(self.generate_edge_case_samples())
        
        # Take exactly n_total (may have slight overage)
        configs = configs[:n_total]
        
        print(f"\nGenerated {len(configs)} unique configurations")
        return configs


class EISComputationEngine:
    """Computes impedance spectra and resnorm values"""
    
    def __init__(self, frequency_range: Tuple[float, float] = (1.0, 1e6), 
                 n_frequencies: int = 50):
        """
        Args:
            frequency_range: (min_freq, max_freq) in Hz
            n_frequencies: Number of frequency points
        """
        self.frequencies = np.logspace(
            np.log10(frequency_range[0]), 
            np.log10(frequency_range[1]), 
            n_frequencies
        )
        self.omega = 2 * np.pi * self.frequencies
    
    def compute_impedance_spectrum(self, config: GroundTruthConfig) -> np.ndarray:
        """
        Compute complex impedance spectrum Z(ω) for given circuit parameters
        Z(ω) = Rs + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)
        """
        omega = self.omega
        
        # Parallel RC branches
        z_a = config.ra / (1 + 1j * omega * config.ra * config.ca)
        z_b = config.rb / (1 + 1j * omega * config.rb * config.cb)
        
        # Total impedance
        z_total = config.rsh + z_a + z_b
        
        return z_total
    
    def compute_resnorm(self, z_measured: np.ndarray, z_model: np.ndarray) -> float:
        """
        Compute residual norm (MAE) between measured and model spectra
        Uses magnitude and phase with low-frequency weighting
        """
        # Magnitude error
        mag_measured = np.abs(z_measured)
        mag_model = np.abs(z_model)
        mag_error = np.abs(mag_measured - mag_model)
        
        # Phase error
        phase_measured = np.angle(z_measured)
        phase_model = np.angle(z_model)
        phase_error = np.abs(phase_measured - phase_model)
        
        # Frequency weighting (emphasize low frequencies for biological systems)
        weights = 1.0 / np.sqrt(self.frequencies)
        weights = weights / weights.sum()
        
        # Weighted MAE
        resnorm = np.sum(weights * (mag_error + phase_error * mag_measured))
        
        return resnorm


class DatasetGenerator:
    """Main orchestrator for generating multi-ground-truth datasets"""
    
    def __init__(self, output_dir: str = "./datasets", n_grid_points: int = 12):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_grid = n_grid_points
        
        # Define parameter ranges (adjust based on your application)
        self.param_ranges = {
            'rsh': (10.0, 10000.0),      # 10 Ω to 10 kΩ
            'ra': (10.0, 10000.0),       # 10 Ω to 10 kΩ
            'rb': (10.0, 10000.0),       # 10 Ω to 10 kΩ
            'ca': (1e-7, 5e-5),          # 0.1 µF to 50 µF
            'cb': (1e-7, 5e-5),          # 0.1 µF to 50 µF
        }
        
        # Create grid mappings
        self.grids = self._create_parameter_grids()
        
        # Initialize engines
        self.sampler = GroundTruthSampler(self.param_ranges)
        self.eis_engine = EISComputationEngine()
    
    def _create_parameter_grids(self) -> Dict[str, np.ndarray]:
        """Create log-spaced grids for each parameter"""
        grids = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            grids[param_name] = np.logspace(
                np.log10(min_val), 
                np.log10(max_val), 
                self.n_grid
            )
        return grids
    
    def generate_grid_for_ground_truth(self, ground_truth: GroundTruthConfig) -> pd.DataFrame:
        """
        Generate complete 12^5 parameter grid and compute resnorm for each model
        
        Returns:
            DataFrame with columns [Model ID, Resnorm, Rsh, Ra, Rb, Ca, Cb]
        """
        # Compute reference spectrum for ground truth
        z_reference = self.eis_engine.compute_impedance_spectrum(ground_truth)
        
        # Generate all grid combinations
        results = []
        
        # Add ground truth as first row
        results.append({
            'Model ID': f'reference_{ground_truth.get_hash()}',
            'Resnorm': 0.0,
            'Rsh (Ω)': ground_truth.rsh,
            'Ra (Ω)': ground_truth.ra,
            'Rb (Ω)': ground_truth.rb,
            'Ca (F)': ground_truth.ca,
            'Cb (F)': ground_truth.cb,
            'Ground Truth ID': ground_truth.id
        })
        
        # Generate all parameter combinations
        total_models = self.n_grid ** 5
        
        with tqdm(total=total_models, desc=f"Computing {ground_truth.id}", leave=False) as pbar:
            for i2 in range(self.n_grid):
                for i3 in range(self.n_grid):
                    for i4 in range(self.n_grid):
                        for i5 in range(self.n_grid):
                            for i6 in range(self.n_grid):
                                # Create model configuration
                                model_config = GroundTruthConfig(
                                    id=f"model_{i2+1:02d}_{i3+1:02d}_{i4+1:02d}_{i5+1:02d}_{i6+1:02d}",
                                    rsh=self.grids['rsh'][i2],
                                    ra=self.grids['ra'][i3],
                                    rb=self.grids['rb'][i4],
                                    ca=self.grids['ca'][i5],
                                    cb=self.grids['cb'][i6]
                                )
                                
                                # Compute impedance spectrum
                                z_model = self.eis_engine.compute_impedance_spectrum(model_config)
                                
                                # Compute resnorm
                                resnorm = self.eis_engine.compute_resnorm(z_reference, z_model)
                                
                                # Store result
                                results.append({
                                    'Model ID': model_config.id,
                                    'Resnorm': resnorm,
                                    'Rsh (Ω)': model_config.rsh,
                                    'Ra (Ω)': model_config.ra,
                                    'Rb (Ω)': model_config.rb,
                                    'Ca (F)': model_config.ca,
                                    'Cb (F)': model_config.cb,
                                    'Ground Truth ID': ground_truth.id
                                })
                                
                                pbar.update(1)
        
        return pd.DataFrame(results)
    
    def generate_complete_dataset(self, n_ground_truths: int = 100, 
                                   parallel: bool = True, n_workers: int = None):
        """
        Generate complete multi-ground-truth dataset
        
        Args:
            n_ground_truths: Number of ground truth configurations
            parallel: Use multiprocessing for parallel generation
            n_workers: Number of parallel workers (None = use all CPUs)
        """
        print("="*70)
        print("MULTI-GROUND-TRUTH DATASET GENERATION")
        print("="*70)
        
        # Generate ground truth configurations
        ground_truths = self.sampler.generate_combined_dataset(n_ground_truths)
        
        # Save ground truth metadata
        metadata_path = self.output_dir / "ground_truth_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump([gt.to_dict() for gt in ground_truths], f, indent=2)
        print(f"\nSaved ground truth metadata to: {metadata_path}")
        
        # Generate datasets
        print(f"\nGenerating {n_ground_truths} datasets ({self.n_grid**5:,} models each)...")
        print(f"Total models: {n_ground_truths * self.n_grid**5:,}")
        print(f"Estimated storage: {(n_ground_truths * 11 / 1024):.2f} GB\n")
        
        start_time = time.time()
        
        if parallel and n_workers != 1:
            # Parallel generation
            if n_workers is None:
                n_workers = mp.cpu_count()
            
            print(f"Using {n_workers} parallel workers...")
            
            with mp.Pool(n_workers) as pool:
                dataframes = list(tqdm(
                    pool.imap(self.generate_grid_for_ground_truth, ground_truths),
                    total=len(ground_truths),
                    desc="Overall Progress"
                ))
        else:
            # Sequential generation
            dataframes = []
            for gt in tqdm(ground_truths, desc="Overall Progress"):
                df = self.generate_grid_for_ground_truth(gt)
                dataframes.append(df)
        
        # Combine all dataframes
        print("\nCombining datasets...")
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Save combined dataset
        output_path = self.output_dir / f"combined_dataset_{n_ground_truths}gt.csv"
        combined_df.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"Total models generated: {len(combined_df):,}")
        print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
        print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"Models/second: {len(combined_df)/elapsed_time:.0f}")
        print(f"Output file: {output_path}")
        
        # Generate summary statistics
        self._generate_summary_report(combined_df, ground_truths, output_path.parent)
        
        return combined_df, ground_truths
    
    def _generate_summary_report(self, df: pd.DataFrame, ground_truths: List[GroundTruthConfig],
                                  output_dir: Path):
        """Generate summary statistics and visualizations"""
        report_path = output_dir / "dataset_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DATASET SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Ground Truths: {len(ground_truths)}\n")
            f.write(f"Total Models: {len(df):,}\n")
            f.write(f"Models per Ground Truth: {len(df) // len(ground_truths):,}\n\n")
            
            f.write("Parameter Ranges:\n")
            for param in ['Rsh (Ω)', 'Ra (Ω)', 'Rb (Ω)', 'Ca (F)', 'Cb (F)']:
                f.write(f"  {param}: {df[param].min():.2e} to {df[param].max():.2e}\n")
            
            f.write("\nResnorm Statistics:\n")
            f.write(f"  Min: {df['Resnorm'].min():.4f}\n")
            f.write(f"  25th percentile: {df['Resnorm'].quantile(0.25):.4f}\n")
            f.write(f"  Median: {df['Resnorm'].median():.4f}\n")
            f.write(f"  75th percentile: {df['Resnorm'].quantile(0.75):.4f}\n")
            f.write(f"  90th percentile: {df['Resnorm'].quantile(0.90):.4f}\n")
            f.write(f"  Max: {df['Resnorm'].max():.4f}\n")
        
        print(f"\nSummary report saved to: {report_path}")


# Main execution
if __name__ == "__main__":
    # Initialize generator
    generator = DatasetGenerator(
        output_dir="./eis_training_data",
        n_grid_points=12
    )
    
    # Generate dataset with 100 ground truths (recommended)
    # For testing, start with n_ground_truths=10
    dataset, ground_truths = generator.generate_complete_dataset(
        n_ground_truths=100,  # Adjust as needed
        parallel=True,
        n_workers=None  # Use all available CPUs
    )
    
    print("\n✓ Dataset generation complete!")
    print("Next step: Train the probabilistic prediction model using this dataset")

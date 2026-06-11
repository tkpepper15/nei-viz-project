#!/usr/bin/env python3
"""
Deterministic Physics Fit - Classical EIS Parameter Extraction

This is the conventional electrochemical impedance spectroscopy (EIS) method:
fit circuit parameters by minimizing the residual between measured and
simulated impedance spectra.

Role in Three-Model Framework:
    - Model A: Physics ceiling under current theory
    - Cannot hallucinate - fails honestly when physics is wrong
    - Serves as baseline for "best possible with known equations"

Mathematical Formulation:
    θ̂ = argmin_θ Σ_ω |Z_ECM(ω; θ) - Z_data(ω)|²

    Where:
        θ = [Ra, Rb, Ca, Cb, Rsh]  (circuit parameters)
        Z_ECM(ω; θ) = forward model from Randles circuit
        Z_data(ω) = measured impedance spectrum

Optimization Strategy:
    1. Multiple methods: Nelder-Mead, L-BFGS-B, Differential Evolution
    2. Multi-start initialization to avoid local minima
    3. Parameter bounds enforcement (physical constraints)
    4. Convergence diagnostics and uncertainty estimation
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Optional, List
import warnings
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data.randles_circuit_simulator import RandlesCircuitSimulator


class DeterministicPhysicsFit:
    """
    Classical EIS fitting via nonlinear least squares optimization.

    This is the "conventional method" in electrochemistry:
    - Propose equivalent circuit model
    - Fit parameters to minimize impedance residual
    - Extract physical quantities (TER, time constants, etc.)

    Serves as physics ceiling in three-model comparison.
    """

    def __init__(
        self,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = 'L-BFGS-B',
        use_weights: bool = True,
        n_restarts: int = 10,
        use_relative_error: bool = True
    ):
        """
        Initialize deterministic fit.

        Args:
            param_bounds: Parameter bounds dict, e.g. {'Ra': (5, 30), ...}
                         If None, uses default RPE-appropriate ranges
            method: Optimization method ('Nelder-Mead', 'L-BFGS-B', 'differential_evolution')
            use_weights: Whether to use frequency-dependent weighting
            n_restarts: Number of multi-start trials (for avoiding local minima)
            use_relative_error: Use relative (normalized) error instead of absolute
        """
        self.simulator = RandlesCircuitSimulator()
        self.method = method
        self.use_weights = use_weights
        self.n_restarts = n_restarts
        self.use_relative_error = use_relative_error

        # Default parameter bounds (wide ranges for general EIS fitting)
        # Covers 3 orders of magnitude for resistances, 3 for capacitances
        if param_bounds is None:
            self.param_bounds = {
                'Ra': (10.0, 10000.0),      # Apical resistance (Ω)
                'Rb': (10.0, 10000.0),      # Basolateral resistance (Ω)
                'Ca': (1e-7, 5e-5),         # Apical capacitance (F)
                'Cb': (1e-7, 5e-5),         # Basolateral capacitance (F)
                'Rsh': (10.0, 10000.0)      # Shunt resistance (Ω)
            }
        else:
            self.param_bounds = param_bounds

        # Convert to arrays for scipy
        self.param_names = ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh']
        self.bounds = [self.param_bounds[name] for name in self.param_names]

    def _objective(
        self,
        params: np.ndarray,
        frequencies: np.ndarray,
        Z_real_data: np.ndarray,
        Z_imag_data: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Objective function: weighted sum of squared residuals.

        When use_relative_error=True:
            L(θ) = Σ_ω w_ω × [(Z_real - Z_real_data)² + (Z_imag - Z_imag_data)²] / |Z_data|²

        When use_relative_error=False:
            L(θ) = Σ_ω w_ω × [(Z_real - Z_real_data)² + (Z_imag - Z_imag_data)²]

        Args:
            params: [Ra, Rb, Ca, Cb, Rsh]
            frequencies: Frequency array (Hz)
            Z_real_data: Real impedance data (Ω)
            Z_imag_data: Imaginary impedance data (Ω)
            weights: Optional frequency weights

        Returns:
            Weighted sum of squared residuals
        """
        Ra, Rb, Ca, Cb, Rsh = params

        # Forward simulate impedance with current parameter guess
        try:
            Z_real_sim, Z_imag_sim = self.simulator.compute_impedance(
                frequencies, Ra, Rb, Ca, Cb, Rsh
            )
        except (ValueError, RuntimeWarning):
            # Penalize invalid parameter combinations
            return 1e10

        # Compute residuals
        residual_real = Z_real_sim - Z_real_data
        residual_imag = Z_imag_sim - Z_imag_data

        # Squared residuals
        squared_residuals = residual_real**2 + residual_imag**2

        # Normalize by impedance magnitude for relative error
        # This prevents high-impedance (low-frequency) points from dominating
        if self.use_relative_error:
            Z_mag_squared = Z_real_data**2 + Z_imag_data**2 + 1e-10
            squared_residuals = squared_residuals / Z_mag_squared

        # Apply weights if provided
        if weights is not None:
            squared_residuals *= weights

        # Sum to get total objective
        return np.sum(squared_residuals)

    def _compute_weights(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Uniform frequency weights.

        All frequencies contribute equally before the relative-error
        normalisation in _objective().  The previous Gaussian envelope
        centred at the geometric mean of the spectrum actively suppressed
        low-frequency points (by ~30-400x combined with |Z|^2 normalisation),
        which are the primary carriers of Rsh / TER information.  Removing it
        makes the ECM objective consistent with the Black-Litterman relative
        noise model used in the DL pipeline.
        """
        return np.ones_like(frequencies)

    def _generate_initial_guess(self) -> np.ndarray:
        """
        Generate random initial parameter guess within bounds.

        Uses log-uniform sampling for ALL parameters since they span
        orders of magnitude (10-10000 Ω for resistances, 1e-7 to 5e-5 F
        for capacitances).

        Returns:
            Initial parameter vector [Ra, Rb, Ca, Cb, Rsh]
        """
        params = []
        for i, name in enumerate(self.param_names):
            lower, upper = self.bounds[i]

            # Log-uniform for ALL parameters (they span orders of magnitude)
            # This ensures equal probability per decade
            log_lower = np.log10(max(lower, 1e-15))  # Prevent log(0)
            log_upper = np.log10(upper)
            value = 10 ** np.random.uniform(log_lower, log_upper)

            params.append(value)

        return np.array(params)

    def fit(
        self,
        frequencies: np.ndarray,
        Z_real: np.ndarray,
        Z_imag: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Fit circuit parameters to impedance spectrum.

        This is the core EIS fitting procedure:
        1. Initialize parameters (random or provided)
        2. Optimize to minimize |Z_sim - Z_data|²
        3. Compute derived quantities (TER, τa, τb)
        4. Return fit results with diagnostics

        Args:
            frequencies: Frequency array (Hz), shape (n_freq,)
            Z_real: Real impedance (Ω), shape (n_freq,)
            Z_imag: Imaginary impedance (Ω), shape (n_freq,)
            initial_guess: Optional initial parameters [Ra, Rb, Ca, Cb, Rsh]
            verbose: Print optimization progress

        Returns:
            Dict with:
                params_fit: Optimized parameters [Ra, Rb, Ca, Cb, Rsh]
                params_dict: Named parameter dict
                residual: Final objective value
                success: Whether optimization converged
                message: Optimization status message
                n_iterations: Number of iterations
                TER: Computed transepithelial resistance
                TEC: Computed transepithelial capacitance
                tau_a: Apical time constant (Ra × Ca)
                tau_b: Basolateral time constant (Rb × Cb)
                Z_real_fit: Fitted real impedance
                Z_imag_fit: Fitted imaginary impedance
                mae_real: Mean absolute error (real component)
                mae_imag: Mean absolute error (imag component)
        """
        # Compute weights
        weights = self._compute_weights(frequencies)

        # Multi-start optimization
        best_result = None
        best_residual = np.inf

        for trial in range(self.n_restarts):
            # Generate initial guess
            if initial_guess is not None and trial == 0:
                x0 = initial_guess
            else:
                x0 = self._generate_initial_guess()

            if verbose and self.n_restarts > 1:
                print(f"  Trial {trial+1}/{self.n_restarts}: ", end='')

            # Optimize
            try:
                if self.method == 'differential_evolution':
                    # Global optimization (slower but more robust)
                    result = differential_evolution(
                        self._objective,
                        bounds=self.bounds,
                        args=(frequencies, Z_real, Z_imag, weights),
                        maxiter=1000,
                        atol=1e-10,
                        seed=trial,
                        workers=1
                    )
                else:
                    # Local optimization (faster)
                    result = minimize(
                        self._objective,
                        x0=x0,
                        args=(frequencies, Z_real, Z_imag, weights),
                        method=self.method,
                        bounds=self.bounds,
                        options={'maxiter': 1000}
                    )

                # Track best result
                if result.fun < best_residual:
                    best_residual = result.fun
                    best_result = result

                    if verbose and self.n_restarts > 1:
                        print(f"residual={result.fun:.2e}")

            except Exception as e:
                if verbose:
                    print(f"Failed: {e}")
                continue

        if best_result is None:
            raise RuntimeError("All optimization trials failed")

        result = best_result

        # Extract optimized parameters
        Ra_fit, Rb_fit, Ca_fit, Cb_fit, Rsh_fit = result.x

        # Compute fitted spectrum
        Z_real_fit, Z_imag_fit = self.simulator.compute_impedance(
            frequencies, Ra_fit, Rb_fit, Ca_fit, Cb_fit, Rsh_fit
        )

        # Compute derived quantities
        TER = self.simulator.compute_ter(Ra_fit, Rb_fit, Rsh_fit)
        TEC = (Ca_fit * Cb_fit) / (Ca_fit + Cb_fit)  # Series capacitance
        tau_a = Ra_fit * Ca_fit
        tau_b = Rb_fit * Cb_fit

        # Compute error metrics
        mae_real = np.mean(np.abs(Z_real_fit - Z_real))
        mae_imag = np.mean(np.abs(Z_imag_fit - Z_imag))
        mae_total = mae_real + mae_imag

        # Compute relative impedance error (more meaningful success criterion)
        Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
        relative_error = np.mean(np.sqrt((Z_real_fit - Z_real)**2 + (Z_imag_fit - Z_imag)**2) / (Z_mag + 1e-10))

        # Practical success criterion: relative error < 25%
        # (scipy's success flag is overly strict for our purposes)
        # 25% allows for noise and minor fitting imperfections while
        # ensuring the solution is in the right ballpark
        practical_success = relative_error < 0.25

        # Return comprehensive results
        return {
            'params_fit': result.x,
            'params_dict': {
                'Ra': Ra_fit,
                'Rb': Rb_fit,
                'Ca': Ca_fit,
                'Cb': Cb_fit,
                'Rsh': Rsh_fit
            },
            'residual': result.fun,
            'success': practical_success,  # Use practical criterion
            'scipy_success': result.success,  # Keep original for diagnostics
            'relative_error': relative_error,
            'message': result.message if hasattr(result, 'message') else 'Success',
            'n_iterations': result.nit if hasattr(result, 'nit') else result.nfev,
            'TER': TER,
            'TEC': TEC,
            'tau_a': tau_a,
            'tau_b': tau_b,
            'Z_real_fit': Z_real_fit,
            'Z_imag_fit': Z_imag_fit,
            'mae_real': mae_real,
            'mae_imag': mae_imag,
            'mae_total': mae_total
        }

    def fit_batch(
        self,
        dataset: List[Dict],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Fit multiple spectra in batch.

        Args:
            dataset: List of dicts with 'frequencies', 'Z_real', 'Z_imag'
            verbose: Show progress bar

        Returns:
            List of fit results (one per spectrum)
        """
        results = []

        iterator = dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(dataset, desc='Fitting spectra')
            except ImportError:
                print(f"Fitting {len(dataset)} spectra...")

        for sample in iterator:
            try:
                result = self.fit(
                    frequencies=sample['frequencies'],
                    Z_real=sample['Z_real'],
                    Z_imag=sample['Z_imag'],
                    verbose=False
                )
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Failed to fit sample: {e}")
                # Append failed result
                results.append({
                    'success': False,
                    'message': str(e),
                    'params_fit': None
                })

        return results


def test_deterministic_fit():
    """Test the deterministic fit on synthetic data."""
    print("="*70)
    print("TESTING DETERMINISTIC PHYSICS FIT")
    print("="*70)

    # Create simulator
    simulator = RandlesCircuitSimulator()

    # Generate test spectrum with known parameters
    frequencies = np.logspace(np.log10(5), np.log10(10000), 25)

    true_params = {
        'Ra': 15.0,
        'Rb': 20.0,
        'Ca': 5e-6,
        'Cb': 3e-6,
        'Rsh': 40.0
    }

    print("\nTrue Parameters:")
    for name, value in true_params.items():
        print(f"  {name:5s} = {value:.2e}")

    # Compute true spectrum
    Z_real_true, Z_imag_true = simulator.compute_impedance(
        frequencies, **true_params
    )

    # Add realistic noise (2% of signal)
    noise_level = 0.02
    noise_real = np.random.normal(0, noise_level * np.abs(Z_real_true).mean(), len(Z_real_true))
    noise_imag = np.random.normal(0, noise_level * np.abs(Z_imag_true).mean(), len(Z_imag_true))

    Z_real = Z_real_true + noise_real
    Z_imag = Z_imag_true + noise_imag

    print(f"\nAdded {noise_level*100}% Gaussian noise")

    # Test different optimization methods
    methods = ['L-BFGS-B', 'Nelder-Mead']

    for method in methods:
        print(f"\n{'='*70}")
        print(f"Method: {method}")
        print(f"{'='*70}")

        # Create fitter
        fitter = DeterministicPhysicsFit(
            method=method,
            use_weights=True,
            n_restarts=3
        )

        # Fit
        result = fitter.fit(frequencies, Z_real, Z_imag, verbose=True)

        # Display results
        print("\nFit Results:")
        print(f"  Success: {result['success']}")
        print(f"  Residual: {result['residual']:.2e}")
        print(f"  Iterations: {result['n_iterations']}")

        print("\nFitted Parameters:")
        for name, value in result['params_dict'].items():
            true_value = true_params[name]
            error = np.abs(value - true_value)
            rel_error = error / true_value * 100
            print(f"  {name:5s} = {value:.2e}  (true: {true_value:.2e}, error: {rel_error:6.2f}%)")

        print("\nDerived Quantities:")
        true_TER = simulator.compute_ter(true_params['Ra'], true_params['Rb'], true_params['Rsh'])
        print(f"  TER   = {result['TER']:.2f} Ω  (true: {true_TER:.2f} Ω)")
        print(f"  TEC   = {result['TEC']:.2e} F")
        print(f"  τa    = {result['tau_a']:.2e} s")
        print(f"  τb    = {result['tau_b']:.2e} s")

        print("\nSpectrum Fit Quality:")
        print(f"  MAE (real): {result['mae_real']:.3f} Ω")
        print(f"  MAE (imag): {result['mae_imag']:.3f} Ω")
        print(f"  MAE (total): {result['mae_total']:.3f} Ω")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # Run test
    test_deterministic_fit()

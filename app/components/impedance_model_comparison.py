import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, differential_evolution
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import itertools
import os
import datetime
import time
from pathlib import Path

# Function to load raw impedance data
def load_impedance_data(file_path):
    """
    Load impedance data from the raw data file.
    The file has multiple sections separated by dates.
    Returns a dictionary with dates as keys and dataframes as values.
    """
    data_dict = {}
    current_date = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    temp_data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a date
        if '/' in line and len(line.split('/')) == 3:
            # If we already have data, save it
            if current_date and temp_data:
                df = pd.DataFrame([x.split('\t') for x in temp_data[1:]], columns=temp_data[0].split('\t'))
                # Convert columns to proper types
                for col in df.columns:
                    if col != 'Index':  # Skip the index column
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                data_dict[current_date] = df
                
            # Set new current date and reset temp data
            current_date = line
            temp_data = []
        else:
            temp_data.append(line)
    
    # Process the last section
    if current_date and temp_data:
        df = pd.DataFrame([x.split('\t') for x in temp_data[1:]], columns=temp_data[0].split('\t'))
        # Convert columns to proper types
        for col in df.columns:
            if col != 'Index':  # Skip the index column
                df[col] = pd.to_numeric(df[col], errors='ignore')
        data_dict[current_date] = df
        
    return data_dict

# Default equivalent circuit model for impedance
def default_circuit_model(params, frequencies):
    """
    Calculate the impedance for a given equivalent circuit model.
    
    This model uses a simple Randles circuit with:
    - Rs (series resistance)
    - R1 || C1 (parallel resistance and capacitance)
    - R2 || C2 (second parallel resistance and capacitance)
    
    Args:
        params: List of circuit parameters [Rs, R1, C1, R2, C2]
        frequencies: Array of frequencies to calculate impedance at
        
    Returns:
        real_imag: Array of complex impedance values with real and imaginary parts flattened
    """
    Rs, R1, C1, R2, C2 = params
    
    # Angular frequency
    omega = 2 * np.pi * np.array(frequencies)
    
    # Initialize arrays for real and imaginary parts
    Z_real = np.zeros_like(omega)
    Z_imag = np.zeros_like(omega)
    
    # Calculate impedance components
    for i, w in enumerate(omega):
        # Impedance of first parallel RC circuit
        Z1_real = R1 / (1 + (w * R1 * C1)**2)
        Z1_imag = -w * R1**2 * C1 / (1 + (w * R1 * C1)**2)
        
        # Impedance of second parallel RC circuit
        Z2_real = R2 / (1 + (w * R2 * C2)**2)
        Z2_imag = -w * R2**2 * C2 / (1 + (w * R2 * C2)**2)
        
        # Total impedance (Rs + Z1 + Z2)
        Z_real[i] = Rs + Z1_real + Z2_real
        Z_imag[i] = Z1_imag + Z2_imag
    
    # Return flattened array [Z_real_1, Z_real_2, ..., Z_imag_1, Z_imag_2, ...]
    return np.concatenate((Z_real, Z_imag))

# Define the residual norm function
def resnorm(params, frequencies, Z_real_measured, Z_imag_measured, circuit_model_func=None):
    """
    Calculate the residual norm between measured and calculated impedance values.
    
    Args:
        params: Circuit model parameters
        frequencies: Frequencies at which measurements were taken
        Z_real_measured: Real part of measured impedance
        Z_imag_measured: Imaginary part of measured impedance
        circuit_model_func: Custom circuit model function (optional)
        
    Returns:
        residuals: Array of residuals between measured and calculated values
    """
    # Use default or custom circuit model function
    model_func = circuit_model_func if circuit_model_func else default_circuit_model
    
    # Calculate model impedance
    Z_calc = model_func(params, frequencies)
    
    # Split into real and imaginary parts
    n = len(frequencies)
    Z_real_calc = Z_calc[:n]
    Z_imag_calc = Z_calc[n:]
    
    # Calculate residuals
    residuals_real = Z_real_calc - Z_real_measured
    residuals_imag = Z_imag_calc - Z_imag_measured
    
    # Return combined residuals
    return np.concatenate((residuals_real, residuals_imag))

# Calculate the residual norm value (scalar)
def calculate_resnorm_value(params, frequencies, Z_real_measured, Z_imag_measured, circuit_model_func=None):
    """
    Calculate the scalar residual norm value (sum of squared residuals).
    """
    residuals = resnorm(params, frequencies, Z_real_measured, Z_imag_measured, circuit_model_func)
    return np.sum(residuals**2)

# Function to fit the circuit model to measured data
def fit_circuit_model(frequencies, Z_real_measured, Z_imag_measured, initial_guess=None, circuit_model_func=None):
    """
    Fit the circuit model to the measured impedance data.
    
    Args:
        frequencies: Frequencies at which measurements were taken
        Z_real_measured: Real part of measured impedance
        Z_imag_measured: Imaginary part of measured impedance
        initial_guess: Initial parameter values for the optimization
        circuit_model_func: Custom circuit model function (optional)
        
    Returns:
        params_opt: Optimized circuit parameters
        residual_norm: Norm of the residuals between measured and calculated values
    """
    model_func = circuit_model_func if circuit_model_func else default_circuit_model
    
    if initial_guess is None:
        # Provide reasonable initial guesses based on data
        Rs_guess = np.min(Z_real_measured)
        R1_guess = np.mean(Z_real_measured) - Rs_guess
        R2_guess = R1_guess
        
        # Estimate capacitance from the peak frequency
        max_imag_idx = np.argmax(abs(Z_imag_measured))
        if max_imag_idx > 0 and max_imag_idx < len(frequencies):
            freq_peak = frequencies[max_imag_idx]
            C1_guess = 1 / (2 * np.pi * freq_peak * R1_guess)
            C2_guess = C1_guess / 2
        else:
            C1_guess = 1e-6
            C2_guess = 5e-7
        
        initial_guess = [Rs_guess, R1_guess, C1_guess, R2_guess, C2_guess]
    
    # Set bounds for parameters (all positive)
    bounds_lower = [0] * len(initial_guess)
    bounds_upper = [np.inf] * len(initial_guess)
    
    # Perform optimization
    result = least_squares(
        resnorm, 
        initial_guess, 
        args=(frequencies, Z_real_measured, Z_imag_measured, model_func),
        bounds=(bounds_lower, bounds_upper),
        method='trf',
        verbose=0
    )
    
    # Calculate the final residual norm
    residual_norm = np.sum(result.fun**2)
    
    return result.x, residual_norm

# Plot comparison between measured and calculated impedance
def plot_comparison(frequencies, Z_real_measured, Z_imag_measured, params, date_label, param_names=None, circuit_model_func=None, output_dir=None, save_prefix="impedance_plot"):
    """
    Plot comparison between measured and model data.
    
    Args:
        frequencies: Frequencies at which measurements were taken
        Z_real_measured: Real part of measured impedance
        Z_imag_measured: Imaginary part of measured impedance
        params: Circuit model parameters
        date_label: Label for the date of the measurements
        param_names: List of parameter names (optional)
        circuit_model_func: Custom circuit model function (optional)
        output_dir: Directory to save the plot (optional)
        save_prefix: Prefix for saved file (optional)
    """
    model_func = circuit_model_func if circuit_model_func else default_circuit_model
    
    # Calculate model impedance
    Z_calc = model_func(params, frequencies)
    n = len(frequencies)
    Z_real_calc = Z_calc[:n]
    Z_imag_calc = Z_calc[n:]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot real part
    ax1.scatter(frequencies, Z_real_measured, label='Measured', color='blue')
    ax1.plot(frequencies, Z_real_calc, label='Model', color='red', linestyle='-')
    ax1.set_xscale('log')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Real Impedance Z\' (Ω)')
    ax1.set_title(f'Real Impedance vs Frequency - {date_label}')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    
    # Plot imaginary part
    ax2.scatter(frequencies, -Z_imag_measured, label='Measured', color='blue')
    ax2.plot(frequencies, -Z_imag_calc, label='Model', color='red', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('-Imaginary Impedance -Z\'\' (Ω)')
    ax2.set_title(f'Imaginary Impedance vs Frequency - {date_label}')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    
    # Add circuit parameters and residual norm to the plot
    residual_norm = calculate_resnorm_value(params, frequencies, Z_real_measured, Z_imag_measured, model_func)
    
    # Calculate normalized resnorm (per data point)
    n_points = len(frequencies) * 2  # Real and imaginary parts
    normalized_resnorm = residual_norm / n_points
    
    # Default parameter names if not provided
    if param_names is None:
        param_names = [f"Param{i+1}" for i in range(len(params))]
    
    # Create parameter text
    param_text = "Circuit Parameters:\n"
    for name, value in zip(param_names, params):
        if abs(value) < 0.01 or abs(value) > 1000:
            param_text += f"{name} = {value:.3e}\n"
        else:
            param_text += f"{name} = {value:.3f}\n"
    
    param_text += f"Total Resnorm = {residual_norm:.2e}\n"
    param_text += f"Normalized Resnorm = {normalized_resnorm:.2e}"
    
    fig.text(0.02, 0.5, param_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)
    
    # Save figure if output directory is provided
    if output_dir is not None:
        sanitized_date = date_label.replace('/', '-').replace(' ', '_')
        filename = f"{save_prefix}_{sanitized_date}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filepath}")
    
    return fig, residual_norm, normalized_resnorm

# Plot Nyquist plot comparison
def plot_nyquist_comparison(Z_real_measured, Z_imag_measured, params, frequencies, date_label, param_names=None, circuit_model_func=None):
    """
    Plot Nyquist comparison between measured and model data.
    
    Args:
        Z_real_measured: Real part of measured impedance
        Z_imag_measured: Imaginary part of measured impedance
        params: Circuit model parameters
        frequencies: Frequencies at which measurements were taken
        date_label: Label for the date of the measurements
        param_names: List of parameter names (optional)
        circuit_model_func: Custom circuit model function (optional)
    """
    model_func = circuit_model_func if circuit_model_func else default_circuit_model
    
    # Calculate model impedance
    Z_calc = model_func(params, frequencies)
    n = len(frequencies)
    Z_real_calc = Z_calc[:n]
    Z_imag_calc = Z_calc[n:]
    
    # Create a figure
    plt.figure(figsize=(8, 8))
    
    # Plot Nyquist plot
    plt.scatter(Z_real_measured, -Z_imag_measured, label='Measured', color='blue')
    plt.plot(Z_real_calc, -Z_imag_calc, label='Model', color='red', linestyle='-')
    
    # Add frequency markers
    for i, freq in enumerate(frequencies):
        if i % 3 == 0:  # Add markers for every 3rd frequency to avoid clutter
            plt.annotate(f"{freq} Hz", 
                         (Z_real_measured[i], -Z_imag_measured[i]),
                         textcoords="offset points", 
                         xytext=(2, 2),
                         fontsize=8)
    
    plt.xlabel('Real Impedance Z\' (Ω)')
    plt.ylabel('-Imaginary Impedance -Z\'\' (Ω)')
    plt.title(f'Nyquist Plot - {date_label}')
    plt.grid(True, ls="--")
    plt.legend()
    plt.axis('equal')  # Equal aspect ratio
    
    # Add circuit parameters and residual norm to the plot
    residual_norm = calculate_resnorm_value(params, frequencies, Z_real_measured, Z_imag_measured, model_func)
    
    # Default parameter names if not provided
    if param_names is None:
        param_names = [f"Param{i+1}" for i in range(len(params))]
    
    # Create parameter text
    param_text = "Circuit Parameters:\n"
    for name, value in zip(param_names, params):
        if abs(value) < 0.01 or abs(value) > 1000:
            param_text += f"{name} = {value:.3e}\n"
        else:
            param_text += f"{name} = {value:.3f}\n"
    
    param_text += f"Resnorm = {residual_norm:.2e}"
    
    plt.text(0.02, 0.5, param_text, fontsize=10, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt.gcf()

# Example of a custom circuit model - users can define their own
def custom_circuit_model(params, frequencies):
    """
    Example of a custom circuit model that users can modify or replace.
    This is a different circuit configuration than the default model.
    
    For this example, let's implement a model with:
    - Rs (series resistance)
    - (R1 || C1) -- (R2 || CPE) (two parallel elements in series)
    
    Args:
        params: List of circuit parameters [Rs, R1, C1, R2, Q, alpha]
        frequencies: Array of frequencies to calculate impedance at
        
    Returns:
        real_imag: Array of complex impedance values with real and imaginary parts flattened
    """
    Rs, R1, C1, R2, Q, alpha = params
    
    # Angular frequency
    omega = 2 * np.pi * np.array(frequencies)
    
    # Initialize arrays for real and imaginary parts
    Z_real = np.zeros_like(omega)
    Z_imag = np.zeros_like(omega)
    
    # Calculate impedance components
    for i, w in enumerate(omega):
        # Impedance of first parallel RC circuit
        Z1_real = R1 / (1 + (w * R1 * C1)**2)
        Z1_imag = -w * R1**2 * C1 / (1 + (w * R1 * C1)**2)
        
        # Impedance of constant phase element (CPE)
        Z_cpe_mag = 1 / (Q * w**alpha)
        Z_cpe_phase = -np.pi * alpha / 2
        Z_cpe_real = Z_cpe_mag * np.cos(Z_cpe_phase)
        Z_cpe_imag = Z_cpe_mag * np.sin(Z_cpe_phase)
        
        # Parallel combination of R2 and CPE
        denom = (1/R2 + Z_cpe_real/(Z_cpe_real**2 + Z_cpe_imag**2))**2 + (Z_cpe_imag/(Z_cpe_real**2 + Z_cpe_imag**2))**2
        Z2_real = (1/R2) / denom
        Z2_imag = (-Z_cpe_imag/(Z_cpe_real**2 + Z_cpe_imag**2)) / denom
        
        # Total impedance (Rs + Z1 + Z2)
        Z_real[i] = Rs + Z1_real + Z2_real
        Z_imag[i] = Z1_imag + Z2_imag
    
    # Return flattened array [Z_real_1, Z_real_2, ..., Z_imag_1, Z_imag_2, ...]
    return np.concatenate((Z_real, Z_imag))

def funRCRCpR(p, w):
    """
    Python implementation of funRCRCpR MATLAB function
    
    Args:
        p: array of log10 of circuit parameters [Rsol, Ra, Ca, Rb, Cb, Rp]
        w: angular frequency (2*pi*f)
    
    Returns:
        Complex impedance array
    """
    # Convert parameters back from log scale
    p = 10**(np.array(p))
    
    Rsol = p[0]
    Ra = p[1]
    Ca = p[2]
    Rb = p[3]
    Cb = p[4]
    Rp = p[5]
    
    # R + RCRC | R - circuit equations
    Za = Ra / (1 + 1j * w * Ra * Ca)
    Zb = Rb / (1 + 1j * w * Rb * Cb)
    Zab = Za + Zb
    Zabs = 1 / (1/Zab + 1/Rp)
    Z = Rsol + Zabs
    
    return Z

def funRCRCpR_penalty(p, w, settings=None):
    """
    Python implementation of funRCRCpR_penalty MATLAB function
    
    Args:
        p: array of log10 of circuit parameters [Rsol, Ra, Ca, Rb, Cb, Rp]
        w: angular frequency (2*pi*f)
        settings: dictionary with penalty settings
    
    Returns:
        Array with [real(Z), imag(Z), penalty]
    """
    # Convert parameters back from log scale
    p = 10**(np.array(p))
    
    Rsol = p[0]
    Ra = p[1]
    Ca = p[2]
    Rb = p[3]
    Cb = p[4]
    Rp = p[5]
    
    # Default settings if none provided
    if settings is None:
        settings = {
            'penalty_method': 'none',
            'lambda_penalty': 0.1
        }
    
    # R + RCRC | R - circuit equations
    Za = Ra / (1 + 1j * w * Ra * Ca)
    Zb = Rb / (1 + 1j * w * Rb * Cb)
    Zab = Za + Zb
    Zabs = 1 / (1/Zab + 1/Rp)
    Z = Rsol + Zabs
    
    # Calculate penalty based on method
    penalty_method = settings.get('penalty_method', 'none').lower()
    total_error = 0
    
    if penalty_method == 'zr_imag_hf':
        target_zr_imag_hf = settings.get('target_zr_imag_hf', 0)
        total_error = impedance_ratio_imaginary_hf_penalty(Ca, Cb, target_zr_imag_hf)
    elif penalty_method == 'jrr':
        target_jrr = settings.get('target_jrr', 0)
        use_log10_ratio = settings.get('use_log10_ratio', False)
        total_error = jrr_penalty(Ra, Rb, Rp, target_jrr, use_log10_ratio)
    
    # Calculate final penalty
    lambda_penalty = settings.get('lambda_penalty', 0.1)
    penalty = lambda_penalty * total_error
    penalty_values = np.full(len(w), penalty/len(w))
    
    return np.column_stack([np.real(Z), np.imag(Z), penalty_values])

def impedance_ratio_imaginary_hf_penalty(Ca, Cb, target_zr_imag_hf):
    """Calculate penalty based on high frequency impedance ratio"""
    if Ca == Cb:
        x = Ca
    else:
        x = Ca * Cb / (Ca - Cb)
    dx = x - target_zr_imag_hf
    return dx**2

def jrr_penalty(Ra, Rb, Rp, target_jrr, use_log10_ratio):
    """Calculate penalty based on junction resistance ratio"""
    if use_log10_ratio:
        dx = np.log10((Ra + Rb) / Rp) - np.log10(target_jrr)
    else:
        dx = (Ra + Rb) / Rp - target_jrr
    return dx**2

def create_output_directory():
    """
    Create a timestamped output directory within the analysis_output folder.
    Returns the path to the created directory.
    """
    # Create analysis_output folder if it doesn't exist
    base_output_dir = Path('analysis_output')
    base_output_dir.mkdir(exist_ok=True)
    
    # Create a timestamped subfolder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir

def calculate_resnorm(p, w, ground_truth_data):
    """
    Calculate residual norm between model and ground truth data
    
    Args:
        p: array of log10 of circuit parameters [Rsol, Ra, Ca, Rb, Cb, Rp]
        w: angular frequency array
        ground_truth_data: array of [Z', -Z''] values
        
    Returns:
        resnorm: sum of squared differences
    """
    # Generate model impedance
    Z_model = funRCRCpR(p, w)
    
    # Extract model values
    model_real = np.real(Z_model)
    model_imag = np.imag(Z_model)
    
    # Extract ground truth values
    z_real_truth = ground_truth_data[:, 0]
    z_imag_truth = ground_truth_data[:, 1]  # This is already -Z''
    
    # Calculate residuals
    residuals_real = model_real - z_real_truth
    residuals_imag = -model_imag - z_imag_truth  # Need negative since ground truth is -Z''
    
    # Calculate squared residuals
    squared_residuals = residuals_real**2 + residuals_imag**2
    resnorm = np.sum(squared_residuals)
    
    return resnorm

def get_derived_params(p):
    """Calculate derived parameters from circuit parameters"""
    # Convert log10 parameters to actual values
    p = 10**(np.array(p))
    Rsol, Ra, Ca, Rb, Cb, Rp = p
    
    # Calculate derived parameters
    Rt = (Ra + Rb) * Rp / (Ra + Rb + Rp)
    JRR = (Ra + Rb) / Rp
    Ct = 1 / (1/Ca + 1/Cb)
    
    return {
        'Rsol': Rsol,
        'Ra': Ra,
        'Ca': Ca,
        'Rb': Rb,
        'Cb': Cb,
        'Rp': Rp,
        'Rt': Rt,
        'JRR': JRR,
        'Ct': Ct,
        'Ra/Rb': Ra/Rb,
        'Ca/Cb': Ca/Cb
    }

def generate_parameter_sets(n_sets=1000, param_ranges=None):
    """Generate n_sets of random parameter sets within specified ranges"""
    if param_ranges is None:
        # Default log10 ranges for [Rsol, Ra, Ca, Rb, Cb, Rp]
        param_ranges = [
            (0.5, 2.0),    # Rsol: 3-100 Ω
            (0.5, 2.5),    # Ra: 3-300 Ω
            (-7.0, -5.0),  # Ca: 0.1-10 μF
            (0.5, 2.5),    # Rb: 3-300 Ω
            (-7.0, -5.0),  # Cb: 0.1-10 μF
            (1.0, 3.0)     # Rp: 10-1000 Ω
        ]
    
    # Generate random parameter sets
    param_sets = []
    for _ in range(n_sets):
        params = [np.random.uniform(low, high) for low, high in param_ranges]
        param_sets.append(params)
    
    return np.array(param_sets)

def generate_grid_parameter_sets(n_points_per_dim=5, param_ranges=None):
    """Generate parameter sets on a grid within specified ranges"""
    if param_ranges is None:
        # Default log10 ranges for [Rsol, Ra, Ca, Rb, Cb, Rp]
        param_ranges = [
            (0.5, 2.0),    # Rsol: 3-100 Ω
            (0.5, 2.5),    # Ra: 3-300 Ω
            (-7.0, -5.0),  # Ca: 0.1-10 μF
            (0.5, 2.5),    # Rb: 3-300 Ω
            (-7.0, -5.0),  # Cb: 0.1-10 μF
            (1.0, 3.0)     # Rp: 10-1000 Ω
        ]
    
    # Create grid points for each parameter
    param_grids = []
    for low, high in param_ranges:
        param_grids.append(np.linspace(low, high, n_points_per_dim))
    
    # Generate all combinations
    param_sets = list(itertools.product(*param_grids))
    
    return np.array(param_sets)

def optimize_parameters(w, ground_truth_data, initial_params, method='TRF'):
    """Optimize parameters to fit the ground truth data"""
    # Define objective function
    def objective(p):
        Z_model = funRCRCpR(p, w)
        
        # Extract model values
        model_real = np.real(Z_model)
        model_imag = np.imag(Z_model)
        
        # Extract ground truth values
        z_real_truth = ground_truth_data[:, 0]
        z_imag_truth = ground_truth_data[:, 1]
        
        # Calculate residuals
        residuals_real = model_real - z_real_truth
        residuals_imag = -model_imag - z_imag_truth
        
        # Return flattened residuals
        return np.concatenate([residuals_real, residuals_imag])
    
    # Parameter bounds (log10 scale)
    bounds = ([0.5, 0.5, -7.0, 0.5, -7.0, 1.0],
              [2.0, 2.5, -5.0, 2.5, -5.0, 3.0])
    
    if method.upper() == 'TRF':
        # Trust Region Reflective algorithm
        result = least_squares(objective, initial_params, method='trf', bounds=bounds)
        p_best = result.x
        resnorm = np.sum(result.fun**2)
    else:
        # Differential Evolution (global optimization)
        result = differential_evolution(
            lambda p: np.sum(objective(p)**2), 
            bounds=list(zip(bounds[0], bounds[1])),
            popsize=20,
            maxiter=1000
        )
        p_best = result.x
        resnorm = result.fun
    
    return p_best, resnorm

def analyze_parameter_space(w, ground_truth_data, n_sets=1000):
    """Analyze parameter space by generating random parameter sets and calculating resnorm"""
    # Generate random parameter sets
    param_sets = generate_parameter_sets(n_sets)
    
    # Calculate resnorm for each parameter set
    results = []
    for i, params in enumerate(param_sets):
        resnorm = calculate_resnorm(params, w, ground_truth_data)
        derived_params = get_derived_params(params)
        results.append({
            'set_id': i,
            'resnorm': resnorm,
            **derived_params
        })
        
        # Print progress
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{n_sets} parameter sets")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Sort by resnorm (best fits first)
    results_df = results_df.sort_values('resnorm')
    
    return results_df

def find_multiple_solutions(w, ground_truth_data, n_attempts=20):
    """Find multiple local minima by starting optimization from different initial points"""
    # Generate diverse initial parameter sets
    param_sets = generate_parameter_sets(n_attempts)
    
    # Optimize from each starting point
    solutions = []
    for i, params in enumerate(param_sets):
        p_best, resnorm = optimize_parameters(w, ground_truth_data, params)
        derived_params = get_derived_params(p_best)
        solutions.append({
            'solution_id': i,
            'initial_params': params,
            'optimized_params': p_best,
            'resnorm': resnorm,
            **derived_params
        })
        print(f"Solution {i+1}/{n_attempts}: Resnorm = {resnorm:.6f}")
    
    # Convert to DataFrame
    solutions_df = pd.DataFrame(solutions)
    
    # Sort by resnorm (best fits first)
    solutions_df = solutions_df.sort_values('resnorm')
    
    return solutions_df

def plot_parameter_relationships(results_df, max_resnorm=None):
    """Plot relationships between parameters and resnorm"""
    # Filter results if max_resnorm is specified
    if max_resnorm is not None:
        df = results_df[results_df['resnorm'] <= max_resnorm].copy()
    else:
        df = results_df.copy()
    
    # Create color map based on resnorm
    norm = plt.Normalize(df['resnorm'].min(), df['resnorm'].max())
    
    # Create plots for key parameter relationships
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Ra/Rb vs JRR colored by resnorm
    ax1 = fig.add_subplot(221)
    scatter1 = ax1.scatter(df['Ra/Rb'], df['JRR'], c=df['resnorm'], cmap='viridis', norm=norm, alpha=0.7)
    ax1.set_xlabel('Ra/Rb')
    ax1.set_ylabel('JRR')
    ax1.set_title('Ra/Rb vs JRR')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    fig.colorbar(scatter1, ax=ax1, label='Resnorm')
    
    # 2. Ca/Cb vs Ct colored by resnorm
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(df['Ca/Cb'], df['Ct'], c=df['resnorm'], cmap='viridis', norm=norm, alpha=0.7)
    ax2.set_xlabel('Ca/Cb')
    ax2.set_ylabel('Ct')
    ax2.set_title('Ca/Cb vs Ct')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    fig.colorbar(scatter2, ax=ax2, label='Resnorm')
    
    # 3. 3D plot of Ra, Rb, and Rp
    ax3 = fig.add_subplot(223, projection='3d')
    scatter3 = ax3.scatter(df['Ra'], df['Rb'], df['Rp'], c=df['resnorm'], cmap='viridis', norm=norm, alpha=0.7)
    ax3.set_xlabel('Ra')
    ax3.set_ylabel('Rb')
    ax3.set_zlabel('Rp')
    ax3.set_title('Ra, Rb, Rp')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_zscale('log')
    fig.colorbar(scatter3, ax=ax3, label='Resnorm')
    
    # 4. 3D plot of Ca, Cb, and Rt
    ax4 = fig.add_subplot(224, projection='3d')
    scatter4 = ax4.scatter(df['Ca'], df['Cb'], df['Rt'], c=df['resnorm'], cmap='viridis', norm=norm, alpha=0.7)
    ax4.set_xlabel('Ca')
    ax4.set_ylabel('Cb')
    ax4.set_zlabel('Rt')
    ax4.set_title('Ca, Cb, Rt')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_zscale('log')
    fig.colorbar(scatter4, ax=ax4, label='Resnorm')
    
    plt.tight_layout()
    plt.show()

def compare_best_solutions(solutions_df, w, ground_truth_data, n_best=5):
    """Compare and plot the best solutions"""
    # Get n_best solutions
    best_solutions = solutions_df.head(n_best)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot ground truth data
    frequencies = w / (2 * np.pi)
    ax1.scatter(frequencies, ground_truth_data[:, 0], color='black', s=100, label='Ground Truth', marker='x')
    ax2.scatter(frequencies, ground_truth_data[:, 1], color='black', s=100, label='Ground Truth', marker='x')
    
    # Plot each solution
    colors = plt.cm.tab10.colors
    for i, (_, row) in enumerate(best_solutions.iterrows()):
        # Get parameters
        params = row['optimized_params']
        
        # Generate model output
        Z_model = funRCRCpR(params, w)
        z_real = np.real(Z_model)
        z_imag = -np.imag(Z_model)  # Negative imaginary part for comparison
        
        # Plot real part
        ax1.scatter(frequencies, z_real, color=colors[i % len(colors)], 
                   label=f"Solution {i+1} (Resnorm: {row['resnorm']:.2f})")
        
        # Plot imaginary part
        ax2.scatter(frequencies, z_imag, color=colors[i % len(colors)])
    
    # Set labels and titles
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Z\' (Ω)')
    ax1.set_title('Real Impedance')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('-Z\" (Ω)')
    ax2.set_title('Imaginary Impedance')
    ax2.set_xscale('log')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print parameter comparison table
    print("\nParameter Comparison for Best Solutions:")
    comparison_rows = []
    for i, (_, row) in enumerate(best_solutions.iterrows()):
        p_actual = 10**(row['optimized_params'])
        comparison_rows.append({
            'Solution': i+1,
            'Resnorm': row['resnorm'],
            'Rsol': p_actual[0],
            'Ra': p_actual[1],
            'Ca': p_actual[2],
            'Rb': p_actual[3],
            'Cb': p_actual[4],
            'Rp': p_actual[5],
            'Rt': row['Rt'],
            'JRR': row['JRR'],
            'Ct': row['Ct'],
            'Ra/Rb': row['Ra/Rb'],
            'Ca/Cb': row['Ca/Cb']
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def main():
    """Main function to run the parameter space exploration"""
    # Define ground truth data from the provided sample
    ground_truth = np.array([
        [18.3282302368751, 4.61153260619826]  # Z', -Z''
    ])
    
    # Frequency in Hz
    frequency = 10000
    # Angular frequency
    w = np.array([2 * np.pi * frequency])
    
    print("Starting parameter space exploration...")
    
    # Strategy 1: Random sampling of parameter space
    print("\n=== Strategy 1: Random Sampling of Parameter Space ===")
    random_results = analyze_parameter_space(w, ground_truth, n_sets=1000)
    print("\nTop 5 random parameter sets:")
    print(random_results.head(5).to_string())
    
    # Strategy 2: Find multiple local minima
    print("\n=== Strategy 2: Finding Multiple Local Minima ===")
    solutions = find_multiple_solutions(w, ground_truth, n_attempts=20)
    
    # Compare and plot best solutions
    print("\n=== Best Solutions Comparison ===")
    comparison = compare_best_solutions(solutions, w, ground_truth, n_best=5)
    
    # Plot parameter relationships
    print("\n=== Parameter Relationship Analysis ===")
    plot_parameter_relationships(random_results, max_resnorm=50)
    
    # Calculate key relationships in the best solutions
    print("\nKey Relationships in Best Solutions:")
    for i, (_, row) in enumerate(solutions.head(5).iterrows()):
        print(f"Solution {i+1} (Resnorm: {row['resnorm']:.2f}):")
        print(f"  Ra*Ca / (Rb*Cb) = {row['Ra']*row['Ca']/(row['Rb']*row['Cb']):.6f}")
        print(f"  (Ra+Rb)/Rp = {row['JRR']:.6f}")
        print(f"  Ra/Rb = {row['Ra/Rb']:.6f}")
        print(f"  Ca/Cb = {row['Ca/Cb']:.6f}")
    
    # Additional analysis for parameter ambiguity
    print("\n=== Parameter Ambiguity Analysis ===")
    # Filter for solutions with similar resnorm
    threshold = 1.5 * solutions.iloc[0]['resnorm']
    similar_solutions = solutions[solutions['resnorm'] <= threshold]
    
    # Calculate coefficient of variation for each parameter
    param_names = ['Rsol', 'Ra', 'Ca', 'Rb', 'Cb', 'Rp', 'Rt', 'JRR', 'Ct']
    cv_values = {}
    for param in param_names:
        mean = similar_solutions[param].mean()
        std = similar_solutions[param].std()
        cv = std / mean
        cv_values[param] = cv
    
    # Sort by coefficient of variation
    cv_sorted = sorted(cv_values.items(), key=lambda x: x[1])
    print("Parameter Stability (from most to least stable):")
    for param, cv in cv_sorted:
        print(f"  {param}: CV = {cv:.6f}")

if __name__ == "__main__":
    main() 
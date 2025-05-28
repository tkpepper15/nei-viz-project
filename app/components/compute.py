from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Optional, Tuple
import cmath
import uvicorn
from scipy.stats import norm
from dataclasses import dataclass
import itertools
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CircuitParameters(BaseModel):
    Rs: float      # Tight junction resistance (Ω)
    Ra: float      # Apical membrane resistance (Ω)
    Ca: float      # Apical membrane capacitance (F)
    Rb: float      # Basal membrane resistance (Ω)
    Cb: float      # Basal membrane capacitance (F)
    frequency_range: List[float]  # Frequency range for impedance calculations (Hz)

class ImpedancePoint(BaseModel):
    real: float
    imaginary: float
    frequency: float
    magnitude: float
    phase: float

# Ground truth dataset based on reference measurements
GROUND_TRUTH_DATASET = [
    {
        "real": 18.3282302368751,
        "imaginary": -4.61153260619826,
        "frequency": 10000,
        "magnitude": np.sqrt(18.3282302368751**2 + 4.61153260619826**2),
        "phase": np.arctan2(-4.61153260619826, 18.3282302368751) * (180 / np.pi)
    },
    {
        "real": 19.0140411028486,
        "imaginary": -7.52984606443549,
        "frequency": 5000,
        "magnitude": np.sqrt(19.0140411028486**2 + 7.52984606443549**2),
        "phase": np.arctan2(-7.52984606443549, 19.0140411028486) * (180 / np.pi)
    },
    {
        "real": 20.7560176016118,
        "imaginary": -16.8293998859934,
        "frequency": 2025,
        "magnitude": np.sqrt(20.7560176016118**2 + 16.8293998859934**2),
        "phase": np.arctan2(-16.8293998859934, 20.7560176016118) * (180 / np.pi)
    },
    {
        "real": 22.3597923401363,
        "imaginary": -21.3644829082237,
        "frequency": 1575,
        "magnitude": np.sqrt(22.3597923401363**2 + 21.3644829082237**2),
        "phase": np.arctan2(-21.3644829082237, 22.3597923401363) * (180 / np.pi)
    },
    {
        "real": 25.4208517108672,
        "imaginary": -28.5723160076139,
        "frequency": 1125,
        "magnitude": np.sqrt(25.4208517108672**2 + 28.5723160076139**2),
        "phase": np.arctan2(-28.5723160076139, 25.4208517108672) * (180 / np.pi)
    }
]

@dataclass
class ParameterSpace:
    ra: Tuple[float, float] = (100, 10000)        # Apical resistance (Ω)
    ca: Tuple[float, float] = (1e-6, 1e-3)        # Apical capacitance (F)
    rb: Tuple[float, float] = (100, 10000)        # Basal resistance (Ω)
    cb: Tuple[float, float] = (1e-6, 1e-3)        # Basal capacitance (F)
    noise_level: float = 0.05  # 5% measurement noise by default

class RegressionMeshParameters(BaseModel):
    reference_cell: CircuitParameters
    mesh_resolution: int = 10  # Points per dimension
    noise_level: float = 0.05  # 5% measurement noise
    top_percentage: float = 10.0  # Keep top 10% of fits
    use_ground_truth: bool = False  # Whether to use ground truth dataset

class ResnormParameters(BaseModel):
    test_data: List[ImpedancePoint]
    use_ground_truth: bool = True  # By default, use ground truth dataset

class MeshPoint(BaseModel):
    parameters: CircuitParameters
    resnorm: float
    alpha: float = 0.0  # Default alpha value

def calculate_membrane_impedance(R: float, C: float, omega: float) -> complex:
    """Calculate the impedance of a single membrane (apical or basal)
    Z(ω) = R + 1/(jωC) where:
    - R is the membrane resistance (Ω)
    - C is the membrane capacitance (F)
    - ω is the angular frequency (rad/s)
    """
    capacitive_reactance = 1 / (omega * C)
    return R - 1j * capacitive_reactance

def calculate_equivalent_impedance(params: CircuitParameters, omega: float) -> complex:
    """Calculate the equivalent impedance of the epithelial model
    Z_eq(ω) = (Rs * (Za(ω) + Zb(ω))) / (Rs + Za(ω) + Zb(ω))
    where:
    - Rs is the tight junction resistance (Ω)
    - Za is the apical membrane impedance (Ω)
    - Zb is the basal membrane impedance (Ω)
    - ω is the angular frequency (rad/s)
    """
    # Calculate individual membrane impedances
    Za = calculate_membrane_impedance(params.Ra, params.Ca, omega)
    Zb = calculate_membrane_impedance(params.Rb, params.Cb, omega)

    # Calculate parallel combination
    Zparallel = (params.Rs * (Za + Zb)) / (params.Rs + Za + Zb)

    # Return the equivalent impedance
    return Zparallel

def calculate_impedance_spectrum(params: CircuitParameters) -> List[ImpedancePoint]:
    """Calculate the complete impedance spectrum for a given frequency range"""
    results = []
    for f in params.frequency_range:
        omega = 2 * np.pi * f
        Z = calculate_equivalent_impedance(params, omega)
        
        results.append(ImpedancePoint(
            real=float(Z.real),
            imaginary=float(Z.imag),
            frequency=f,
            magnitude=float(abs(Z)),
            phase=float(cmath.phase(Z) * 180 / np.pi)
        ))
    return results

def calculate_TER(params: CircuitParameters) -> float:
    """Calculate TER (Transepithelial Resistance)
    TER = Rs * (Ra + Rb) / (Rs + Ra + Rb)
    where:
    - Rs is the tight junction resistance (Ω)
    - Ra is the apical membrane resistance (Ω)
    - Rb is the basal membrane resistance (Ω)
    """
    numerator = params.Rs * (params.Ra + params.Rb)
    denominator = params.Rs + params.Ra + params.Rb
    return numerator / denominator

def calculate_TEC(impedance_data: List[ImpedancePoint]) -> float:
    """Calculate TEC (Transepithelial Capacitance)
    1/TEC = (2/π) ∫₀^∞ Re(Z_eq(ω)) dω
    where:
    - Z_eq is the equivalent impedance as a function of angular frequency
    """
    integral = 0
    for i in range(len(impedance_data) - 1):
        current = impedance_data[i]
        next_point = impedance_data[i + 1]
        delta_freq = next_point.frequency - current.frequency
        avg_real = (current.real + next_point.real) / 2
        integral += avg_real * delta_freq
    return 1 / ((2 / np.pi) * integral)

def calculate_nyquist_characteristics(impedance_data: List[ImpedancePoint]) -> Dict[str, float]:
    """Calculate Nyquist plot characteristics"""
    # Find low-frequency intercept (TER)
    low_freq_point = impedance_data[-1]
    
    # Find high-frequency point
    high_freq_point = impedance_data[0]
    
    # Calculate semicircle area (related to TEC)
    area = 0
    for i in range(len(impedance_data) - 1):
        current = impedance_data[i]
        next_point = impedance_data[i + 1]
        area += abs(
            (current.real - high_freq_point.real) * (next_point.imaginary - current.imaginary) -
            (current.imaginary - next_point.imaginary) * (next_point.real - current.real)
        ) / 2
    
    return {
        "TER": low_freq_point.real,
        "semicircle_area": area
    }

# Add a global variable to store computation logs
global_computation_logs = []

# Function to add a log message
def add_log(message: str):
    """Add a log message to both the logger and global logs list"""
    logger.info(message)
    # Keep only the last 100 logs
    global global_computation_logs
    global_computation_logs.append(message)
    if len(global_computation_logs) > 100:
        global_computation_logs = global_computation_logs[-100:]

# Add an endpoint to get computation progress
@app.get("/api/mesh_progress")
async def get_mesh_progress():
    """Get current computation progress and logs"""
    return {
        "logs": global_computation_logs,
        "timestamp": time.time()
    }

@app.post("/api/compute_impedance")
async def compute_impedance(params: CircuitParameters) -> List[ImpedancePoint]:
    """Compute impedance for a range of frequencies."""
    try:
        return calculate_impedance_spectrum(params)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/calculate_resnorm")
async def compute_resnorm(params: ResnormParameters) -> float:
    """Calculate resnorm between test data and ground truth data."""
    try:
        if params.use_ground_truth:
            reference_data = [ImpedancePoint(**point) for point in GROUND_TRUTH_DATASET]
        else:
            reference_data = params.test_data
        
        # Calculate resnorm using the improved method
        sum_of_squared_residuals = 0
        n = min(len(reference_data), len(params.test_data))

        for i in range(n):
            ref = reference_data[i]
            test = params.test_data[i]
            
            # Calculate residuals for real and imaginary components
            real_residual = test.real - ref.real
            imag_residual = test.imaginary - ref.imaginary
            
            # Calculate the squared difference
            residual_squared = real_residual ** 2 + imag_residual ** 2
            
            sum_of_squared_residuals += residual_squared
        
        return float(np.sqrt(sum_of_squared_residuals))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/regression_mesh")
async def compute_regression_mesh(params: RegressionMeshParameters) -> List[MeshPoint]:
    """Compute regression mesh analysis for parameter space exploration."""
    try:
        # Clear previous logs at the start of a new computation
        global global_computation_logs
        global_computation_logs = []
        
        start_time = time.time()
        add_log("Starting regression mesh computation")
        
        # Use frequencies from the request
        frequencies = params.reference_cell.frequency_range
        if not frequencies:
            frequencies = np.logspace(0, 4, 20)  # Fallback to default if none provided
        add_log(f"Using {len(frequencies)} frequency points")
        
        # Use ground truth or generate reference data from parameters
        if params.use_ground_truth:
            reference_data = [ImpedancePoint(**point) for point in GROUND_TRUTH_DATASET]
            add_log("Using ground truth dataset for reference data")
        else:
            try:
                reference_data = [
                    calculate_impedance_spectrum(params.reference_cell)[i]
                    for i in range(len(params.reference_cell.frequency_range))
                ]
                add_log(f"Generated reference data in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error generating reference data: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error generating reference data: {str(e)}"
                )
        
        # Generate parameter mesh
        try:
            # Modify parameter space to be more reasonable - reduces computation time
            param_space = ParameterSpace(
                ra=(100, 1000),               # Apical resistance range (Ω)
                ca=(1e-6, 5e-6),              # Apical capacitance range (F)
                rb=(100, 1000),               # Basal resistance range (Ω)
                cb=(1e-6, 5e-6),              # Basal capacitance range (F)
                noise_level=params.noise_level
            )
            
            # Use the full resolution requested
            mesh = generate_parameter_mesh(params.mesh_resolution, param_space)
            add_log(f"Generated parameter mesh with {len(mesh)} points")
        except Exception as e:
            logger.error(f"Error generating parameter mesh: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error generating parameter mesh: {str(e)}"
            )
        
        # Calculate resnorm for each point
        results = []
        resnorms = []
        batch_size = 50  # Process in smaller batches to reduce memory usage
        
        add_log(f"Processing {len(mesh)} points in batches of {batch_size}")
        
        for i in range(0, len(mesh), batch_size):
            batch = mesh[i:i+batch_size]
            batch_results = []
            
            add_log(f"Processing batch {i//batch_size + 1}/{(len(mesh) + batch_size - 1)//batch_size}, points {i+1}-{min(i+batch_size, len(mesh))}")
            
            for j, point in enumerate(batch):
                test_params = CircuitParameters(
                    Rs=float(point[0]),
                    Ra=float(point[1]),
                    Ca=float(point[2]),
                    Rb=float(point[3]),
                    Cb=float(point[4]),
                    frequency_range=frequencies
                )
                
                # Calculate test data for the same frequencies as reference data
                test_data = calculate_impedance_spectrum(test_params)
                
                # Calculate resnorm using the improved method that matches the frontend
                resnorm = calculate_resnorm_improved(reference_data, test_data)
                resnorms.append(resnorm)
                
                # Create a MeshPoint with the correct format
                mesh_point = MeshPoint(
                    parameters=test_params,
                    resnorm=resnorm,
                    alpha=0.0  # Default alpha value
                )
                batch_results.append(mesh_point)
                
                # Log progress for every 20% of points in the batch
                if (j + 1) % max(1, len(batch) // 5) == 0:
                    add_log(f"  Processed {j+1}/{len(batch)} points in current batch, latest resnorm: {resnorm:.4f}")
            
            results.extend(batch_results)
            add_log(f"Completed batch {i//batch_size + 1}, elapsed time: {time.time() - start_time:.2f}s")
        
        add_log(f"Completed all mesh calculations in {time.time() - start_time:.2f}s")
        add_log(f"Total results generated: {len(results)}")
        if results:
             add_log(f"First 3 results (before filtering): {results[:3]}") # Log first 3 results
        
        if not results:
            logger.error("No valid results generated from mesh analysis")
            raise HTTPException(
                status_code=400,
                detail="No valid results generated from mesh analysis"
            )
        
        # Calculate alpha values and filter top percentage
        try:
            resnorms = np.array(resnorms)
            # Check for non-finite values before calculating min/max
            finite_resnorms = resnorms[np.isfinite(resnorms)]
            if len(finite_resnorms) == 0:
                logger.error("All resnorm values are non-finite. Cannot proceed with filtering.")
                raise HTTPException(
                    status_code=400,
                    detail="All calculated resnorm values are non-finite. Check input parameters or calculation logic."
                )
            
            min_resnorm = np.min(finite_resnorms)
            max_resnorm = np.max(finite_resnorms)
            
            add_log(f"Resnorm range (finite values): {min_resnorm:.4f} - {max_resnorm:.4f}")
            
            # Keep only top percentage of fits based on finite resnorms
            # Handle cases where all resnorms might be identical
            if min_resnorm == max_resnorm:
                 threshold = max_resnorm # Keep all if they are the same
                 add_log(f"All finite resnorms are identical ({max_resnorm:.4f}). Keeping all finite results.")
            else:
                threshold = np.percentile(finite_resnorms, params.top_percentage)
                add_log(f"Using top {params.top_percentage}% threshold: {threshold:.4f}")
            
            filtered_results = []
            
            for result in results:
                # Only process results with finite resnorm values
                if np.isfinite(result.resnorm) and result.resnorm <= threshold:
                    # Calculate alpha based on the range of finite resnorms
                    alpha = calculate_alpha(result.resnorm, min_resnorm, max_resnorm)
                    result.alpha = alpha
                    filtered_results.append(result)
            
            add_log(f"Filtered to {len(filtered_results)} results out of {len(results)}")
            if filtered_results:
                add_log(f"First 3 results (after filtering): {filtered_results[:3]}") # Log first 3 filtered results
            add_log(f"Total computation time: {time.time() - start_time:.2f}s")
            
            if not filtered_results:
                logger.error("No results remained after filtering")
                raise HTTPException(
                    status_code=400,
                    detail="No results remained after filtering"
                )
            
            # Sort by resnorm for better results
            filtered_results.sort(key=lambda x: x.resnorm)
            add_log(f"Best resnorm value: {filtered_results[0].resnorm:.4f}")
            
            # Return only the top 10 results max to reduce payload size
            filtered_results = filtered_results[:10]
            add_log(f"Returning top {len(filtered_results)} results")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing results: {str(e)}"
            )
        
    except HTTPException:
        # Re-raise HTTPException directly
        raise
    except Exception as e:
        # Log the detailed error traceback
        logger.exception(f"Unexpected error during regression mesh computation: {str(e)}")
        # Return a 500 error with a clear detail message
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during computation: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/state/all")
async def get_all_states():
    """Return all saved states (handled on client side for now)."""
    # This is a stub endpoint that returns an empty array
    # The actual state management is done on the client side for now
    return []

def calculate_resnorm_improved(reference_data: List[ImpedancePoint], test_data: List[ImpedancePoint]) -> float:
    """Calculate the residual norm between reference and test impedance data.
    This is used to measure the goodness of fit between simulated and measured data.
    """
    sum_of_squared_residuals = 0
    n = min(len(reference_data), len(test_data))

    for i in range(n):
        ref = reference_data[i]
        test = test_data[i]
        
        # Calculate residuals for real and imaginary components
        real_residual = test.real - ref.real
        imag_residual = test.imaginary - ref.imaginary
        
        # Calculate the squared difference
        residual_squared = real_residual ** 2 + imag_residual ** 2
        
        sum_of_squared_residuals += residual_squared
    
    return float(np.sqrt(sum_of_squared_residuals))

def generate_parameter_mesh(resolution: int, param_space: ParameterSpace) -> List[List[float]]:
    """Generate a parameter mesh for circuit parameter space exploration.
    
    Args:
        resolution: Number of points per dimension
        param_space: Parameter space with ranges for each parameter
        
    Returns:
        List of parameter combinations (Rs, Ra, Ca, Rb, Cb)
    """
    # Define the parameter ranges
    Rs_range = np.linspace(100, 1000, resolution)  # Tight junction resistance
    Ra_range = np.linspace(param_space.ra[0], param_space.ra[1], resolution)  # Apical resistance
    Ca_range = np.logspace(np.log10(param_space.ca[0]), np.log10(param_space.ca[1]), resolution)  # Apical capacitance
    Rb_range = np.linspace(param_space.rb[0], param_space.rb[1], resolution)  # Basal resistance
    Cb_range = np.logspace(np.log10(param_space.cb[0]), np.log10(param_space.cb[1]), resolution)  # Basal capacitance
    
    # Generate combinations of parameters (use a subset to reduce computation time)
    mesh = []
    for Rs in Rs_range:
        for Ra in Ra_range[::max(1, resolution//4)]:
            for Ca in Ca_range[::max(1, resolution//4)]:
                for Rb in Rb_range[::max(1, resolution//4)]:
                    for Cb in Cb_range[::max(1, resolution//4)]:
                        mesh.append([Rs, Ra, Ca, Rb, Cb])
    
    return mesh

def calculate_alpha(resnorm: float, min_resnorm: float, max_resnorm: float) -> float:
    """Calculate alpha value (transparency) based on resnorm.
    
    Args:
        resnorm: Residual norm value
        min_resnorm: Minimum resnorm in the dataset
        max_resnorm: Maximum resnorm in the dataset
        
    Returns:
        Alpha value between 0 and 1
    """
    if min_resnorm == max_resnorm:
        return 1.0  # If all values are the same, use full opacity
    
    # Normalize and invert (lower resnorm = higher alpha)
    normalized = (resnorm - min_resnorm) / (max_resnorm - min_resnorm)
    alpha = 1.0 - normalized
    
    # Limit range to [0.1, 1.0] so even poor fits are somewhat visible
    return max(0.1, min(1.0, alpha))

# Start the server when the script is run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

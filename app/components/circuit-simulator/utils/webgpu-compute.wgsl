// WebGPU Compute Shader for Circuit Impedance Calculation
// Implements the Randles equivalent circuit model: Z(ω) = Rs + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)

// Struct definitions for data layout
struct CircuitParams {
    Rs: f32,  // Shunt resistance
    Ra: f32,  // Apical resistance  
    Ca: f32,  // Apical capacitance
    Rb: f32,  // Basal resistance
    Cb: f32,  // Basal capacitance
    pad0: f32, // Padding for 4-component alignment
    pad1: f32,
    pad2: f32,
};

struct ComplexNumber {
    real: f32,
    imag: f32,
};

struct ImpedanceResult {
    frequency: f32,
    real: f32,
    imag: f32,
    magnitude: f32,
    phase: f32,
    resnorm: f32,
    pad0: f32, // Padding for alignment
    pad1: f32,
};

// Bind group 0: Input/Output buffers
@group(0) @binding(0) var<storage, read> circuit_parameters: array<CircuitParams>;
@group(0) @binding(1) var<storage, read> frequencies: array<f32>;
@group(0) @binding(2) var<storage, read> reference_spectrum: array<ComplexNumber>;
@group(0) @binding(3) var<storage, read_write> results: array<ImpedanceResult>;

// Bind group 1: Constants and configuration
@group(1) @binding(0) var<uniform> config: vec4<u32>; // [num_params, num_freqs, resnorm_method, padding]

// Complex number operations
fn complex_add(a: ComplexNumber, b: ComplexNumber) -> ComplexNumber {
    return ComplexNumber(a.real + b.real, a.imag + b.imag);
}

fn complex_multiply(a: ComplexNumber, b: ComplexNumber) -> ComplexNumber {
    return ComplexNumber(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

fn complex_divide(a: ComplexNumber, b: ComplexNumber) -> ComplexNumber {
    let denom = b.real * b.real + b.imag * b.imag;
    return ComplexNumber(
        (a.real * b.real + a.imag * b.imag) / denom,
        (a.imag * b.real - a.real * b.imag) / denom
    );
}

fn complex_magnitude(z: ComplexNumber) -> f32 {
    return sqrt(z.real * z.real + z.imag * z.imag);
}

fn complex_phase(z: ComplexNumber) -> f32 {
    return atan2(z.imag, z.real) * (180.0 / 3.141592653589793);
}

// Calculate impedance for a single parameter set and frequency
fn calculate_impedance(params: CircuitParams, omega: f32) -> ComplexNumber {
    // Za = Ra/(1+jωRaCa)
    let za_denom = ComplexNumber(1.0, omega * params.Ra * params.Ca);
    let za = complex_divide(ComplexNumber(params.Ra, 0.0), za_denom);
    
    // Zb = Rb/(1+jωRbCb)  
    let zb_denom = ComplexNumber(1.0, omega * params.Rb * params.Cb);
    let zb = complex_divide(ComplexNumber(params.Rb, 0.0), zb_denom);
    
    // Z_total = Rs + Za + Zb (series combination)
    let z_series = complex_add(complex_add(ComplexNumber(params.Rs, 0.0), za), zb);
    
    return z_series;
}

// Calculate resnorm using MAE method (configurable via uniform)
fn calculate_resnorm(test_spectrum: array<ComplexNumber>, freq_count: u32) -> f32 {
    var total_error: f32 = 0.0;
    var valid_points: u32 = 0u;
    
    for (var i = 0u; i < freq_count; i = i + 1u) {
        let ref_point = reference_spectrum[i];
        let test_point = test_spectrum[i];
        
        let ref_mag = complex_magnitude(ref_point);
        let test_mag = complex_magnitude(test_point);
        
        // Skip points with zero reference magnitude
        if (ref_mag > 0.0001) {
            let normalized_residual = abs((test_mag - ref_mag) / ref_mag);
            total_error = total_error + normalized_residual;
            valid_points = valid_points + 1u;
        }
    }
    
    if (valid_points > 0u) {
        return total_error / f32(valid_points);
    } else {
        return 1e9; // Large error for invalid cases
    }
}

// Main compute shader - processes multiple parameter sets in parallel
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let param_idx = global_id.x;
    let num_params = config.x;
    let num_freqs = config.y;
    
    // Bounds check
    if (param_idx >= num_params) {
        return;
    }
    
    let params = circuit_parameters[param_idx];
    
    // Calculate impedance spectrum for this parameter set
    var test_spectrum: array<ComplexNumber, 256>; // Max 256 frequency points
    var total_resnorm: f32 = 0.0;
    
    for (var freq_idx = 0u; freq_idx < min(num_freqs, 256u); freq_idx = freq_idx + 1u) {
        let frequency = frequencies[freq_idx];
        let omega = 2.0 * 3.141592653589793 * frequency;
        
        let impedance = calculate_impedance(params, omega);
        test_spectrum[freq_idx] = impedance;
        
        // Store result for the first frequency point (for visualization)
        if (freq_idx == 0u) {
            let result_idx = param_idx * num_freqs + freq_idx;
            if (result_idx < arrayLength(&results)) {
                results[result_idx] = ImpedanceResult(
                    frequency,
                    impedance.real,
                    impedance.imag,
                    complex_magnitude(impedance),
                    complex_phase(impedance),
                    0.0, // Will be updated with resnorm
                    0.0,
                    0.0
                );
            }
        }
    }
    
    // Calculate resnorm for this parameter set
    let resnorm = calculate_resnorm(test_spectrum, min(num_freqs, 256u));
    
    // Update all results for this parameter set with resnorm
    for (var freq_idx = 0u; freq_idx < min(num_freqs, 256u); freq_idx = freq_idx + 1u) {
        let result_idx = param_idx * num_freqs + freq_idx;
        if (result_idx < arrayLength(&results)) {
            if (freq_idx < min(num_freqs, 256u)) {
                let frequency = frequencies[freq_idx];
                let impedance = test_spectrum[freq_idx];
                
                results[result_idx] = ImpedanceResult(
                    frequency,
                    impedance.real,
                    impedance.imag,
                    complex_magnitude(impedance),
                    complex_phase(impedance),
                    resnorm,
                    0.0,
                    0.0
                );
            }
        }
    }
}
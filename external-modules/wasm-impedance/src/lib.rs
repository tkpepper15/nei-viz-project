use wasm_bindgen::prelude::*;
use js_sys::Array;

// Circuit parameter structure
#[wasm_bindgen]
pub struct CircuitParams {
    pub rsh: f64,  // Shunt resistance (Ω)
    pub ra: f64,   // Apical resistance (Ω)
    pub ca: f64,   // Apical capacitance (F)
    pub rb: f64,   // Basal resistance (Ω)
    pub cb: f64,   // Basal capacitance (F)
}

#[wasm_bindgen]
impl CircuitParams {
    #[wasm_bindgen(constructor)]
    pub fn new(rsh: f64, ra: f64, ca: f64, rb: f64, cb: f64) -> CircuitParams {
        CircuitParams { rsh, ra, ca, rb, cb }
    }
}

// Complex number structure for impedance calculations
#[derive(Clone, Copy)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }

    fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    fn add(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }

    fn multiply(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    fn divide(&self, other: &Complex) -> Complex {
        let denom = other.real * other.real + other.imag * other.imag;
        Complex {
            real: (self.real * other.real + self.imag * other.imag) / denom,
            imag: (self.imag * other.real - self.real * other.imag) / denom,
        }
    }
}

// Core impedance calculation for Randles circuit
fn calculate_impedance(params: &CircuitParams, omega: f64) -> Complex {
    let tau_a = params.ra * params.ca;
    let tau_b = params.rb * params.cb;
    
    // Za = Ra / (1 + jωτa)
    let za_denom = Complex::new(1.0, omega * tau_a);
    let za = Complex::new(params.ra, 0.0).divide(&za_denom);
    
    // Zb = Rb / (1 + jωτb)  
    let zb_denom = Complex::new(1.0, omega * tau_b);
    let zb = Complex::new(params.rb, 0.0).divide(&zb_denom);
    
    // Z_series = Za + Zb
    let z_series = za.add(&zb);
    
    // Z_total = (Rsh * Z_series) / (Rsh + Z_series)
    let rsh_complex = Complex::new(params.rsh, 0.0);
    let numerator = rsh_complex.multiply(&z_series);
    let denominator = rsh_complex.add(&z_series);
    
    numerator.divide(&denominator)
}

// Batch compute impedance spectra for multiple parameters
#[wasm_bindgen]
pub fn compute_impedance_batch(
    param_data: &[f64],  // Flattened: [Rsh1, Ra1, Ca1, Rb1, Cb1, Rsh2, ...]
    frequencies: &[f64],
    output: &mut [f64],  // Flattened: [real1_f1, imag1_f1, mag1_f1, phase1_f1, ...]
) {
    let param_count = param_data.len() / 5;
    let freq_count = frequencies.len();
    
    for param_idx in 0..param_count {
        let param_offset = param_idx * 5;
        let params = CircuitParams {
            rsh: param_data[param_offset],
            ra: param_data[param_offset + 1], 
            ca: param_data[param_offset + 2],
            rb: param_data[param_offset + 3],
            cb: param_data[param_offset + 4],
        };
        
        for (freq_idx, &frequency) in frequencies.iter().enumerate() {
            let omega = 2.0 * std::f64::consts::PI * frequency;
            let impedance = calculate_impedance(&params, omega);
            
            let output_offset = (param_idx * freq_count + freq_idx) * 4;
            output[output_offset] = impedance.real;
            output[output_offset + 1] = impedance.imag;
            output[output_offset + 2] = impedance.magnitude();
            output[output_offset + 3] = impedance.phase();
        }
    }
}

// Compute spectral fingerprints for fast screening
#[wasm_bindgen]
pub fn compute_fingerprints(
    param_data: &[f64],
    fp_frequencies: &[f64],  // 6-8 fingerprint frequencies
    output_fingerprints: &mut [f64],  // Output: log10(magnitudes)
) {
    let param_count = param_data.len() / 5;
    let fp_freq_count = fp_frequencies.len();
    
    for param_idx in 0..param_count {
        let param_offset = param_idx * 5;
        let params = CircuitParams {
            rsh: param_data[param_offset],
            ra: param_data[param_offset + 1],
            ca: param_data[param_offset + 2], 
            rb: param_data[param_offset + 3],
            cb: param_data[param_offset + 4],
        };
        
        for (freq_idx, &frequency) in fp_frequencies.iter().enumerate() {
            let omega = 2.0 * std::f64::consts::PI * frequency;
            let impedance = calculate_impedance(&params, omega);
            let log_magnitude = impedance.magnitude().log10();
            
            let output_offset = param_idx * fp_freq_count + freq_idx;
            output_fingerprints[output_offset] = log_magnitude;
        }
    }
}

// Compute cheap resnorm for screening (magnitude-only, fewer frequencies)
#[wasm_bindgen]  
pub fn compute_cheap_resnorm(
    param_data: &[f64],
    approx_frequencies: &[f64],  // 5 frequencies for cheap screening
    reference_magnitudes: &[f64],  // Reference magnitudes at these frequencies
    output_resnorms: &mut [f64],
) {
    let param_count = param_data.len() / 5;
    let freq_count = approx_frequencies.len();
    
    for param_idx in 0..param_count {
        let param_offset = param_idx * 5;
        let params = CircuitParams {
            rsh: param_data[param_offset],
            ra: param_data[param_offset + 1],
            ca: param_data[param_offset + 2],
            rb: param_data[param_offset + 3], 
            cb: param_data[param_offset + 4],
        };
        
        let mut total_error = 0.0;
        let mut valid_points = 0;
        
        for (freq_idx, &frequency) in approx_frequencies.iter().enumerate() {
            let omega = 2.0 * std::f64::consts::PI * frequency;
            let impedance = calculate_impedance(&params, omega);
            let test_magnitude = impedance.magnitude();
            let ref_magnitude = reference_magnitudes[freq_idx];
            
            let error = (test_magnitude - ref_magnitude).abs();
            total_error += error;
            valid_points += 1;
        }
        
        output_resnorms[param_idx] = if valid_points > 0 { 
            total_error / valid_points as f64 
        } else { 
            f64::INFINITY 
        };
    }
}

// Compute full complex SSR for final refinement
#[wasm_bindgen]
pub fn compute_full_ssr(
    param_data: &[f64],
    full_frequencies: &[f64],  // 20-40 frequencies for full SSR
    reference_real: &[f64],    // Reference real parts
    reference_imag: &[f64],    // Reference imaginary parts
    output_ssr: &mut [f64],
) {
    let param_count = param_data.len() / 5;
    let freq_count = full_frequencies.len();
    
    for param_idx in 0..param_count {
        let param_offset = param_idx * 5;
        let params = CircuitParams {
            rsh: param_data[param_offset],
            ra: param_data[param_offset + 1],
            ca: param_data[param_offset + 2],
            rb: param_data[param_offset + 3],
            cb: param_data[param_offset + 4],
        };
        
        let mut total_ssr = 0.0;
        let mut valid_points = 0;
        
        for (freq_idx, &frequency) in full_frequencies.iter().enumerate() {
            let omega = 2.0 * std::f64::consts::PI * frequency;
            let impedance = calculate_impedance(&params, omega);
            
            let real_diff = impedance.real - reference_real[freq_idx];
            let imag_diff = impedance.imag - reference_imag[freq_idx];
            let ssr_contribution = real_diff * real_diff + imag_diff * imag_diff;
            
            total_ssr += ssr_contribution;
            valid_points += 1;
        }
        
        output_ssr[param_idx] = if valid_points > 0 {
            total_ssr / valid_points as f64
        } else {
            f64::INFINITY
        };
    }
}

// Utility: quantize fingerprint for grouping
#[wasm_bindgen]
pub fn quantize_fingerprint(
    fingerprint: &[f64],
    quantization_bin: f64,  // e.g., 0.05 dex
    output: &mut [f64],
) {
    for (i, &value) in fingerprint.iter().enumerate() {
        output[i] = (value / quantization_bin).round() * quantization_bin;
    }
}
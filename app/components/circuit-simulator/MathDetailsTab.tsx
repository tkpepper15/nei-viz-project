import React from 'react';
import { CircuitParameters } from '../circuit-simulator/utils/impedance';

type MathDetailsTabProps = {
  parameters: CircuitParameters;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  referenceModel?: {
    parameters: CircuitParameters;
    name?: string;
  } | null;
};

// Helper functions
function calculateTER(params: CircuitParameters): number {
  const numerator = params.Rs * (params.Ra + params.Rb);
  const denominator = params.Rs + params.Ra + params.Rb;
  return numerator / denominator;
}

function calculateTEC(params: CircuitParameters): number {
  // TEC calculation formula: CaCb/(Ca+Cb)
  return (params.Ca * params.Cb) / (params.Ca + params.Cb);
}

// Calculate impedance at a specific frequency
function calculateImpedance(params: CircuitParameters, frequency: number): { real: number; imaginary: number; magnitude: number; phase: number } {
  const omega = 2 * Math.PI * frequency;
  
  // Calculate apical membrane impedance
  const Za_real = params.Ra / (1 + Math.pow(omega * params.Ra * params.Ca, 2));
  const Za_imag = -omega * Math.pow(params.Ra, 2) * params.Ca / (1 + Math.pow(omega * params.Ra * params.Ca, 2));
  
  // Calculate basal membrane impedance
  const Zb_real = params.Rb / (1 + Math.pow(omega * params.Rb * params.Cb, 2));
  const Zb_imag = -omega * Math.pow(params.Rb, 2) * params.Cb / (1 + Math.pow(omega * params.Rb * params.Cb, 2));
  
  // Calculate total apical + basal impedance
  const Zab_real = Za_real + Zb_real;
  const Zab_imag = Za_imag + Zb_imag;
  
  // Calculate parallel combination with Rs
  const denominator = Math.pow(params.Rs + Zab_real, 2) + Math.pow(Zab_imag, 2);
  
  const Z_real = (params.Rs * Zab_real) / denominator;
  const Z_imag = (params.Rs * Zab_imag) / denominator;
  
  // Calculate magnitude and phase
  const magnitude = Math.sqrt(Math.pow(Z_real, 2) + Math.pow(Z_imag, 2));
  const phase = Math.atan2(Z_imag, Z_real) * (180 / Math.PI);
  
  return { real: Z_real, imaginary: Z_imag, magnitude, phase };
}

// Define proper interfaces for resnorm details
interface ResnormDetail {
  frequency: number;
  z1: { real: number; imaginary: number; magnitude: number };
  z2: { real: number; imaginary: number; magnitude: number };
  residuals: { real: number; imaginary: number; combined: number };
  weight: number;
  weightedResidual: number;
}

interface ResnormSummary {
  sumOfSquaredResiduals: number;
  sumWeights: number;
  finalResnorm: number;
}

// Extend Window interface to add our properties
declare global {
  interface Window {
    resnormDetails: ResnormDetail[];
    resnormSummary: ResnormSummary;
  }
}

// Calculate resnorm between two parameter sets
function calculateResnorm(params1: CircuitParameters, params2: CircuitParameters, frequencies: number[]): number {
  let sumOfSquaredResiduals = 0;
  let sumWeights = 0;
  
  // Store details for display
  const details: ResnormDetail[] = [];
  
  for (const freq of frequencies) {
    const z1 = calculateImpedance(params1, freq);
    const z2 = calculateImpedance(params2, freq);
    
    // Magnitude of reference for normalization
    const refMagnitude = Math.sqrt(z2.real * z2.real + z2.imaginary * z2.imaginary);
    const normFactor = refMagnitude > 0 ? refMagnitude : 1;
    
    // Calculate normalized residuals for real and imaginary components
    const realResidual = (z1.real - z2.real) / normFactor;
    const imagResidual = (z1.imaginary - z2.imaginary) / normFactor;
    
    // Calculate the weighted squared difference (weighted by frequency importance)
    const weight = 1 / Math.max(1, Math.log10(freq));
    const residualSquared = (realResidual * realResidual + imagResidual * imagResidual) * weight;
    
    // Add to running totals
    sumOfSquaredResiduals += residualSquared;
    sumWeights += weight;
    
    // Store calculation details
    details.push({
      frequency: freq,
      z1: { 
        real: z1.real, 
        imaginary: z1.imaginary,
        magnitude: Math.sqrt(z1.real * z1.real + z1.imaginary * z1.imaginary)
      },
      z2: { 
        real: z2.real, 
        imaginary: z2.imaginary,
        magnitude: refMagnitude
      },
      residuals: {
        real: realResidual,
        imaginary: imagResidual,
        combined: Math.sqrt(realResidual * realResidual + imagResidual * imagResidual)
      },
      weight,
      weightedResidual: residualSquared
    });
  }
  
  // Store details for rendering
  window.resnormDetails = details;
  window.resnormSummary = {
    sumOfSquaredResiduals,
    sumWeights,
    finalResnorm: Math.sqrt(sumOfSquaredResiduals / sumWeights)
  };
  
  return Math.sqrt(sumOfSquaredResiduals / sumWeights);
}

const MathDetailsTab: React.FC<MathDetailsTabProps> = ({ 
  parameters, 
  minFreq, 
  maxFreq, 
  numPoints,
  referenceModel
}) => {
  // Calculate derived values
  const terValue = calculateTER(parameters);
  const tecValue = calculateTEC(parameters);
  
  // Generate impedance values at several frequencies for debugging
  const frequencies = [
    minFreq,
    minFreq * 2,
    minFreq * 5,
    minFreq * 10,
    Math.sqrt(minFreq * maxFreq), // Geometric midpoint
    maxFreq / 5,
    maxFreq / 2,
    maxFreq
  ];
  
  // Calculate impedance values for all frequencies for debugging
  const impedanceValues = frequencies.map(f => ({
    frequency: f,
    ...calculateImpedance(parameters, f)
  }));
  
  // Calculate resnorm if reference model exists
  const resnorm = referenceModel ? calculateResnorm(parameters, referenceModel.parameters, frequencies) : 0;
  
  const equations = [
    { name: 'Epithelial Impedance Model', equation: 'Z_eq(ω) = (Rs * (Za(ω) + Zb(ω))) / (Rs + Za(ω) + Zb(ω))' },
    { name: 'Apical Membrane Impedance', equation: 'Za(ω) = Ra/(1+jωRaCa)' },
    { name: 'Basal Membrane Impedance', equation: 'Zb(ω) = Rb/(1+jωRbCb)' },
    { name: 'Angular Frequency', equation: 'ω = 2πf' },
    { name: 'Transepithelial Resistance (TER)', equation: 'TER = Rs * (Ra + Rb) / (Rs + Ra + Rb)' },
    { name: 'Transepithelial Capacitance (TEC)', equation: 'TEC = (Ca * Cb) / (Ca + Cb)' },
  ];

  return (
    <div className="card p-6 space-y-6">
      <div className="flex items-center border-b border-neutral-700 pb-4 mb-6">
        <svg className="w-5 h-5 mr-3 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
        </svg>
        <h3 className="text-lg font-medium text-neutral-200">Mathematical Details</h3>
      </div>
      
      {/* Current Parameters and Calculated Values */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
          <h4 className="text-sm font-medium text-neutral-300 mb-4 flex items-center">
            <svg className="w-4 h-4 mr-2 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
            Current Parameters
          </h4>
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 bg-neutral-700/30 rounded">
              <div className="text-xs text-neutral-400">Rs</div>
              <div className="font-mono font-medium text-neutral-200">{parameters.Rs.toFixed(2)} Ω</div>
            </div>
            <div className="p-3 bg-neutral-700/30 rounded">
              <div className="text-xs text-neutral-400">Ra</div>
              <div className="font-mono font-medium text-neutral-200">{parameters.Ra.toFixed(0)} Ω</div>
            </div>
            <div className="p-3 bg-neutral-700/30 rounded">
              <div className="text-xs text-neutral-400">Ca</div>
              <div className="font-mono font-medium text-neutral-200">{(parameters.Ca * 1e6).toFixed(2)} μF</div>
            </div>
            <div className="p-3 bg-neutral-700/30 rounded">
              <div className="text-xs text-neutral-400">Rb</div>
              <div className="font-mono font-medium text-neutral-200">{parameters.Rb.toFixed(0)} Ω</div>
            </div>
            <div className="p-3 bg-neutral-700/30 rounded">
              <div className="text-xs text-neutral-400">Cb</div>
              <div className="font-mono font-medium text-neutral-200">{(parameters.Cb * 1e6).toFixed(2)} μF</div>
            </div>
          </div>

          {referenceModel && (
            <div className="mt-4">
              <div className="flex items-center justify-between">
                <h5 className="text-xs font-medium text-neutral-300 flex items-center">
                  <svg className="w-3 h-3 mr-1 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Reference Model {referenceModel.name && `(${referenceModel.name})`}
                </h5>
              </div>
              <div className="grid grid-cols-5 gap-2 mt-2">
                <div className="p-2 bg-neutral-800/60 rounded">
                  <div className="text-xs text-neutral-500">Rs</div>
                  <div className="font-mono text-xs text-white">{referenceModel.parameters.Rs.toFixed(0)} Ω</div>
                </div>
                <div className="p-2 bg-neutral-800/60 rounded">
                  <div className="text-xs text-neutral-500">Ra</div>
                  <div className="font-mono text-xs text-white">{referenceModel.parameters.Ra.toFixed(0)} Ω</div>
                </div>
                <div className="p-2 bg-neutral-800/60 rounded">
                  <div className="text-xs text-neutral-500">Ca</div>
                  <div className="font-mono text-xs text-white">{(referenceModel.parameters.Ca * 1e6).toFixed(2)} μF</div>
                </div>
                <div className="p-2 bg-neutral-800/60 rounded">
                  <div className="text-xs text-neutral-500">Rb</div>
                  <div className="font-mono text-xs text-white">{referenceModel.parameters.Rb.toFixed(0)} Ω</div>
                </div>
                <div className="p-2 bg-neutral-800/60 rounded">
                  <div className="text-xs text-neutral-500">Cb</div>
                  <div className="font-mono text-xs text-white">{(referenceModel.parameters.Cb * 1e6).toFixed(2)} μF</div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
          <h4 className="text-sm font-medium text-neutral-300 mb-4 flex items-center">
            <svg className="w-4 h-4 mr-2 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
            Calculated Values
          </h4>
          
          <div className="space-y-4">
            <div className="p-4 bg-neutral-700/30 rounded">
              <div className="text-sm text-neutral-400 mb-1">Transepithelial Resistance (TER)</div>
              <div className="font-mono font-medium text-xl text-neutral-200">{terValue.toFixed(1)} Ω</div>
              <div className="text-xs text-neutral-500 mt-2">Rs * (Ra + Rb) / (Rs + Ra + Rb)</div>
              
              {referenceModel && (
                <div className="mt-2 p-2 bg-neutral-800/60 rounded">
                  <div className="flex justify-between items-center">
                    <div className="text-xs text-neutral-500">Reference TER:</div>
                    <div className="font-mono text-xs text-white">
                      {calculateTER(referenceModel.parameters).toFixed(1)} Ω
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <div className="p-4 bg-neutral-700/30 rounded">
              <div className="text-sm text-neutral-400 mb-1">Transepithelial Capacitance (TEC)</div>
              <div className="font-mono font-medium text-xl text-neutral-200">{tecValue.toFixed(3)} μF</div>
              <div className="text-xs text-neutral-500 mt-2">(Ca * Cb) / (Ca + Cb)</div>
              
              {referenceModel && (
                <div className="mt-2 p-2 bg-neutral-800/60 rounded">
                  <div className="flex justify-between items-center">
                    <div className="text-xs text-neutral-500">Reference TEC:</div>
                    <div className="font-mono text-xs text-white">
                      {calculateTEC(referenceModel.parameters).toFixed(3)} μF
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <div className="p-4 bg-neutral-700/30 rounded">
              <div className="text-sm text-neutral-400 mb-1">Frequency Range</div>
              <div className="font-mono font-medium text-neutral-200">{minFreq.toFixed(1)} - {maxFreq.toFixed(1)} Hz</div>
              <div className="text-xs text-neutral-500 mt-2">Using {numPoints} logarithmically spaced points</div>
            </div>
            
            {referenceModel && (
              <div className="p-4 bg-neutral-700/30 rounded">
                <div className="text-sm text-neutral-400 mb-1">Resnorm (Current vs Reference)</div>
                <div className="font-mono font-medium text-xl text-neutral-200">{resnorm.toExponential(3)}</div>
                <div className="text-xs text-neutral-500 mt-2">
                  Weighted Root Mean Square Error (RMSE) of normalized impedance differences
                  <span className={`ml-2 px-2 py-0.5 rounded font-semibold ${
                    resnorm < 0.05 ? 'bg-green-500/20 text-green-400' : 
                    resnorm < 0.15 ? 'bg-blue-500/20 text-blue-400' : 
                    resnorm < 0.3 ? 'bg-amber-500/20 text-amber-400' : 
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {resnorm < 0.05 ? 'Excellent match' : 
                     resnorm < 0.15 ? 'Good match' : 
                     resnorm < 0.3 ? 'Moderate match' : 'Poor match'}
                  </span>
                </div>
                <div className="text-xs text-neutral-500 mt-1">
                  <ul className="list-disc pl-4 space-y-1">
                    <li>Industry-standard metric for comparing impedance spectra</li>
                    <li>Normalized by reference impedance magnitude at each frequency</li>
                    <li>Weighted to emphasize lower frequencies (more important for cell characterization)</li>
                    <li>Values &lt; 0.05 typically indicate excellent model fit</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Mathematical Formulas */}
      <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700 mt-6">
        <h4 className="text-sm font-medium text-neutral-300 mb-4 flex items-center">
          <svg className="w-4 h-4 mr-2 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
          Mathematical Formulations
        </h4>
        
        <div className="space-y-3">
          {equations.map((eq, idx) => (
            <div key={idx} className="p-3 bg-neutral-700/30 rounded">
              <div className="text-xs text-neutral-400 mb-1">{eq.name}</div>
              <div className="font-mono font-medium text-neutral-200">{eq.equation}</div>
            </div>
          ))}
          
          <div className="p-3 bg-neutral-700/30 rounded">
            <div className="text-xs text-neutral-400 mb-1">Step-by-Step Impedance Calculation</div>
            <div className="font-mono text-xs text-neutral-400 bg-neutral-900/50 p-3 rounded-md mt-1 whitespace-pre overflow-x-auto">
{`// 1. Calculate angular frequency
ω = 2π * f = 2π * ${frequencies[4].toFixed(1)} = ${(2 * Math.PI * frequencies[4]).toFixed(1)} rad/s

// 2. Calculate apical membrane impedance
Za_real = Ra / (1 + (ω*Ra*Ca)²) = ${parameters.Ra.toFixed(1)} / (1 + (${(2 * Math.PI * frequencies[4]).toFixed(1)}*${parameters.Ra.toFixed(1)}*${parameters.Ca.toExponential(4)})²) = ${(parameters.Ra / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2))).toFixed(3)}
Za_imag = -ω*Ra²*Ca / (1 + (ω*Ra*Ca)²) = -${(2 * Math.PI * frequencies[4]).toFixed(1)}*${Math.pow(parameters.Ra, 2).toFixed(1)}*${parameters.Ca.toExponential(4)} / (1 + (${(2 * Math.PI * frequencies[4]).toFixed(1)}*${parameters.Ra.toFixed(1)}*${parameters.Ca.toExponential(4)})²) = ${(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Ra, 2) * parameters.Ca / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2))).toFixed(3)}

// 3. Calculate basal membrane impedance
Zb_real = Rb / (1 + (ω*Rb*Cb)²) = ${parameters.Rb.toFixed(1)} / (1 + (${(2 * Math.PI * frequencies[4]).toFixed(1)}*${parameters.Rb.toFixed(1)}*${parameters.Cb.toExponential(4)})²) = ${(parameters.Rb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)}
Zb_imag = -ω*Rb²*Cb / (1 + (ω*Rb*Cb)²) = -${(2 * Math.PI * frequencies[4]).toFixed(1)}*${Math.pow(parameters.Rb, 2).toFixed(1)}*${parameters.Cb.toExponential(4)} / (1 + (${(2 * Math.PI * frequencies[4]).toFixed(1)}*${parameters.Rb.toFixed(1)}*${parameters.Cb.toExponential(4)})²) = ${(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Rb, 2) * parameters.Cb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)}

// 4. Sum the apical and basal impedances
Zab_real = Za_real + Zb_real = ${(parameters.Ra / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2))).toFixed(3)} + ${(parameters.Rb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)} = ${(parameters.Ra / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) + parameters.Rb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)}
Zab_imag = Za_imag + Zb_imag = ${(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Ra, 2) * parameters.Ca / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2))).toFixed(3)} + ${(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Rb, 2) * parameters.Cb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)} = ${(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Ra, 2) * parameters.Ca / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) - 2 * Math.PI * frequencies[4] * Math.pow(parameters.Rb, 2) * parameters.Cb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)}

// 5. Calculate denominator for parallel combination
denom = (Rs + Zab_real)² + Zab_imag² = (${parameters.Rs.toFixed(1)} + ${(parameters.Ra / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) + parameters.Rb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)})² + (${(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Ra, 2) * parameters.Ca / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) - 2 * Math.PI * frequencies[4] * Math.pow(parameters.Rb, 2) * parameters.Cb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)})² = ${Math.pow(parameters.Rs + parameters.Ra / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) + parameters.Rb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2)), 2) + Math.pow(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Ra, 2) * parameters.Ca / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) - 2 * Math.PI * frequencies[4] * Math.pow(parameters.Rb, 2) * parameters.Cb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2)), 2)}

// 6. Calculate final impedance
Z_real = (Rs * Zab_real) / denom = ${parameters.Rs.toFixed(1)} * ${(parameters.Ra / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) + parameters.Rb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)} / ${Math.pow(parameters.Rs + parameters.Ra / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) + parameters.Rb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2)), 2) + Math.pow(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Ra, 2) * parameters.Ca / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) - 2 * Math.PI * frequencies[4] * Math.pow(parameters.Rb, 2) * parameters.Cb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2)), 2)} = ${impedanceValues[4].real.toFixed(3)}
Z_imag = (Rs * Zab_imag) / denom = ${parameters.Rs.toFixed(1)} * ${(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Ra, 2) * parameters.Ca / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) - 2 * Math.PI * frequencies[4] * Math.pow(parameters.Rb, 2) * parameters.Cb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2))).toFixed(3)} / ${Math.pow(parameters.Rs + parameters.Ra / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) + parameters.Rb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2)), 2) + Math.pow(-2 * Math.PI * frequencies[4] * Math.pow(parameters.Ra, 2) * parameters.Ca / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Ra * parameters.Ca, 2)) - 2 * Math.PI * frequencies[4] * Math.pow(parameters.Rb, 2) * parameters.Cb / (1 + Math.pow(2 * Math.PI * frequencies[4] * parameters.Rb * parameters.Cb, 2)), 2)} = ${impedanceValues[4].imaginary.toFixed(3)}

// 7. Calculate magnitude and phase
|Z| = √(Z_real² + Z_imag²) = √(${impedanceValues[4].real.toFixed(3)}² + ${impedanceValues[4].imaginary.toFixed(3)}²) = ${impedanceValues[4].magnitude.toFixed(3)}
∠Z = atan2(Z_imag, Z_real) = atan2(${impedanceValues[4].imaginary.toFixed(3)}, ${impedanceValues[4].real.toFixed(3)}) = ${impedanceValues[4].phase.toFixed(3)}°
`}
            </div>
            <div className="text-xs text-neutral-500 mt-2">This detailed step-by-step calculation helps debug the impedance calculation process for frequency = {frequencies[4].toFixed(1)} Hz.</div>
          </div>
          
          <div className="p-3 bg-neutral-700/30 rounded">
            <div className="text-xs text-neutral-400 mb-1">Frequency Response Debugging Table</div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-neutral-300 mt-2">
                <thead className="bg-neutral-900/50">
                  <tr>
                    <th className="p-2 text-left">Frequency (Hz)</th>
                    <th className="p-2 text-left">Real (Ω)</th>
                    <th className="p-2 text-left">Imag (Ω)</th>
                    <th className="p-2 text-left">|Z| (Ω)</th>
                    <th className="p-2 text-left">Phase (°)</th>
                  </tr>
                </thead>
                <tbody>
                  {impedanceValues.map((imp, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? "bg-neutral-800/30" : ""}>
                      <td className="p-2">{imp.frequency.toFixed(1)}</td>
                      <td className="p-2">{imp.real.toFixed(2)}</td>
                      <td className="p-2">{imp.imaginary.toFixed(2)}</td>
                      <td className="p-2">{imp.magnitude.toFixed(2)}</td>
                      <td className="p-2">{imp.phase.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="text-xs text-neutral-500 mt-2">This table shows impedance values across multiple frequencies, useful for identifying patterns in the underdetermined system.</div>
          </div>
          
          {referenceModel && (
            <div className="p-3 bg-neutral-700/30 rounded">
              <div className="text-xs text-neutral-400 mb-1">Reference Model Comparison</div>
              <div className="bg-neutral-900/50 p-3 rounded-md mt-2">
                <div className="text-sm font-medium text-neutral-300 mb-2">Understanding Current vs Reference Model Fit</div>
                <div className="text-xs text-neutral-400">
                  <p className="mb-2">The calculated resnorm value of <span className="font-mono font-medium text-primary">{resnorm.toFixed(3)}</span> quantifies the difference between your current parameter set and the reference model.</p>
                  
                  <div className="grid grid-cols-2 gap-3 my-3">
                    <div className="bg-neutral-800/60 p-2 rounded">
                      <div className="font-medium text-neutral-300 mb-1">Current Parameters</div>
                      <div className="grid grid-cols-2 gap-1">
                        <div className="text-neutral-500">TER:</div>
                        <div className="text-right text-white">{terValue.toFixed(1)} Ω</div>
                        <div className="text-neutral-500">TEC:</div>
                        <div className="text-right text-white">{tecValue.toFixed(3)} μF</div>
                        <div className="text-neutral-500">Ra/Rb:</div>
                        <div className="text-right text-white">{(parameters.Ra/parameters.Rb).toFixed(2)}</div>
                        <div className="text-neutral-500">Ca/Cb:</div>
                        <div className="text-right text-white">{(parameters.Ca/parameters.Cb).toFixed(2)}</div>
                      </div>
                    </div>
                    
                    <div className="bg-neutral-800/60 p-2 rounded">
                      <div className="font-medium text-neutral-300 mb-1">Reference Model</div>
                      <div className="grid grid-cols-2 gap-1">
                        <div className="text-neutral-500">TER:</div>
                        <div className="text-right text-white">{calculateTER(referenceModel.parameters).toFixed(1)} Ω</div>
                        <div className="text-neutral-500">TEC:</div>
                        <div className="text-right text-white">{calculateTEC(referenceModel.parameters).toFixed(3)} μF</div>
                        <div className="text-neutral-500">Ra/Rb:</div>
                        <div className="text-right text-white">{(referenceModel.parameters.Ra/referenceModel.parameters.Rb).toFixed(2)}</div>
                        <div className="text-neutral-500">Ca/Cb:</div>
                        <div className="text-right text-white">{(referenceModel.parameters.Ca/referenceModel.parameters.Cb).toFixed(2)}</div>
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-neutral-400">Key Insights:</p>
                  <ul className="list-disc list-inside space-y-1 mt-1 ml-2">
                    <li>
                      {resnorm < 1 ? 
                        "Your current parameters are very close to the reference model." : 
                        resnorm < 5 ? 
                        "Your current parameters produce a similar impedance profile to the reference model." :
                        "Your current parameters differ significantly from the reference model."}
                    </li>
                    <li>
                      {Math.abs(terValue - calculateTER(referenceModel.parameters)) < 1 ?
                        "TER values match closely." :
                        `TER values differ by ${Math.abs(terValue - calculateTER(referenceModel.parameters)).toFixed(1)} Ω.`}
                    </li>
                    <li>
                      {Math.abs(parameters.Ra/parameters.Rb - referenceModel.parameters.Ra/referenceModel.parameters.Rb) < 0.1 ?
                        "Ra/Rb ratios match closely." :
                        `Ra/Rb ratios differ (${(parameters.Ra/parameters.Rb).toFixed(2)} vs ${(referenceModel.parameters.Ra/referenceModel.parameters.Rb).toFixed(2)}).`}
                    </li>
                  </ul>
                </div>
              </div>
              
              <div className="mt-3">
                <div className="text-xs text-neutral-400 mb-1">Frequency Response Comparison</div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-neutral-300 mt-2">
                    <thead className="bg-neutral-900/50">
                      <tr>
                        <th className="p-2 text-left">Frequency (Hz)</th>
                        <th className="p-2 text-left">Current |Z| (Ω)</th>
                        <th className="p-2 text-left">Reference |Z| (Ω)</th>
                        <th className="p-2 text-left">Difference (Ω)</th>
                        <th className="p-2 text-left">Difference (%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {frequencies.map((freq, idx) => {
                        const currentZ = calculateImpedance(parameters, freq);
                        const referenceZ = calculateImpedance(referenceModel.parameters, freq);
                        const diff = Math.abs(currentZ.magnitude - referenceZ.magnitude);
                        const diffPercent = (diff / referenceZ.magnitude) * 100;
                        
                        return (
                          <tr key={idx} className={idx % 2 === 0 ? "bg-neutral-800/30" : ""}>
                            <td className="p-2">{freq.toFixed(1)}</td>
                            <td className="p-2">{currentZ.magnitude.toFixed(2)}</td>
                            <td className="p-2">{referenceZ.magnitude.toFixed(2)}</td>
                            <td className="p-2">{diff.toFixed(2)}</td>
                            <td className="p-2" style={{
                              color: diffPercent > 10 ? '#f87171' : 
                                    diffPercent > 5 ? '#fbbf24' : 
                                    '#34d399'
                            }}>
                              {diffPercent.toFixed(1)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="text-xs text-neutral-500 mt-2">
                  This table compares impedance magnitude between current and reference parameters across frequencies.
                  <span className="ml-1 text-red-400">Red</span> indicates large differences ({'>'}10%), 
                  <span className="ml-1 text-amber-400">yellow</span> indicates moderate differences ({'>'}5%), and
                  <span className="ml-1 text-green-400">green</span> indicates small differences ({'<'}5%).
                </div>
              </div>
            </div>
          )}
          
          <div className="p-3 bg-neutral-700/30 rounded">
            <div className="text-xs text-neutral-400 mb-1">Resnorm Mathematical Definition</div>
            <div className="font-mono font-medium text-neutral-200 bg-neutral-900/50 p-3 rounded-md tracking-wide leading-6">
              <div className="flex items-center space-x-1 flex-wrap justify-center">
                <span className="text-amber-400">Resnorm</span>
                <span>=</span>
                <span className="text-sky-400">√</span>
                <span className="text-neutral-400">(</span>
                <span className="text-green-400">1/n</span>
                <span className="relative px-1">
                  <span className="text-pink-400">∑</span>
                  <span className="absolute -bottom-1 left-0 text-pink-300 text-xs">i=1</span>
                  <span className="absolute -top-1 left-1 text-pink-300 text-xs">n</span>
                </span>
                <span className="text-neutral-400">[</span>
                <span className="text-neutral-400">|</span>
                <span className="relative pr-5">
                  <span className="text-primary">Z</span>
                  <span className="absolute -bottom-1 left-2 text-primary-light text-xs">ref</span>
                </span>
                <span className="text-neutral-400">(</span>
                <span className="relative px-1">
                  <span className="text-yellow-400">f</span>
                  <span className="absolute -bottom-1 left-1 text-yellow-300 text-xs">i</span>
                </span>
                <span className="text-neutral-400">)</span>
                <span className="text-teal-400">−</span>
                <span className="relative pr-5">
                  <span className="text-primary">Z</span>
                  <span className="absolute -bottom-1 left-2 text-primary-light text-xs">test</span>
                </span>
                <span className="text-neutral-400">(</span>
                <span className="relative px-1">
                  <span className="text-yellow-400">f</span>
                  <span className="absolute -bottom-1 left-1 text-yellow-300 text-xs">i</span>
                </span>
                <span className="text-neutral-400">)</span>
                <span className="text-neutral-400">|</span>
                <span className="relative">
                  <span className="text-neutral-300 text-xs absolute -top-1 right-0">2</span>
                </span>
                <span className="text-neutral-400">)</span>
              </div>
            </div>
            <div className="font-mono text-xs text-neutral-400 mt-3 bg-neutral-900/50 p-3 rounded-md tracking-wide leading-6 text-center">
              <div className="flex items-center flex-wrap justify-center">
                <span>=</span>
                <span className="text-sky-400 mx-1">√</span>
                <span className="text-neutral-500">(</span>
                <span className="text-green-400 mx-1">1/n</span>
                <span className="relative px-2">
                  <span className="text-pink-400">∑</span>
                  <span className="absolute -bottom-1 left-0 text-pink-300 text-xs">i=1</span>
                  <span className="absolute -top-1 left-2 text-pink-300 text-xs">n</span>
                </span>
                <span className="text-neutral-500 mx-1">[</span>
                <span className="px-4">
                  <span className="text-primary">(Re[Z</span>
                  <span className="text-neutral-300 text-xs">ref</span>
                  <span className="text-primary">-Z</span>
                  <span className="text-neutral-300 text-xs">test</span>
                  <span className="text-primary">])</span>
                  <span className="text-neutral-500">²</span>
                  <span className="text-teal-400 mx-1">+</span>
                  <span className="text-primary">(Im[Z</span>
                  <span className="text-neutral-300 text-xs">ref</span>
                  <span className="text-primary">-Z</span>
                  <span className="text-neutral-300 text-xs">test</span>
                  <span className="text-primary">])</span>
                  <span className="text-neutral-500">²</span>
                </span>
                <span className="text-neutral-500 mx-1">]</span>
                <span className="text-neutral-500">)</span>
              </div>
            </div>
            <div className="text-xs text-neutral-500 mt-2">• Sum over all frequency points (f₁...fₙ) in the specified frequency range</div>
            <div className="text-xs text-neutral-500">• Each impedance Z(f) has real (Re) and imaginary (Im) components</div>
            <div className="text-xs text-neutral-500">• Euclidean distance in complex plane normalized by number of frequency points</div>
            <div className="text-xs text-neutral-500">• Lower values indicate better match between impedance spectra</div>
          </div>
          
          <div className="p-3 bg-neutral-700/30 rounded">
            <div className="text-xs text-neutral-400 mb-1">Analyzing Underdetermined System</div>
            <div className="font-mono text-xs text-neutral-400 mt-2 bg-neutral-900/50 p-3 rounded-md">
              For a circuit with 5 parameters (Rs, Ra, Ca, Rb, Cb), multiple combinations can produce similar impedance spectra, especially when:
            </div>
            <div className="text-xs text-neutral-500 mt-2">
              • Different parameter sets can yield nearly identical impedance responses
            </div>
            <div className="text-xs text-neutral-500">
              • The problem becomes more underdetermined when using a limited frequency range
            </div>
            <div className="text-xs text-neutral-500">
              • Parameter relationships: Check ratio Ra/Rb and product Ca*Cb for patterns
            </div>
            <div className="text-xs text-neutral-500">
              • Debugging approach: Plot parameter relationships vs resnorm in spider/radar plots
            </div>
            
            <div className="mt-3 bg-neutral-900/50 p-3 rounded-md">
              <div className="text-xs text-neutral-300 font-medium">Key Parameter Relationships for Debugging</div>
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div className="font-mono text-xs text-neutral-400">
                  Ra/Rb = {(parameters.Ra/parameters.Rb).toFixed(4)}
                </div>
                <div className="font-mono text-xs text-neutral-400">
                  Ca/Cb = {(parameters.Ca/parameters.Cb).toFixed(4)}
                </div>
                <div className="font-mono text-xs text-neutral-400">
                  Ra*Ca = {(parameters.Ra*parameters.Ca).toExponential(4)}
                </div>
                <div className="font-mono text-xs text-neutral-400">
                  Rb*Cb = {(parameters.Rb*parameters.Cb).toExponential(4)}
                </div>
                <div className="font-mono text-xs text-neutral-400">
                  Rs/(Ra+Rb) = {(parameters.Rs/(parameters.Ra+parameters.Rb)).toFixed(4)}
                </div>
                <div className="font-mono text-xs text-neutral-400">
                  Ra*Ca/Rb*Cb = {((parameters.Ra*parameters.Ca)/(parameters.Rb*parameters.Cb)).toFixed(4)}
                </div>
              </div>
              <div className="text-xs text-neutral-500 mt-2">
                These parameter relationships are critical for understanding the underdetermined nature of the system. Similar impedance curves may arise when these relationships are preserved, even if individual parameters differ.
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Frequency Spectrum Analysis */}
      <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700 mt-6">
        <h4 className="text-sm font-medium text-neutral-300 mb-4 flex items-center">
          <svg className="w-4 h-4 mr-2 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Impedance Spectrum Debugging
        </h4>
        
        <div className="space-y-3">
          <div className="p-3 bg-neutral-700/30 rounded">
            <div className="text-xs text-neutral-400 mb-1">Parameter Sensitivity Analysis</div>
            <div className="font-mono text-xs text-neutral-400 mt-2">
              The sensitivity of impedance to each parameter varies with frequency:
            </div>
            
            <div className="mt-3 bg-neutral-900/50 p-3 rounded-md">
              <div className="grid grid-cols-3 gap-2">
                <div className="bg-green-500/20 text-green-400 text-xs p-2 rounded-md">
                  <div className="font-semibold mb-1">Rs Sensitivity</div>
                  <div>- Affects high-frequency impedance</div>
                  <div>- Controls right shift of Nyquist plot</div>
                  <div>- Low Rs = smaller circle radius</div>
                </div>
                <div className="bg-blue-500/20 text-blue-400 text-xs p-2 rounded-md">
                  <div className="font-semibold mb-1">Ra/Rb Sensitivity</div>
                  <div>- Affects mid-frequency region</div>
                  <div>- Ra/Rb ratio affects circle shape</div>
                  <div>- Sum (Ra+Rb) affects overall size</div>
                </div>
                <div className="bg-amber-500/20 text-amber-400 text-xs p-2 rounded-md">
                  <div className="font-semibold mb-1">Ca/Cb Sensitivity</div>
                  <div>- Affects frequency response position</div>
                  <div>- Controls peak frequencies</div>
                  <div>- Shifts impedance curve along frequency axis</div>
                </div>
              </div>
              <div className="text-xs text-neutral-400 mt-3">
                Debugging strategy: Systematically vary each parameter to observe how it affects the impedance spectrum, particularly at these diagnostic frequencies: {frequencies[0].toFixed(1)}Hz, {frequencies[4].toFixed(1)}Hz, and {frequencies[7].toFixed(1)}Hz.
              </div>
            </div>
          </div>
          
          <div className="p-3 bg-neutral-700/30 rounded">
            <div className="text-xs text-neutral-400 mb-1">Diagnostics for Underdetermined Systems</div>
            <div className="font-mono text-xs text-neutral-400 mt-2">
              When multiple parameter combinations produce similar impedance spectra, check:
            </div>
            <div className="grid grid-cols-1 gap-2 mt-2">
              <div className="p-2 bg-neutral-900/50 rounded-md">
                <div className="font-medium text-neutral-300">1. Parameter Coupling</div>
                <div className="text-neutral-400 text-xs mt-1">
                  Rs and (Ra+Rb) are coupled - check TER value: {terValue.toFixed(2)} Ω
                </div>
                <div className="text-neutral-400 text-xs">
                  Ca and Cb are coupled - check TEC value: {(tecValue * 1e6).toFixed(2)} μF
                </div>
              </div>
              <div className="p-2 bg-neutral-900/50 rounded-md">
                <div className="font-medium text-neutral-300">2. Equivalent Circuit Invariants</div>
                <div className="text-neutral-400 text-xs mt-1">
                  Product Ra*Ca = {(parameters.Ra*parameters.Ca).toExponential(4)} should be similar to Rb*Cb = {(parameters.Rb*parameters.Cb).toExponential(4)}
                </div>
                <div className="text-neutral-400 text-xs">
                  Check if parallel impedance (Ra||Rb) is consistent: {(parameters.Ra * parameters.Rb / (parameters.Ra + parameters.Rb)).toFixed(2)} Ω
                </div>
              </div>
              <div className="p-2 bg-neutral-900/50 rounded-md">
                <div className="font-medium text-neutral-300">3. Characteristic Impedance Features</div>
                <div className="text-neutral-400 text-xs mt-1">
                  Low frequency limit: {impedanceValues[0].real.toFixed(2)} Ω ≈ {(parameters.Rs + parameters.Ra * parameters.Rb / (parameters.Ra + parameters.Rb)).toFixed(2)} Ω
                </div>
                <div className="text-neutral-400 text-xs">
                  High frequency impedance: {impedanceValues[7].real.toFixed(2)} Ω ≈ {parameters.Rs.toFixed(2)} Ω
                </div>
                <div className="text-neutral-400 text-xs">
                  Impedance phase min: {Math.min(...impedanceValues.map(v => v.phase)).toFixed(2)}° at characteristic frequency
                </div>
              </div>
            </div>
          </div>
          
          <div className="p-3 bg-neutral-700/30 rounded">
            <div className="text-xs text-neutral-400 mb-1">Debugging Visualizations</div>
            <div className="text-xs text-neutral-500 mt-2">
              <ol className="list-decimal list-inside space-y-1">
                <li>Nyquist plot (Re vs -Im) reveals underdetermined parameter regions</li>
                <li>Spider/radar plots of key parameters help identify equivalent solutions</li>
                <li>Frequency vs |Z| log-log plots highlight different parameter sensitivities</li>
                <li>Parameter dependency heatmaps reveal coupling relationships</li>
              </ol>
            </div>
            <div className="text-xs text-neutral-500 mt-2">
              Key breakpoints in impedance spectra:
            </div>
            <div className="grid grid-cols-3 gap-2 mt-2">
              {[0, 3, 7].map((idx) => (
                <div key={idx} className={`p-2 rounded-md ${idx === 0 ? 'bg-green-500/20 text-green-400' : idx === 3 ? 'bg-blue-500/20 text-blue-400' : 'bg-amber-500/20 text-amber-400'}`}>
                  <div>{impedanceValues[idx].frequency.toFixed(1)} Hz:</div>
                  <div>Z = {impedanceValues[idx].real.toFixed(2)} - j{Math.abs(impedanceValues[idx].imaginary).toFixed(2)} Ω</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ResnormDetails component to display detailed resnorm calculations */}
      <ResnormDetails resnorm={resnorm} />
    </div>
  );
};

// ResnormDetails component to display detailed resnorm calculations
const ResnormDetails: React.FC<{ resnorm: number }> = ({ resnorm }) => {
  const details = window.resnormDetails || [];
  const summary = window.resnormSummary || { 
    sumOfSquaredResiduals: 0, 
    sumWeights: 0,
    finalResnorm: resnorm
  };

  // Select a few points to display (start, middle, end)
  const displayPoints: ResnormDetail[] = [];
  if (details.length > 0) {
    // Add lowest frequency point
    displayPoints.push(details[0]);
    
    // Add a middle point if available
    if (details.length > 2) {
      displayPoints.push(details[Math.floor(details.length / 2)]);
    }
    
    // Add highest frequency point
    displayPoints.push(details[details.length - 1]);
  }

  return (
    <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700 mt-6">
      <h4 className="text-sm font-medium text-neutral-300 mb-4 flex items-center">
        <svg className="w-4 h-4 mr-2 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
        Resnorm Calculation Details
      </h4>

      <div className="space-y-4">
        <div className="p-4 bg-neutral-700/30 rounded">
          <div className="text-sm text-neutral-400 mb-2">Resnorm Formula (Weighted RMSE)</div>
          <div className="font-mono text-xs text-neutral-300 bg-neutral-800/60 p-3 rounded">
            <p>Resnorm = √(Σ(w·r²) / Σw)</p>
            <p className="mt-1">where:</p>
            <ul className="list-disc pl-5 mt-1 space-y-1">
              <li>r = normalized residuals at each frequency</li>
              <li>w = weight factor (1/log10(f)) to emphasize lower frequencies</li>
            </ul>
          </div>
        </div>

        <div className="p-4 bg-neutral-700/30 rounded">
          <div className="text-sm text-neutral-400 mb-2">Sample Calculations</div>
          
          {displayPoints.map((point, idx) => (
            <div key={idx} className="mb-4 last:mb-0">
              <div className="text-xs font-medium text-neutral-300 mb-1">
                Frequency: {point.frequency.toFixed(2)} Hz
              </div>
              <div className="font-mono text-xs text-neutral-300 bg-neutral-800/60 p-3 rounded">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p>Test Z = {point.z1.real.toExponential(4)} + j{point.z1.imaginary.toExponential(4)} Ω</p>
                    <p>|Test Z| = {point.z1.magnitude.toExponential(4)} Ω</p>
                  </div>
                  <div>
                    <p>Ref Z = {point.z2.real.toExponential(4)} + j{point.z2.imaginary.toExponential(4)} Ω</p>
                    <p>|Ref Z| = {point.z2.magnitude.toExponential(4)} Ω</p>
                  </div>
                </div>
                <div className="mt-2">
                  <p>Normalized Residuals:</p>
                  <p>Real: {point.residuals.real.toExponential(4)}</p>
                  <p>Imaginary: {point.residuals.imaginary.toExponential(4)}</p>
                  <p>Combined: {point.residuals.combined.toExponential(4)}</p>
                </div>
                <div className="mt-2">
                  <p>Weight: {point.weight.toFixed(4)}</p>
                  <p>Weighted squared residual: {point.weightedResidual.toExponential(4)}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="p-4 bg-neutral-700/30 rounded">
          <div className="text-sm text-neutral-400 mb-2">Final Calculation</div>
          <div className="font-mono text-xs text-neutral-300 bg-neutral-800/60 p-3 rounded">
            <p>Sum of weighted squared residuals: {summary.sumOfSquaredResiduals.toExponential(4)}</p>
            <p>Sum of weights: {summary.sumWeights.toFixed(4)}</p>
            <p>Final resnorm = √({summary.sumOfSquaredResiduals.toExponential(4)} / {summary.sumWeights.toFixed(4)}) = {summary.finalResnorm.toExponential(4)}</p>
          </div>
          <div className="mt-3 text-xs text-neutral-500 italic">
            Industry-standard resnorm calculation normalized by reference impedance magnitude at each frequency.
            This provides a physically meaningful measure of the goodness of fit between impedance spectra.
          </div>
        </div>
      </div>
    </div>
  );
};

export default MathDetailsTab; 
import React from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import { CircuitParameters } from './types/parameters';
import { ModelSnapshot } from './utils/types';

interface MathDetailsTabProps {
  parameters: CircuitParameters;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  referenceModel: ModelSnapshot | null;
}

// Helper functions
function calculateTER(params: CircuitParameters): number {
  return (params.Rs * (params.Ra + params.Rb)) / (params.Rs + params.Ra + params.Rb);
}

function calculateTEC(params: CircuitParameters): number {
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
  
  // Calculate sum of membrane impedances
  const Zab_real = Za_real + Zb_real;
  const Zab_imag = Za_imag + Zb_imag;
  
  // Calculate parallel combination: Z_total = (Rs * (Za + Zb)) / (Rs + Za + Zb)
  const num_real = params.Rs * Zab_real;
  const num_imag = params.Rs * Zab_imag;
  
  const denom_real = params.Rs + Zab_real;
  const denom_imag = Zab_imag;
  
  const denom_mag_squared = denom_real * denom_real + denom_imag * denom_imag;
  
  const real = (num_real * denom_real + num_imag * denom_imag) / denom_mag_squared;
  const imaginary = (num_imag * denom_real - num_real * denom_imag) / denom_mag_squared;
  
  const magnitude = Math.sqrt(Math.pow(real, 2) + Math.pow(imaginary, 2));
  const phase = Math.atan2(imaginary, real) * (180 / Math.PI);
  
  return { real, imaginary, magnitude, phase };
}

// Calculate resnorm between two parameter sets
function calculateResnorm(params1: CircuitParameters, params2: CircuitParameters, frequencies: number[]): number {
  let sumOfSquaredResiduals = 0;
  const n = frequencies.length;
  
  for (const freq of frequencies) {
    const z1 = calculateImpedance(params1, freq);
    const z2 = calculateImpedance(params2, freq);
    
    const realResidual = z1.real - z2.real;
    const imagResidual = z1.imaginary - z2.imaginary;
    const residualSquared = realResidual * realResidual + imagResidual * imagResidual;
    
    sumOfSquaredResiduals += residualSquared;
  }
  
  return (1 / n) * Math.sqrt(sumOfSquaredResiduals);
}

export const MathDetailsTab: React.FC<MathDetailsTabProps> = ({ 
  parameters, 
  minFreq, 
  maxFreq, 
  numPoints,
  referenceModel
}) => {
  // Calculate derived values
  const terValue = calculateTER(parameters);
  const tecValue = calculateTEC(parameters);
  
  // Calculate resnorm if reference model exists
  const frequencies = Array.from({ length: 8 }, (_, i) => 
    minFreq * Math.pow(maxFreq / minFreq, i / 7)
  );
  const resnorm = referenceModel ? calculateResnorm(parameters, referenceModel.parameters, frequencies) : 0;

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 p-4 space-y-6 min-h-0">
        {/* Frequency Range Information */}
        <div className="bg-primary/10 border border-primary/20 rounded-lg p-4">
          <h4 className="text-sm font-medium text-primary mb-2">Frequency Analysis Range</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="bg-primary/5 p-3 rounded">
              <div className="text-xs text-neutral-400">Min Frequency</div>
              <div className="font-mono font-medium text-primary">
                {minFreq < 1000 ? `${minFreq.toFixed(1)} Hz` : `${(minFreq/1000).toFixed(1)} kHz`}
              </div>
            </div>
            <div className="bg-primary/5 p-3 rounded">
              <div className="text-xs text-neutral-400">Max Frequency</div>
              <div className="font-mono font-medium text-primary">
                {maxFreq < 1000 ? `${maxFreq.toFixed(1)} Hz` : `${(maxFreq/1000).toFixed(1)} kHz`}
              </div>
            </div>
            <div className="bg-primary/5 p-3 rounded">
              <div className="text-xs text-neutral-400">Data Points</div>
              <div className="font-mono font-medium text-primary">{numPoints}</div>
            </div>
          </div>
        </div>
        
        {/* Current Parameters and Results */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Parameters */}
          <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
            <h4 className="text-sm font-medium text-neutral-300 mb-4">Current Parameters</h4>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-neutral-700/30 rounded">
                  <div className="text-xs text-neutral-400"><InlineMath>{`R_s`}</InlineMath> (Solution)</div>
                  <div className="font-mono font-medium text-neutral-200">{parameters.Rs.toFixed(1)} Ω</div>
                </div>
                <div className="p-3 bg-neutral-700/30 rounded">
                  <div className="text-xs text-neutral-400"><InlineMath>{`R_a`}</InlineMath> (Apical)</div>
                  <div className="font-mono font-medium text-neutral-200">{parameters.Ra.toFixed(0)} Ω</div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-neutral-700/30 rounded">
                  <div className="text-xs text-neutral-400"><InlineMath>{`R_b`}</InlineMath> (Basal)</div>
                  <div className="font-mono font-medium text-neutral-200">{parameters.Rb.toFixed(0)} Ω</div>
                </div>
                <div className="p-3 bg-neutral-700/30 rounded">
                  <div className="text-xs text-neutral-400"><InlineMath>{`C_a`}</InlineMath> (Apical)</div>
                  <div className="font-mono font-medium text-neutral-200">{(parameters.Ca * 1e6).toFixed(2)} μF</div>
                </div>
              </div>
              <div className="p-3 bg-neutral-700/30 rounded">
                <div className="text-xs text-neutral-400"><InlineMath>{`C_b`}</InlineMath> (Basal)</div>
                <div className="font-mono font-medium text-neutral-200">{(parameters.Cb * 1e6).toFixed(2)} μF</div>
              </div>
            </div>
          </div>
          
          {/* Calculated Values */}
          <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
            <h4 className="text-sm font-medium text-neutral-300 mb-4">Calculated Values</h4>
            <div className="space-y-3">
              <div className="p-3 bg-neutral-700/30 rounded">
                <div className="text-xs text-neutral-400">Transepithelial Resistance (TER)</div>
                <div className="font-mono font-medium text-xl text-neutral-200">{terValue.toFixed(1)} Ω</div>
                {referenceModel && (
                  <div className="text-xs text-neutral-500 mt-1">
                    Reference: {calculateTER(referenceModel.parameters).toFixed(1)} Ω
                  </div>
                )}
              </div>
              
              <div className="p-3 bg-neutral-700/30 rounded">
                <div className="text-xs text-neutral-400">Transepithelial Capacitance (TEC)</div>
                <div className="font-mono font-medium text-xl text-neutral-200">{(tecValue * 1e6).toFixed(2)} μF</div>
                {referenceModel && (
                  <div className="text-xs text-neutral-500 mt-1">
                    Reference: {(calculateTEC(referenceModel.parameters) * 1e6).toFixed(2)} μF
                  </div>
                )}
              </div>
              
              {referenceModel && (
                <div className="p-3 bg-neutral-700/30 rounded">
                  <div className="text-xs text-neutral-400">Resnorm vs Reference</div>
                  <div className="font-mono font-medium text-xl text-neutral-200">{resnorm.toExponential(3)}</div>
                  <div className={`text-xs mt-1 ${
                    resnorm < 0.05 ? 'text-green-400' : 
                    resnorm < 0.15 ? 'text-blue-400' : 
                    resnorm < 0.3 ? 'text-amber-400' : 'text-red-400'
                  }`}>
                    {resnorm < 0.05 ? 'Excellent match' : 
                     resnorm < 0.15 ? 'Good match' : 
                     resnorm < 0.3 ? 'Moderate match' : 'Poor match'}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Mathematical Equations */}
        <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
          <h4 className="text-sm font-medium text-neutral-300 mb-4">Mathematical Formulations</h4>
          
          <div className="space-y-6">
            {/* Circuit Model */}
            <div className="p-4 bg-neutral-700/20 rounded-lg">
              <h5 className="text-sm font-medium text-neutral-200 mb-4">Epithelial Circuit Model</h5>
              <div className="space-y-6">
                {/* Total Impedance */}
                <div className="bg-neutral-900/50 p-4 rounded">
                  <div className="text-xs text-neutral-400 mb-3 text-center">Total Circuit Impedance</div>
                  <div className="text-center text-neutral-200">
                    <BlockMath>{`Z_{\\text{total}}(\\omega) = \\frac{R_s \\cdot [Z_a(\\omega) + Z_b(\\omega)]}{R_s + [Z_a(\\omega) + Z_b(\\omega)]}`}</BlockMath>
                  </div>
                  <div className="text-xs text-neutral-500 mt-3 text-center">
                    Parallel combination of solution resistance with membrane impedances
                  </div>
                </div>
                
                {/* Individual Membrane Impedances */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-neutral-900/50 p-4 rounded">
                    <div className="text-xs text-neutral-400 mb-3 text-center">Apical Membrane</div>
                    <div className="text-center text-neutral-200">
                      <BlockMath>{`Z_a(\\omega) = \\frac{R_a}{1 + j\\omega R_a C_a}`}</BlockMath>
                    </div>
                  </div>
                  
                  <div className="bg-neutral-900/50 p-4 rounded">
                    <div className="text-xs text-neutral-400 mb-3 text-center">Basal Membrane</div>
                    <div className="text-center text-neutral-200">
                      <BlockMath>{`Z_b(\\omega) = \\frac{R_b}{1 + j\\omega R_b C_b}`}</BlockMath>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Derived Parameters */}
            <div className="p-4 bg-neutral-700/20 rounded-lg">
              <h5 className="text-sm font-medium text-neutral-200 mb-4">Transepithelial Parameters</h5>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-neutral-900/50 p-4 rounded">
                  <div className="text-xs text-neutral-400 mb-3 text-center">Transepithelial Resistance</div>
                  <div className="text-center text-neutral-200">
                    <BlockMath>{`\\text{TER} = \\frac{R_s \\cdot (R_a + R_b)}{R_s + (R_a + R_b)}`}</BlockMath>
                  </div>
                  <div className="text-xs text-neutral-500 mt-3 text-center">DC resistance (Ω)</div>
                </div>
                
                <div className="bg-neutral-900/50 p-4 rounded">
                  <div className="text-xs text-neutral-400 mb-3 text-center">Transepithelial Capacitance</div>
                  <div className="text-center text-neutral-200">
                    <BlockMath>{`\\text{TEC} = \\frac{C_a \\cdot C_b}{C_a + C_b}`}</BlockMath>
                  </div>
                  <div className="text-xs text-neutral-500 mt-3 text-center">Total capacitance (F)</div>
                </div>
              </div>
            </div>
            
            {/* Resnorm Definition */}
            {referenceModel && (
              <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                <h5 className="text-sm font-medium text-amber-200 mb-4">Resnorm Calculation</h5>
                <div className="bg-neutral-900/50 p-4 rounded">
                  <div className="text-xs text-neutral-400 mb-3 text-center">Root Mean Square Error</div>
                  <div className="text-center text-neutral-200">
                    <BlockMath>{`\\text{Resnorm} = \\frac{1}{n} \\sqrt{\\sum_{i=1}^{n} |Z_{\\text{ref}}(f_i) - Z_{\\text{test}}(f_i)|^2}`}</BlockMath>
                  </div>
                  <div className="text-xs text-neutral-500 mt-3 space-y-1">
                    <div className="text-center">where <InlineMath>{`n = ${frequencies.length}`}</InlineMath> frequency points</div>
                    <div className="text-center"><InlineMath>{`|Z|^2 = (\\text{Re}[Z])^2 + (\\text{Im}[Z])^2`}</InlineMath></div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Complex Impedance Expansions */}
            <div className="p-4 bg-neutral-700/20 rounded-lg">
              <h5 className="text-sm font-medium text-neutral-200 mb-4">Complex Impedance Components</h5>
              <div className="space-y-4">
                {/* Real and Imaginary Parts */}
                <div className="bg-neutral-900/50 p-4 rounded">
                  <div className="text-xs text-neutral-400 mb-3 text-center">Real and Imaginary Components</div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="text-center text-neutral-200">
                      <div className="text-xs text-neutral-400 mb-2">Real Part</div>
                      <BlockMath>{`\\text{Re}[Z_a] = \\frac{R_a}{1 + (\\omega R_a C_a)^2}`}</BlockMath>
                    </div>
                    <div className="text-center text-neutral-200">
                      <div className="text-xs text-neutral-400 mb-2">Imaginary Part</div>
                      <BlockMath>{`\\text{Im}[Z_a] = \\frac{-\\omega R_a^2 C_a}{1 + (\\omega R_a C_a)^2}`}</BlockMath>
                    </div>
                  </div>
                </div>
                
                {/* Magnitude and Phase */}
                <div className="bg-neutral-900/50 p-4 rounded">
                  <div className="text-xs text-neutral-400 mb-3 text-center">Magnitude and Phase</div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="text-center text-neutral-200">
                      <div className="text-xs text-neutral-400 mb-2">Magnitude</div>
                      <BlockMath>{`|Z| = \\sqrt{(\\text{Re}[Z])^2 + (\\text{Im}[Z])^2}`}</BlockMath>
                    </div>
                    <div className="text-center text-neutral-200">
                      <div className="text-xs text-neutral-400 mb-2">Phase</div>
                      <BlockMath>{`\\phi = \\arctan\\left(\\frac{\\text{Im}[Z]}{\\text{Re}[Z]}\\right)`}</BlockMath>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Angular Frequency */}
            <div className="p-4 bg-neutral-700/20 rounded-lg">
              <h5 className="text-sm font-medium text-neutral-200 mb-4">Angular Frequency</h5>
              <div className="bg-neutral-900/50 p-4 rounded text-center">
                <div className="text-neutral-200">
                  <BlockMath>{`\\omega = 2\\pi f`}</BlockMath>
                </div>
                <div className="text-xs text-neutral-500 mt-2">
                  where <InlineMath>{`f`}</InlineMath> is frequency in Hz, <InlineMath>{`\\omega`}</InlineMath> is angular frequency in rad/s
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Units and Notes */}
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
          <h5 className="text-sm font-medium text-blue-300 mb-2">Units and Conventions</h5>
          <div className="text-xs text-blue-200 space-y-1">
            <div>• <InlineMath>{`R_s`}</InlineMath>: Solution resistance in Ohms (Ω)</div>
            <div>• <InlineMath>{`R_a, R_b`}</InlineMath>: Membrane resistances in Ohms (Ω)</div>
            <div>• <InlineMath>{`C_a, C_b`}</InlineMath>: Membrane capacitances in Farads (F), displayed as microfarads (μF)</div>
            <div>• <InlineMath>{`f`}</InlineMath>: Frequency in Hertz (Hz)</div>
            <div>• <InlineMath>{`\\omega`}</InlineMath>: Angular frequency in radians per second (rad/s)</div>
            <div>• <InlineMath>{`j`}</InlineMath>: Imaginary unit, <InlineMath>{`j = \\sqrt{-1}`}</InlineMath></div>
          </div>
        </div>
      </div>
    </div>
  );
}; 
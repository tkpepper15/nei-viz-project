import React from 'react';
import { InlineMath } from 'react-katex';
import { CircuitParameters } from '../types/parameters';

interface ParameterDisplaySectionProps {
  parameters: CircuitParameters;
  minFreq: number;
  maxFreq: number;
  ter: number;
  tauA: number;
  tauB: number;
}

export const ParameterDisplaySection: React.FC<ParameterDisplaySectionProps> = ({
  parameters,
  minFreq,
  maxFreq,
  ter,
  tauA,
  tauB
}) => {
  return (
    <>
      {/* Current Parameters */}
      <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
        <h3 className="text-lg font-medium text-neutral-200 mb-4">Current Parameters</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <div className="bg-neutral-900/50 p-3 rounded">
            <div className="text-neutral-400 text-xs mb-1">Shunt Resistance</div>
            <div className="font-mono text-neutral-200">{parameters.Rsh.toFixed(2)} Ω</div>
          </div>
          
          <div className="bg-neutral-900/50 p-3 rounded">
            <div className="text-neutral-400 text-xs mb-1">Apical Resistance</div>
            <div className="font-mono text-neutral-200">{parameters.Ra.toFixed(2)} Ω</div>
          </div>
          
          <div className="bg-neutral-900/50 p-3 rounded">
            <div className="text-neutral-400 text-xs mb-1">Apical Capacitance</div>
            <div className="font-mono text-neutral-200">{(parameters.Ca * 1e6).toFixed(2)} μF</div>
          </div>
          
          <div className="bg-neutral-900/50 p-3 rounded">
            <div className="text-neutral-400 text-xs mb-1">Basal Resistance</div>
            <div className="font-mono text-neutral-200">{parameters.Rb.toFixed(2)} Ω</div>
          </div>
          
          <div className="bg-neutral-900/50 p-3 rounded">
            <div className="text-neutral-400 text-xs mb-1">Basal Capacitance</div>
            <div className="font-mono text-neutral-200">{(parameters.Cb * 1e6).toFixed(2)} μF</div>
          </div>
          
          <div className="bg-neutral-900/50 p-3 rounded">
            <div className="text-neutral-400 text-xs mb-1">Frequency Range</div>
            <div className="font-mono text-neutral-200">{minFreq} - {maxFreq} Hz</div>
          </div>
        </div>
      </div>

      {/* Derived Values */}
      <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
        <h3 className="text-lg font-medium text-neutral-200 mb-4">Derived Values</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-green-900/20 p-4 rounded border border-green-700/30">
            <div className="text-green-200 text-sm font-medium mb-2">TER (DC Resistance)</div>
            <div className="font-mono text-green-300 text-lg">{ter.toFixed(2)} Ω</div>
            <div className="text-xs text-green-400 mt-1">
              <InlineMath>{`\\frac{R_{sh} \\cdot (R_a + R_b)}{R_{sh} + (R_a + R_b)}`}</InlineMath>
            </div>
          </div>
          
          <div className="bg-blue-900/20 p-4 rounded border border-blue-700/30">
            <div className="text-blue-200 text-sm font-medium mb-2">Apical Time Constant</div>
            <div className="font-mono text-blue-300 text-lg">{(tauA * 1000).toFixed(2)} ms</div>
            <div className="text-xs text-blue-400 mt-1">
              <InlineMath>{`\\tau_a = R_a \\cdot C_a`}</InlineMath>
            </div>
          </div>
          
          <div className="bg-purple-900/20 p-4 rounded border border-purple-700/30">
            <div className="text-purple-200 text-sm font-medium mb-2">Basal Time Constant</div>
            <div className="font-mono text-purple-300 text-lg">{(tauB * 1000).toFixed(2)} ms</div>
            <div className="text-xs text-purple-400 mt-1">
              <InlineMath>{`\\tau_b = R_b \\cdot C_b`}</InlineMath>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};
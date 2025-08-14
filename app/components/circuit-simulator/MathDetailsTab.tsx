import React from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import { CircuitParameters } from './types/parameters';
import { ModelSnapshot } from './types';

interface MathDetailsTabProps {
  parameters: CircuitParameters;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  referenceModel: ModelSnapshot | null;
}

// Helper functions for calculations
function calculateTER(params: CircuitParameters): number {
  return (params.Rsh * (params.Ra + params.Rb)) / (params.Rsh + params.Ra + params.Rb);
}

function calculateTimeConstants(params: CircuitParameters): { tauA: number; tauB: number } {
  return {
    tauA: params.Ra * params.Ca,
    tauB: params.Rb * params.Cb
  };
}

export const MathDetailsTab: React.FC<MathDetailsTabProps> = ({
  parameters,
  minFreq,
  maxFreq,
  referenceModel
}) => {
  const ter = calculateTER(parameters);
  const { tauA, tauB } = calculateTimeConstants(parameters);

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-neutral-200 mb-2">Mathematical Model</h2>
        <p className="text-neutral-400">Core circuit equations and parameter calculations</p>
      </div>

      {/* Quick Reference */}
      <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
        <div className="flex items-center mb-3">
          <svg className="w-4 h-4 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm font-medium text-blue-200">💡 For detailed step-by-step calculations</span>
        </div>
        <p className="text-sm text-blue-300/80">
          Visit the <strong>Data Table</strong> tab, select a frequency, and click &quot;Show Math&quot; on any parameter set to see complete impedance calculations with intermediate steps.
        </p>
      </div>

      {/* Circuit Diagram */}
      <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
        <h3 className="text-lg font-medium text-neutral-200 mb-4">Circuit Model</h3>
        <div className="bg-neutral-900/50 p-4 rounded text-center">
          <div className="text-xs text-neutral-400 mb-3">Circuit Topology</div>
          <div className="font-mono text-neutral-300 text-sm space-y-1">
            <div>       Rₛₕ (Shunt Resistance)</div>
            <div>   ────[Rₛₕ]────┬──────────┬──────</div>
            <div>                 │          │</div>
            <div>             [Rₐ]│      [Rᵦ]│</div>
            <div>                 │          │</div>
            <div>             [Cₐ]│      [Cᵦ]│</div>
            <div>                 │          │</div>
            <div>                 └──────────┘</div>
          </div>
          <div className="text-xs text-neutral-500 mt-3">
            Apical (Rₐ‖Cₐ) and Basal (Rᵦ‖Cᵦ) membranes in series, parallel to shunt resistance
          </div>
        </div>
      </div>

      {/* Core Equations */}
      <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
        <h3 className="text-lg font-medium text-neutral-200 mb-4">Core Equations</h3>
        
        {/* Master Equation */}
        <div className="p-4 bg-neutral-700/20 rounded-lg mb-4">
          <h4 className="text-sm font-medium text-neutral-200 mb-3">Total Impedance</h4>
          <div className="bg-neutral-900/50 p-4 rounded">
            <div className="text-center text-neutral-200">
              <BlockMath>{`Z_{\\text{total}}(\\omega) = \\frac{R_{\\text{sh}} \\cdot [Z_a(\\omega) + Z_b(\\omega)]}{R_{\\text{sh}} + [Z_a(\\omega) + Z_b(\\omega)]}`}</BlockMath>
            </div>
            <div className="text-xs text-neutral-500 mt-3 text-center">
              Parallel combination: Rₛₕ || (Za + Zb)
            </div>
          </div>
        </div>
        
        {/* Component Impedances */}
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

      {/* Angular Frequency */}
      <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
        <h3 className="text-lg font-medium text-neutral-200 mb-4">Angular Frequency</h3>
        <div className="bg-neutral-900/50 p-4 rounded text-center">
          <div className="text-neutral-200">
            <BlockMath>{`\\omega = 2\\pi f`}</BlockMath>
          </div>
          <div className="text-xs text-neutral-500 mt-2">
            Where f is frequency in Hz, ω is angular frequency in rad/s
          </div>
        </div>
      </div>

      {/* Reference Model Comparison */}
      {referenceModel && (
        <div className="bg-yellow-900/20 border border-yellow-700/30 rounded-lg p-4">
          <h3 className="text-lg font-medium text-yellow-200 mb-4">Reference Model Comparison</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
            <div>
              <div className="text-yellow-300 text-xs mb-1">Rsh</div>
              <div className="font-mono text-yellow-200">Ref: {referenceModel.parameters.Rsh.toFixed(2)}</div>
              <div className="font-mono text-neutral-300">Cur: {parameters.Rsh.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-yellow-300 text-xs mb-1">Ra</div>
              <div className="font-mono text-yellow-200">Ref: {referenceModel.parameters.Ra.toFixed(2)}</div>
              <div className="font-mono text-neutral-300">Cur: {parameters.Ra.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-yellow-300 text-xs mb-1">Ca (μF)</div>
              <div className="font-mono text-yellow-200">Ref: {(referenceModel.parameters.Ca * 1e6).toFixed(2)}</div>
              <div className="font-mono text-neutral-300">Cur: {(parameters.Ca * 1e6).toFixed(2)}</div>
            </div>
            <div>
              <div className="text-yellow-300 text-xs mb-1">Rb</div>
              <div className="font-mono text-yellow-200">Ref: {referenceModel.parameters.Rb.toFixed(2)}</div>
              <div className="font-mono text-neutral-300">Cur: {parameters.Rb.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-yellow-300 text-xs mb-1">Cb (μF)</div>
              <div className="font-mono text-yellow-200">Ref: {(referenceModel.parameters.Cb * 1e6).toFixed(2)}</div>
              <div className="font-mono text-neutral-300">Cur: {(parameters.Cb * 1e6).toFixed(2)}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
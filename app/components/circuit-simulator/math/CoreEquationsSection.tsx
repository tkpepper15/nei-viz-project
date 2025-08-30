import React from 'react';
import { BlockMath } from 'react-katex';

export const CoreEquationsSection: React.FC = () => {
  return (
    <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
      <h3 className="text-lg font-medium text-neutral-200 mb-4">Core Equations</h3>
      
      {/* Total Impedance Formula */}
      <div className="p-4 bg-neutral-700/20 rounded-lg mb-4">
        <h4 className="text-sm font-medium text-neutral-200 mb-3">Total Impedance Calculation</h4>
        <div className="bg-neutral-900/50 p-4 rounded">
          <div className="text-center text-neutral-200">
            <BlockMath>{`Z_{\\text{total}} = \\frac{R_{\\text{sh}} \\cdot (Z_a + Z_b)}{R_{\\text{sh}} + (Z_a + Z_b)}`}</BlockMath>
          </div>
          <div className="text-xs text-neutral-500 mt-3 text-center">
            Parallel combination of shunt resistance with series membranes
          </div>
        </div>
      </div>
      
      {/* Individual Membrane Impedances */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-neutral-900/50 p-4 rounded">
          <div className="text-xs text-neutral-400 mb-3 text-center">Apical Membrane</div>
          <div className="text-center text-neutral-200">
            <BlockMath>{`Z_a = \\frac{R_a}{1 + j\\omega R_a C_a}`}</BlockMath>
          </div>
          <div className="text-xs text-neutral-500 mt-2 text-center">
            RC parallel combination
          </div>
        </div>
        
        <div className="bg-neutral-900/50 p-4 rounded">
          <div className="text-xs text-neutral-400 mb-3 text-center">Basal Membrane</div>
          <div className="text-center text-neutral-200">
            <BlockMath>{`Z_b = \\frac{R_b}{1 + j\\omega R_b C_b}`}</BlockMath>
          </div>
          <div className="text-xs text-neutral-500 mt-2 text-center">
            RC parallel combination
          </div>
        </div>
      </div>

      {/* Implementation Details */}
      <div className="mt-4 p-4 bg-blue-900/10 rounded-lg border border-blue-700/20">
        <h4 className="text-sm font-medium text-blue-200 mb-3">Implementation Details</h4>
        <div className="space-y-3 text-sm">
          <div className="bg-neutral-900/50 p-3 rounded">
            <div className="text-xs text-neutral-400 mb-2">Complex impedance expansion:</div>
            <div className="text-center text-neutral-200">
              <BlockMath>{`Z = \\frac{R \\cdot (1 - j\\omega RC)}{1 + (\\omega RC)^2}`}</BlockMath>
            </div>
            <div className="text-xs text-neutral-500 mt-1 text-center">
              Real and imaginary parts separated for numerical computation
            </div>
          </div>
          <div className="bg-neutral-900/50 p-3 rounded">
            <div className="text-xs text-neutral-400 mb-2">Complex division formula:</div>
            <div className="text-center text-neutral-200">
              <BlockMath>{`\\frac{a + jb}{c + jd} = \\frac{(ac + bd) + j(bc - ad)}{c^2 + d^2}`}</BlockMath>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
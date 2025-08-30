import React from 'react';
import { BlockMath } from 'react-katex';

export const ComplexImpedanceSection: React.FC = () => {
  return (
    <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
      <h3 className="text-lg font-medium text-neutral-200 mb-4">Complex Impedance Derivation</h3>
      
      {/* General Form */}
      <div className="p-4 bg-neutral-900/50 rounded-lg mb-4">
        <div className="text-center text-neutral-200">
          <BlockMath>{`Z(\\omega) = Z_{\\text{real}}(\\omega) + j \\cdot Z_{\\text{imag}}(\\omega)`}</BlockMath>
        </div>
        <div className="text-xs text-neutral-400 mt-2 text-center">
          Where ω = 2πf and j is the imaginary unit
        </div>
      </div>

      {/* Component Expansions */}
      <div className="space-y-3">
        <div className="bg-neutral-900/50 p-3 rounded">
          <div className="text-xs text-neutral-400 mb-2">Apical Membrane Impedance</div>
          <div className="text-center text-neutral-200 text-sm">
            <BlockMath>{`Z_a = \\frac{R_a}{1 + j\\omega R_a C_a} = \\frac{R_a(1 - j\\omega R_a C_a)}{1 + (\\omega R_a C_a)^2}`}</BlockMath>
          </div>
        </div>
        <div className="bg-neutral-900/50 p-3 rounded">
          <div className="text-xs text-neutral-400 mb-2">Basal Membrane Impedance</div>
          <div className="text-center text-neutral-200 text-sm">
            <BlockMath>{`Z_b = \\frac{R_b}{1 + j\\omega R_b C_b} = \\frac{R_b(1 - j\\omega R_b C_b)}{1 + (\\omega R_b C_b)^2}`}</BlockMath>
          </div>
        </div>
      </div>
    </div>
  );
};
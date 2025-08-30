import React from 'react';
import { BlockMath } from 'react-katex';

export const NyquistPlotSection: React.FC = () => {
  return (
    <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
      <h3 className="text-lg font-medium text-neutral-200 mb-4">Nyquist Plot Analysis</h3>
      
      <div className="p-4 bg-neutral-900/50 rounded-lg">
        <div className="text-center text-neutral-200 mb-3">
          <BlockMath>{`Z(\\omega) = Z_{\\text{real}}(\\omega) + j \\cdot Z_{\\text{imag}}(\\omega)`}</BlockMath>
        </div>
        <div className="text-xs text-neutral-400 text-center mb-4">
          Complex plane plot: Real part (x-axis) vs Imaginary part (y-axis)
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="text-center">
            <div className="text-xs text-neutral-400 mb-2">Magnitude</div>
            <BlockMath>{`|Z| = \\sqrt{Z_{\\text{real}}^2 + Z_{\\text{imag}}^2}`}</BlockMath>
          </div>
          <div className="text-center">
            <div className="text-xs text-neutral-400 mb-2">Phase Angle</div>
            <BlockMath>{`\\phi = \\arctan\\left(\\frac{Z_{\\text{imag}}}{Z_{\\text{real}}}\\right)`}</BlockMath>
          </div>
        </div>
      </div>
    </div>
  );
};
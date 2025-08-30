import React from 'react';
import { BlockMath } from 'react-katex';

export const ResnormMethodsSection: React.FC = () => {
  return (
    <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
      <h3 className="text-lg font-medium text-neutral-200 mb-4">Residual Norm (Resnorm) for Parameter Fitting</h3>
      
      {/* Core Concept */}
      <div className="p-4 bg-neutral-900/50 rounded-lg mb-4">
        <h4 className="text-sm font-medium text-neutral-200 mb-3">Sum of Squared Residuals (SSR)</h4>
        <p className="text-sm text-neutral-300 mb-3">
          The application uses Sum of Squared Residuals (SSR) method for consistent impedance analysis.
        </p>
        <div className="text-center text-neutral-200 mb-3">
          <BlockMath>{`\\text{SSR} = \\frac{1}{N} \\sum_{i=1}^{N} \\sqrt{(Z_{\\text{real,test},i} - Z_{\\text{real,ref},i})^2 + (Z_{\\text{imag,test},i} - Z_{\\text{imag,ref},i})^2}`}</BlockMath>
        </div>
        <div className="text-xs text-neutral-400 text-center">
          Square root of sum of squared residuals with equal weight to real and imaginary components
        </div>
      </div>

      {/* SSR Method Details */}
      <div className="p-4 bg-neutral-900/50 rounded">
        <h4 className="text-sm font-medium text-blue-200 mb-3">Method Characteristics</h4>
        <ul className="text-xs text-neutral-400 space-y-1">
          <li>• Provides balanced weighting between real and imaginary impedance components</li>
          <li>• Suitable for electrochemical impedance spectroscopy parameter fitting</li>
          <li>• Consistent error measurement across frequency ranges</li>
          <li>• Standard approach for circuit parameter optimization</li>
        </ul>
      </div>

      {/* Implementation Notes */}
      <div className="mt-4 p-3 bg-blue-900/10 rounded border border-blue-700/20">
        <div className="text-xs text-blue-200 font-medium mb-2">Implementation:</div>
        <div className="text-xs text-blue-300/80 space-y-1">
          <div>• Direct calculation using real and imaginary components</div>
          <div>• No normalization needed - uses absolute impedance values</div>
          <div>• Optional frequency weighting and range amplification</div>
          <div>• Fixed method ensures consistent parameter fitting results</div>
        </div>
      </div>
    </div>
  );
};
"use client";

import React, { useState } from 'react';
import { ResnormConfig } from '../utils/resnorm';

interface ResnormMethodToggleProps {
  config: ResnormConfig;
  onChange: (config: ResnormConfig) => void;
  className?: string;
}

const ResnormMethodToggle: React.FC<ResnormMethodToggleProps> = ({
  config,
  onChange,
  className = ""
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleRangeAmplificationChange = (enabled: boolean) => {
    onChange({ ...config, useRangeAmplification: enabled });
  };

  const handleFrequencyWeightingChange = (enabled: boolean) => {
    onChange({ ...config, useFrequencyWeighting: enabled });
  };

  return (
    <div className={`bg-neutral-800/50 p-4 rounded border border-neutral-700 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-neutral-300">
          Resnorm Method
        </h4>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-primary hover:text-primary-light transition-colors"
        >
          {showAdvanced ? 'Hide' : 'Show'} Advanced
        </button>
      </div>

      {/* Fixed SSR Method Display */}
      <div className="mb-4 p-3 bg-neutral-900/50 rounded border border-neutral-600">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
            <div className="w-2 h-2 bg-white rounded-full"></div>
          </div>
          <div className="flex-1">
            <span className="text-sm text-neutral-200 font-medium">SSR - Sum of Squared Residuals</span>
            <p className="text-xs text-neutral-400">Fixed method for consistent impedance analysis</p>
          </div>
        </div>
      </div>

      {/* Advanced Options */}
      {showAdvanced && (
        <div className="space-y-3 pt-3 border-t border-neutral-600">
          <h5 className="text-xs font-medium text-neutral-400 uppercase tracking-wide">
            Advanced Options
          </h5>

          {/* General Options */}
          <div className="space-y-2">
            <label className="flex items-center justify-between">
              <span className="text-sm text-neutral-300">Range Amplification</span>
              <input
                type="checkbox"
                checked={config.useRangeAmplification || false}
                onChange={(e) => handleRangeAmplificationChange(e.target.checked)}
                className="w-4 h-4 text-primary border-neutral-600 rounded focus:ring-primary focus:ring-2"
              />
            </label>
            
            <label className="flex items-center justify-between">
              <span className="text-sm text-neutral-300">Frequency Weighting</span>
              <input
                type="checkbox"
                checked={config.useFrequencyWeighting || false}
                onChange={(e) => handleFrequencyWeightingChange(e.target.checked)}
                className="w-4 h-4 text-primary border-neutral-600 rounded focus:ring-primary focus:ring-2"
              />
            </label>
          </div>
        </div>
      )}

      {/* Current Method Summary */}
      <div className="mt-4 pt-3 border-t border-neutral-600">
        <div className="text-xs text-neutral-400">
          <span className="font-medium">Method:</span> SSR (Fixed)
          {config.useRangeAmplification && <span className="ml-2">• Range Amp</span>}
          {config.useFrequencyWeighting && <span className="ml-2">• Freq Weight</span>}
        </div>
      </div>
    </div>
  );
};

export default ResnormMethodToggle;
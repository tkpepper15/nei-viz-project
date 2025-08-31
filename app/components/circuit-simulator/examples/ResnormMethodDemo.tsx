"use client";

import React, { useState } from 'react';
import { 
  ResnormMethod, 
  ResnormConfig, 
  calculateResnormWithConfig, 
  groundTruthDataset,
  ImpedancePoint 
} from '../utils/resnorm';
import ResnormMethodToggle from '../controls/ResnormMethodToggle';

const ResnormMethodDemo: React.FC = () => {
  const [resnormConfig, setResnormConfig] = useState<ResnormConfig>({
    method: ResnormMethod.SSR,
    useRangeAmplification: false,
    useFrequencyWeighting: false
  });

  // Sample test data with some noise added to ground truth
  const testData: ImpedancePoint[] = groundTruthDataset.map(point => ({
    ...point,
    real: point.real * (1 + (Math.random() - 0.5) * 0.1), // ±5% noise
    imaginary: point.imaginary * (1 + (Math.random() - 0.5) * 0.1)
  }));

  const calculateExampleResnorm = (): number => {
    return calculateResnormWithConfig(
      testData,
      groundTruthDataset,
      undefined,
      undefined,
      resnormConfig
    );
  };

  const resnormValue = calculateExampleResnorm();

  // Calculate resnorm values for all methods for comparison
  const allMethods = Object.values(ResnormMethod);
  const comparisonResults = allMethods.map(method => {
    const config = { ...resnormConfig, method };
    const value = calculateResnormWithConfig(testData, groundTruthDataset, undefined, undefined, config);
    return { method, value };
  });

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-neutral-200 mb-2">
          Resnorm Method Comparison Demo
        </h1>
        <p className="text-neutral-400">
          Interactive demonstration of different resnorm calculation methods
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Method Toggle */}
        <div>
          <ResnormMethodToggle 
            config={resnormConfig}
            onChange={setResnormConfig}
          />
        </div>

        {/* Current Result */}
        <div className="bg-neutral-800/50 p-4 rounded border border-neutral-700">
          <h3 className="text-lg font-medium text-neutral-300 mb-3">
            Current Result
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-neutral-400">Method:</span>
              <span className="text-sm font-medium text-neutral-200">
                {resnormConfig.method.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-neutral-400">Resnorm Value:</span>
              <span className="text-lg font-mono text-primary">
                {resnormValue.toExponential(3)}
              </span>
            </div>
            {resnormConfig.useRangeAmplification && (
              <div className="text-xs text-yellow-400">
                + Range Amplification
              </div>
            )}
            {resnormConfig.useFrequencyWeighting && (
              <div className="text-xs text-blue-400">
                + Frequency Weighting
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Method Comparison Table */}
      <div className="bg-neutral-800/50 p-4 rounded border border-neutral-700">
        <h3 className="text-lg font-medium text-neutral-300 mb-3">
          Method Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-neutral-600">
                <th className="text-left py-2 text-neutral-300">Method</th>
                <th className="text-right py-2 text-neutral-300">Resnorm Value</th>
                <th className="text-right py-2 text-neutral-300">Relative to MAE</th>
              </tr>
            </thead>
            <tbody>
              {comparisonResults.map(({ method, value }) => {
                const maeValue = comparisonResults.find(r => r.method === ResnormMethod.SSR)?.value || 1;
                const relative = value / maeValue;
                const isCurrent = method === resnormConfig.method;
                
                return (
                  <tr 
                    key={method} 
                    className={`border-b border-neutral-700 ${isCurrent ? 'bg-primary/10' : ''}`}
                  >
                    <td className="py-2">
                      <span className={`font-medium ${isCurrent ? 'text-primary' : 'text-neutral-200'}`}>
                        {method.toUpperCase()}
                      </span>
                    </td>
                    <td className="text-right py-2 font-mono text-neutral-300">
                      {value.toExponential(3)}
                    </td>
                    <td className="text-right py-2 font-mono text-neutral-400">
                      {relative.toFixed(2)}×
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Data Information */}
      <div className="bg-neutral-800/50 p-4 rounded border border-neutral-700">
        <h3 className="text-lg font-medium text-neutral-300 mb-3">
          Test Data Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium text-neutral-200 mb-2">Ground Truth Data</h4>
            <ul className="space-y-1 text-neutral-400">
              <li>• {groundTruthDataset.length} frequency points</li>
              <li>• Frequency range: {Math.min(...groundTruthDataset.map(p => p.frequency))} - {Math.max(...groundTruthDataset.map(p => p.frequency))} Hz</li>
              <li>• Real impedance: {groundTruthDataset[0].real.toFixed(2)} - {groundTruthDataset[groundTruthDataset.length-1].real.toFixed(2)} Ω</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-neutral-200 mb-2">Test Data</h4>
            <ul className="space-y-1 text-neutral-400">
              <li>• Ground truth + ±5% random noise</li>
              <li>• Same frequency points</li>
              <li>• Simulates measurement uncertainty</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResnormMethodDemo;
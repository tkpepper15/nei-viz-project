import React from 'react';
import { CircuitParameters } from './types/parameters';
import { ModelSnapshot } from './types';
import { CircuitModelSection } from './math/CircuitModelSection';
import { CoreEquationsSection } from './math/CoreEquationsSection';
import { ParameterDisplaySection } from './math/ParameterDisplaySection';
import { ComplexImpedanceSection } from './math/ComplexImpedanceSection';
import { ResnormMethodsSection } from './math/ResnormMethodsSection';
import { NyquistPlotSection } from './math/NyquistPlotSection';
import { ReferenceModelComparison } from './math/ReferenceModelComparison';
import { calculateTER, calculateTimeConstants } from './math/utils';

interface MathDetailsTabProps {
  parameters: CircuitParameters;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  referenceModel: ModelSnapshot | null;
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
        <p className="text-neutral-400">Circuit equations and parameter analysis</p>
      </div>

      {/* Quick Reference */}
      <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
        <div className="flex items-center mb-3">
          <svg className="w-4 h-4 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm font-medium text-blue-200">Step-by-step calculations</span>
        </div>
        <p className="text-sm text-blue-300/80">
          Visit the <strong>Data Table</strong> tab, select a frequency, and click &quot;Show Math&quot; on any parameter set to see complete impedance calculations with intermediate steps.
        </p>
      </div>

      <CircuitModelSection />
      
      <CoreEquationsSection />
      
      <ParameterDisplaySection 
        parameters={parameters}
        minFreq={minFreq}
        maxFreq={maxFreq}
        ter={ter}
        tauA={tauA}
        tauB={tauB}
      />
      
      <ComplexImpedanceSection />
      
      <ResnormMethodsSection />
      
      <NyquistPlotSection />

      {referenceModel && (
        <ReferenceModelComparison
          parameters={parameters}
          referenceModel={referenceModel}
        />
      )}
    </div>
  );
};
import React from 'react';
import { CircuitParameters } from '../types/parameters';
import { ModelSnapshot } from '../types';

interface ReferenceModelComparisonProps {
  parameters: CircuitParameters;
  referenceModel: ModelSnapshot;
}

export const ReferenceModelComparison: React.FC<ReferenceModelComparisonProps> = ({
  parameters,
  referenceModel
}) => {
  return (
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
  );
};
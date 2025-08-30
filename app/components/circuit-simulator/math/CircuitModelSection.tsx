import React from 'react';
import { CircuitDiagram } from '../visualizations/CircuitDiagram';

export const CircuitModelSection: React.FC = () => {
  return (
    <div className="bg-neutral-800/50 rounded-lg p-5 border border-neutral-700">
      <h3 className="text-lg font-medium text-neutral-200 mb-4">Circuit Model</h3>
      <CircuitDiagram showLabels={true} />
      <div className="mt-4 p-3 bg-blue-900/10 rounded border border-blue-700/20">
        <div className="text-xs text-blue-200 text-center">
          <strong>Circuit Topology:</strong> Rₛₕ ∥ [(Rₐ ∥ Cₐ) + (Rᵦ ∥ Cᵦ)]
        </div>
        <div className="text-xs text-blue-300/80 mt-2 text-center">
          Apical and Basal membranes (RC circuits) in series, with parallel shunt resistance representing intercellular current flow
        </div>
      </div>
    </div>
  );
};
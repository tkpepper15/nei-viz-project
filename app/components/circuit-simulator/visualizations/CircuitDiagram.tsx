import React from 'react';
import Image from 'next/image';

interface CircuitDiagramProps {
  className?: string;
  showLabels?: boolean;
}

export const CircuitDiagram: React.FC<CircuitDiagramProps> = ({
  className = ''
}) => {
  return (
    <div className={`bg-neutral-900/50 p-6 rounded-lg border border-neutral-700 ${className}`}>
      <div className="text-center mb-4">
        <h4 className="text-sm font-medium text-neutral-200 mb-2">Modified Randles Circuit Model</h4>
        <p className="text-xs text-neutral-400">RPE Cell Impedance Equivalent Circuit</p>
      </div>
      
      <div className="flex justify-center">
        <Image 
          src="/img_9.png" 
          alt="Circuit diagram showing derivation of equivalent impedance Z_eq(ω) with Ra and Ca in apical membrane (Za), Rb and Cb in basal membrane (Zb), and mathematical formulas"
          width={800}
          height={400}
          className="max-w-full h-auto rounded-lg bg-white p-4"
          style={{ maxHeight: '400px' }}
        />
      </div>
      
      <div className="mt-4 text-center">
        <div className="text-neutral-400 text-xs">
          Circuit diagram showing mathematical derivation of equivalent impedance with Za(ω) and Zb(ω) formulations
        </div>
      </div>
    </div>
  );
};
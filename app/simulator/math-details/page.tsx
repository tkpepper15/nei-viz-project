"use client";

import { useState } from 'react';
import { MathDetailsTab } from '../../components/circuit-simulator/MathDetailsTab';
import { CircuitParameters } from '../../components/circuit-simulator/types/parameters';

const DEFAULT_PARAMS: CircuitParameters = {
  Rsh: 1000,
  Ra: 500,
  Ca: 1e-6,
  Rb: 300,
  Cb: 2e-6,
  frequency_range: [0.1, 100000],
};

export default function MathDetailsPage() {
  const [parameters] = useState<CircuitParameters>(DEFAULT_PARAMS);
  const [minFreq] = useState(0.1);
  const [maxFreq] = useState(100000);
  const [numPoints] = useState(50);

  return (
    <main className="min-h-screen bg-background">
      <div className="h-screen overflow-auto p-6">
        <MathDetailsTab
          parameters={parameters}
          minFreq={minFreq}
          maxFreq={maxFreq}
          numPoints={numPoints}
          referenceModel={null}
        />
      </div>
    </main>
  );
}

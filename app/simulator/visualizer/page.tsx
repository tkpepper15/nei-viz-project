"use client";

import { useState } from 'react';
import { VisualizerTab } from '../../components/circuit-simulator/VisualizerTab';
import { CircuitParameters } from '../../components/circuit-simulator/types/parameters';
import { ResnormGroup } from '../../components/circuit-simulator/types';
import { GridParameterArrays } from '../../components/circuit-simulator/types';
import { StaticRenderSettings, defaultStaticRenderSettings } from '../../components/circuit-simulator/controls/StaticRenderControls';
import { ResnormConfig, ResnormMethod } from '../../components/circuit-simulator/utils/resnorm';

const DEFAULT_PARAMS: CircuitParameters = {
  Rsh: 1000,
  Ra: 500,
  Ca: 1e-6,
  Rb: 300,
  Cb: 2e-6,
  frequency_range: [0.1, 100000],
};

export default function VisualizerPage() {
  const [resnormGroups] = useState<ResnormGroup[]>([]);
  const [hiddenGroups] = useState<number[]>([]);
  const [opacityLevel] = useState(0.8);
  const [opacityExponent, setOpacityExponent] = useState(1.0);
  const [gridSize] = useState(9);
  const [staticRenderSettings, setStaticRenderSettings] = useState<StaticRenderSettings>(defaultStaticRenderSettings);
  const [groundTruthParams] = useState<CircuitParameters>(DEFAULT_PARAMS);
  const [resnormConfig] = useState<ResnormConfig>({
    method: ResnormMethod.SSR,
    useFrequencyWeighting: false,
    useRangeAmplification: false,
  });
  const [groupPortion, setGroupPortion] = useState(0.25);

  const handleGridValuesGenerated = (values: GridParameterArrays) => {
    console.log('Grid values generated:', values);
  };

  const handleVisualizationSettingsChange = (settings: unknown) => {
    console.log('Visualization settings changed:', settings);
  };

  const handleGroupPortionChange = (value: number) => {
    setGroupPortion(value);
  };

  return (
    <main className="min-h-screen bg-background">
      <div className="h-screen overflow-hidden">
        <VisualizerTab
            resnormGroups={resnormGroups}
            hiddenGroups={hiddenGroups}
            opacityLevel={opacityLevel}
            referenceModelId={null}
            gridSize={gridSize}
            onGridValuesGenerated={handleGridValuesGenerated}
            opacityExponent={opacityExponent}
            onOpacityExponentChange={setOpacityExponent}
            userReferenceParams={null}
            showLabels={true}
            onVisualizationSettingsChange={handleVisualizationSettingsChange}
            staticRenderSettings={staticRenderSettings}
            onStaticRenderSettingsChange={setStaticRenderSettings}
            groundTruthParams={groundTruthParams}
            numPoints={50}
            resnormConfig={resnormConfig}
            groupPortion={groupPortion}
            onGroupPortionChange={handleGroupPortionChange}
          />
      </div>
    </main>
  );
}

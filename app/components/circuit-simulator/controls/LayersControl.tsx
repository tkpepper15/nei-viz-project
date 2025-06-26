import React from 'react';
import { ResnormGroup } from '../utils/types';

interface LayersControlProps {
  resnormGroups: ResnormGroup[];
  hiddenGroups: number[];
  referenceModelId: string | null;
}

export const LayersControl: React.FC<LayersControlProps> = ({
  resnormGroups,
  hiddenGroups,
  referenceModelId
}) => {
  // Function to toggle reference model visibility
  const handleToggleReference = () => {
    const event = new CustomEvent('toggleReferenceModel');
    window.dispatchEvent(event);
  };

  // Function to toggle resnorm group visibility
  const handleToggleGroup = (groupIndex: number) => {
    const event = new CustomEvent('toggleResnormGroup', {
      detail: { groupIndex }
    });
    window.dispatchEvent(event);
  };

  return (
    <div className="bg-neutral-900/50 rounded-lg border border-neutral-800 p-4 space-y-4">
      <div className="space-y-2">
        {/* Reference model toggle */}
        <div className="flex items-center justify-between p-2 bg-neutral-100/5 rounded-lg">
          <div className="flex items-center space-x-2">
            <div
              className={`w-3 h-3 rounded-full bg-black ${
                !referenceModelId ? 'opacity-50' : ''
              }`}
            />
            <span className="text-sm text-neutral-300">Reference Model</span>
          </div>
          <button
            onClick={() => handleToggleReference()}
            className="text-xs text-neutral-400 hover:text-neutral-300"
          >
            {referenceModelId ? 'Hide' : 'Show'}
          </button>
        </div>

        {/* Resnorm groups */}
        {resnormGroups.map((group, idx) => (
          <div key={idx} className="flex items-center justify-between p-2 bg-neutral-100/5 rounded-lg">
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    hiddenGroups.includes(idx) ? 'opacity-50' : ''
                  }`}
                  style={{ backgroundColor: group.color }}
                />
                <div className="flex flex-col">
                  <span className="text-sm text-neutral-300">{group.label}</span>
                  <span className="text-xs text-neutral-500">{group.description}</span>
                  <span className="text-xs text-neutral-400">
                    {group.items.length} {group.items.length === 1 ? 'model' : 'models'}
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={() => handleToggleGroup(idx)}
              className="text-xs text-neutral-400 hover:text-neutral-300 ml-2"
            >
              {hiddenGroups.includes(idx) ? 'Show' : 'Hide'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}; 
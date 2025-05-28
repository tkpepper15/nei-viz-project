import React from 'react';
import { ModelSnapshot } from './types';

interface ModelListProps {
  groundTruth: ModelSnapshot;
  optimizedModels: ModelSnapshot[];
  onToggleVisibility: (modelId: string) => void;
  activeTab: string;
}

export const ModelList: React.FC<ModelListProps> = ({
  groundTruth,
  optimizedModels,
  onToggleVisibility,
  activeTab
}) => {
  // Get models specific to the active tab
  const getTabSpecificModels = () => {
    if (activeTab === 'default') {
      // In default view, show only the ground truth model (but not visible by default)
      return groundTruth ? [groundTruth] : [];
    }
    
    // For specific model tabs, show that model and its optimized variations
    const baseModel = optimizedModels.find(m => m.id === activeTab);
    if (!baseModel) return [];
    
    // Get all models that were generated from this base model
    const relatedModels = optimizedModels.filter(m => 
      m.id !== baseModel.id && 
      m.name.includes(baseModel.name)
    );
    
    // Sort related models by resnorm
    const sortedRelatedModels = [...relatedModels].sort((a, b) => 
      (a.resnorm || 0) - (b.resnorm || 0)
    );
    
    return [baseModel, ...sortedRelatedModels];
  };

  const tabModels = getTabSpecificModels();

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">Model States</h3>
        <span className="text-xs text-gray-500">
          {tabModels.filter(m => m?.isVisible).length}/{tabModels.length} visible
        </span>
      </div>
      <div className="flex-1 overflow-y-auto">
        <div className="space-y-2">
          {tabModels.map((model) => (
            <div
              key={model.id}
              className="flex items-center justify-between p-2 bg-gray-50 rounded-lg hover:bg-gray-100"
            >
              <div className="flex items-center space-x-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ 
                    backgroundColor: model.color,
                    opacity: model.isVisible ? 1 : 0.5
                  }}
                />
                <div className="flex flex-col">
                  <span className="text-sm text-gray-700">{model.name}</span>
                  <span className="text-xs text-gray-500">
                    Resnorm: {model.resnorm?.toExponential(3) || '0'}
                  </span>
                </div>
              </div>
              <button
                onClick={() => onToggleVisibility(model.id)}
                className={`p-1 rounded ${
                  model.isVisible
                    ? 'text-blue-600 hover:bg-blue-50'
                    : 'text-gray-400 hover:bg-gray-200'
                }`}
                title={model.isVisible ? 'Hide model' : 'Show model'}
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  {model.isVisible ? (
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                  ) : (
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
                    />
                  )}
                </svg>
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}; 
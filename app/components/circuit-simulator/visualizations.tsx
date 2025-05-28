import React from 'react';
import { ModelSnapshot } from './types';

interface SpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string;
}

export const SpiderPlot: React.FC<SpiderPlotProps> = ({ meshItems, referenceId }) => {
  // Find the reference item
  const referenceItem = referenceId 
    ? meshItems.find(item => item.id === referenceId) 
    : meshItems[0];

  return (
    <div className="h-full w-full flex flex-col">
      <div className="py-2 px-4 border-b">
        <h3 className="text-sm font-medium text-gray-700">Parameter Spider Plot</h3>
        <p className="text-xs text-gray-500 mt-1">
          Showing {meshItems.length} model(s) 
          {referenceItem && ` with reference: ${referenceItem.name}`}
        </p>
      </div>
      
      <div className="flex-1 flex items-center justify-center bg-gray-50 p-4">
        {/* This would be replaced with an actual visualization library */}
        <div className="text-center">
          <div className="mb-4">
            <p className="text-gray-700 mb-2">Spider plot visualization would go here</p>
            <p className="text-xs text-gray-500">Normally rendered with a visualization library</p>
          </div>
          
          <div className="text-left max-w-md mx-auto bg-white border rounded p-4 shadow-sm">
            <h4 className="text-sm font-medium mb-2">Model Data Summary</h4>
            <ul className="text-xs space-y-1">
              <li>Total models: {meshItems.length}</li>
              <li>Reference model: {referenceItem?.name || 'None'}</li>
              <li>Parameter dimensions: 5 (Rs, Ra, Ca, Rb, Cb)</li>
              <li>
                Resnorm range: {
                  meshItems.length > 0 
                    ? `${Math.min(...meshItems.filter(m => m.resnorm !== undefined).map(m => m.resnorm || 0)).toExponential(2)} - ${Math.max(...meshItems.map(m => m.resnorm || 0)).toExponential(2)}`
                    : 'N/A'
                }
              </li>
            </ul>
            
            <div className="mt-4 grid grid-cols-2 gap-2">
              {meshItems.slice(0, 6).map(item => (
                <div 
                  key={item.id} 
                  className="p-2 text-xs rounded border" 
                  style={{
                    borderColor: item.color,
                    opacity: item.opacity
                  }}
                >
                  <div className="flex items-center mb-1">
                    <div 
                      className="w-2 h-2 rounded-full mr-2" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="font-medium">{item.name}</span>
                  </div>
                  <div>
                    Resnorm: {item.resnorm?.toExponential(2) || 'N/A'}
                  </div>
                </div>
              ))}
            </div>
            
            {meshItems.length > 6 && (
              <p className="text-xs text-gray-500 mt-2">+ {meshItems.length - 6} more models</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}; 
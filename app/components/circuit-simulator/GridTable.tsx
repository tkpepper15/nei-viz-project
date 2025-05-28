import React from 'react';
import { ModelSnapshot } from './types';

interface GridTableProps {
  models: ModelSnapshot[];
}

export const GridTable: React.FC<GridTableProps> = ({ models }) => {
  return (
    <div className="w-full">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rs</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Ra</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Ca</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rb</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Cb</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Resnorm</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {models.map((model, index) => (
            <tr key={model.id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{model.name}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{model.parameters.Rs.toFixed(2)}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{model.parameters.ra.toFixed(2)}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(model.parameters.ca * 1e6).toFixed(2)} μF</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{model.parameters.rb.toFixed(2)}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(model.parameters.cb * 1e6).toFixed(2)} μF</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{model.resnorm?.toExponential(3) || '0'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}; 
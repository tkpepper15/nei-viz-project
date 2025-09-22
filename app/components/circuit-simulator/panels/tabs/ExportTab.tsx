/**
 * Export Tab
 * ==========
 *
 * Dedicated tab for exporting impedance data and circuit configurations
 * - CSV export functionality
 * - JSON export for configurations
 * - Multiple format options
 */

import React, { useCallback, useState } from 'react';
import { BottomPanelTabProps } from '../CollapsibleBottomPanel';
import { ArrowDownTrayIcon, DocumentIcon, TableCellsIcon } from '@heroicons/react/24/outline';

interface ExportTabProps extends BottomPanelTabProps {
  exportFormat?: 'csv' | 'json';
}

export const ExportTab: React.FC<ExportTabProps> = ({
  gridResults,
  topConfigurations,
  currentParameters,
  isVisible
}) => {
  const [exportFormat, setExportFormat] = useState<'csv' | 'json'>('csv');
  const [isExporting, setIsExporting] = useState(false);

  // CSV export functionality
  const exportToCSV = useCallback(() => {
    if (!gridResults || gridResults.length === 0) return;

    setIsExporting(true);

    try {
      const headers = [
        'Model ID',
        'Resnorm',
        'Rsh (Ω)',
        'Ra (Ω)',
        'Rb (Ω)',
        'Ca (F)',
        'Cb (F)',
        'Frequency (Hz)',
        'Real (Ω)',
        'Imaginary (Ω)',
        'Magnitude (Ω)'
      ];

      const rows = [headers.join(',')];

      gridResults.forEach((result, index) => {
        if (result.parameters && result.data) {
          result.data.forEach(point => {
            const row = [
              `model-${index + 1}`,
              result.resnorm?.toFixed(6) || '0',
              result.parameters.Rsh.toFixed(6),
              result.parameters.Ra.toFixed(6),
              result.parameters.Rb.toFixed(6),
              result.parameters.Ca.toExponential(6),
              result.parameters.Cb.toExponential(6),
              point.frequency.toFixed(6),
              point.real.toFixed(6),
              point.imaginary.toFixed(6),
              point.magnitude.toFixed(6)
            ];
            rows.push(row.join(','));
          });
        }
      });

      const csvContent = rows.join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);

      link.setAttribute('href', url);
      link.setAttribute('download', `impedance_data_${new Date().toISOString().split('T')[0]}.csv`);
      link.style.visibility = 'hidden';

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      console.log(`✅ Exported ${gridResults.length} models to CSV`);
    } catch (error) {
      console.error('❌ CSV export failed:', error);
    } finally {
      setIsExporting(false);
    }
  }, [gridResults]);

  // JSON export functionality
  const exportToJSON = useCallback(() => {
    if (!gridResults || gridResults.length === 0) return;

    setIsExporting(true);

    try {
      const exportData = {
        exportDate: new Date().toISOString(),
        referenceParameters: currentParameters,
        models: gridResults.map((result, index) => ({
          id: `model-${index + 1}`,
          resnorm: result.resnorm,
          parameters: result.parameters,
          impedanceData: result.data
        }))
      };

      const jsonContent = JSON.stringify(exportData, null, 2);
      const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);

      link.setAttribute('href', url);
      link.setAttribute('download', `impedance_models_${new Date().toISOString().split('T')[0]}.json`);
      link.style.visibility = 'hidden';

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      console.log(`✅ Exported ${gridResults.length} models to JSON`);
    } catch (error) {
      console.error('❌ JSON export failed:', error);
    } finally {
      setIsExporting(false);
    }
  }, [gridResults, currentParameters]);

  if (!isVisible) return null;

  return (
    <div className="flex-1 overflow-y-auto min-h-0 bg-neutral-900">
      <div className="p-4 space-y-4">

        {/* Export Summary */}
        <div className="bg-neutral-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-neutral-200 mb-2">Export Summary</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-neutral-400">Available Models:</span>
              <span className="ml-2 font-mono text-neutral-200">{gridResults?.length || 0}</span>
            </div>
            <div>
              <span className="text-neutral-400">Configurations:</span>
              <span className="ml-2 font-mono text-neutral-200">{topConfigurations?.length || 0}</span>
            </div>
          </div>
        </div>

        {/* Export Format Selection */}
        <div className="bg-neutral-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-neutral-200 mb-3">Export Format</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setExportFormat('csv')}
              className={`flex-1 px-3 py-2 rounded-md text-sm transition-colors flex items-center justify-center gap-2 ${
                exportFormat === 'csv'
                  ? 'bg-blue-600 text-white'
                  : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
              }`}
            >
              <TableCellsIcon className="w-4 h-4" />
              CSV (Spreadsheet)
            </button>
            <button
              onClick={() => setExportFormat('json')}
              className={`flex-1 px-3 py-2 rounded-md text-sm transition-colors flex items-center justify-center gap-2 ${
                exportFormat === 'json'
                  ? 'bg-blue-600 text-white'
                  : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
              }`}
            >
              <DocumentIcon className="w-4 h-4" />
              JSON (Structured)
            </button>
          </div>
        </div>

        {/* Export Actions */}
        <div className="bg-neutral-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-neutral-200 mb-3">Export Data</h3>

          {exportFormat === 'csv' && (
            <div className="space-y-3">
              <p className="text-xs text-neutral-400">
                Export impedance data as CSV file including model parameters, frequency points,
                and complex impedance values. Compatible with Excel, Google Sheets, and data analysis tools.
              </p>
              <button
                onClick={exportToCSV}
                disabled={isExporting || !gridResults || gridResults.length === 0}
                className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-neutral-600 disabled:cursor-not-allowed text-white rounded-md transition-colors flex items-center justify-center gap-2"
              >
                {isExporting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Exporting...
                  </>
                ) : (
                  <>
                    <ArrowDownTrayIcon className="w-4 h-4" />
                    Export to CSV
                  </>
                )}
              </button>
            </div>
          )}

          {exportFormat === 'json' && (
            <div className="space-y-3">
              <p className="text-xs text-neutral-400">
                Export complete model data as JSON including metadata, parameters, and impedance spectra.
                Ideal for programmatic analysis and data processing workflows.
              </p>
              <button
                onClick={exportToJSON}
                disabled={isExporting || !gridResults || gridResults.length === 0}
                className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-neutral-600 disabled:cursor-not-allowed text-white rounded-md transition-colors flex items-center justify-center gap-2"
              >
                {isExporting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Exporting...
                  </>
                ) : (
                  <>
                    <ArrowDownTrayIcon className="w-4 h-4" />
                    Export to JSON
                  </>
                )}
              </button>
            </div>
          )}

          {(!gridResults || gridResults.length === 0) && (
            <div className="text-center py-4 text-neutral-400">
              <p className="text-sm">No data available for export</p>
              <p className="text-xs mt-1">Run a computation to generate exportable data</p>
            </div>
          )}
        </div>

        {/* Export Notes */}
        <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-700/30">
          <h4 className="text-xs font-medium text-blue-200 mb-2">Export Notes</h4>
          <ul className="text-xs text-blue-300/80 space-y-1">
            <li>• CSV format: One row per frequency point per model</li>
            <li>• JSON format: Hierarchical structure with metadata</li>
            <li>• All impedance values in Ohms (Ω)</li>
            <li>• Capacitance values in Farads (F)</li>
            <li>• Files named with current date for organization</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
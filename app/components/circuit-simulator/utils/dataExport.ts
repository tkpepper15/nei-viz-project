/**
 * Data Export Utilities for Directional Analysis Results
 *
 * Provides functions to export analysis results in various formats:
 * - CSV for tabular data
 * - JSON for structured data
 * - NPZ format for numpy compatibility
 */

import { CircuitParameters } from '../types/parameters';
import { DirectionalSensitivity } from './directionalAnalysis';

export interface ExportableResults {
  sensitivity: DirectionalSensitivity;
  adaptiveSamples: CircuitParameters[];
  uncertaintyMap: { params: CircuitParameters; uncertainty: number }[];
  spectralCompression: unknown;
  metadata: {
    exportDate: string;
    version: string;
    analysisType: 'directional_sensitivity';
  };
}

/**
 * Export directional analysis results to CSV format
 */
export function exportToCSV(results: ExportableResults): string {
  const { sensitivity, adaptiveSamples, uncertaintyMap } = results;

  let csv = '';

  // Header information
  csv += '# Directional Analysis Export\n';
  csv += `# Export Date: ${results.metadata.exportDate}\n`;
  csv += `# Analysis Type: ${results.metadata.analysisType}\n`;
  csv += `# Condition Number: ${sensitivity.conditionNumber.toFixed(6)}\n`;
  csv += '\n';

  // Principal directions
  csv += '# Principal Directions\n';
  csv += 'Direction,Sensitivity,Rsh_Component,Ra_Component,Ca_Component,Rb_Component,Cb_Component,Interpretation,Spectral_Effect\n';

  sensitivity.principalDirections.forEach((dir, index) => {
    csv += `${index + 1},${dir.sensitivity.toFixed(6)},`;
    csv += dir.direction.map(d => d.toFixed(6)).join(',');
    csv += `,"${dir.interpretation}","${dir.spectralEffect}"\n`;
  });

  csv += '\n';

  // Adaptive samples
  csv += '# Adaptive Sampling Points\n';
  csv += 'Sample_ID,Rsh,Ra,Ca,Rb,Cb,Freq_Min,Freq_Max\n';

  adaptiveSamples.forEach((params, index) => {
    csv += `${index + 1},${params.Rsh.toFixed(3)},${params.Ra.toFixed(3)},`;
    csv += `${params.Ca.toExponential(3)},${params.Rb.toFixed(3)},${params.Cb.toExponential(3)},`;
    csv += `${params.frequency_range[0]},${params.frequency_range[1]}\n`;
  });

  csv += '\n';

  // Uncertainty map
  if (uncertaintyMap.length > 0) {
    csv += '# Model Uncertainty Map\n';
    csv += 'Point_ID,Rsh,Ra,Ca,Rb,Cb,Uncertainty\n';

    uncertaintyMap.forEach((point, index) => {
      csv += `${index + 1},${point.params.Rsh.toFixed(3)},${point.params.Ra.toFixed(3)},`;
      csv += `${point.params.Ca.toExponential(3)},${point.params.Rb.toFixed(3)},${point.params.Cb.toExponential(3)},`;
      csv += `${point.uncertainty.toFixed(6)}\n`;
    });
  }

  return csv;
}

/**
 * Export directional analysis results to JSON format
 */
export function exportToJSON(results: ExportableResults): string {
  // Create a clean export object with all numeric values properly formatted
  const cleanResults = {
    metadata: results.metadata,
    sensitivity: {
      conditionNumber: Number(results.sensitivity.conditionNumber.toFixed(6)),
      principalDirections: results.sensitivity.principalDirections.map(dir => ({
        direction: dir.direction.map(d => Number(d.toFixed(6))),
        sensitivity: Number(dir.sensitivity.toFixed(6)),
        interpretation: dir.interpretation,
        spectralEffect: dir.spectralEffect
      })),
      curvatureEigenvalues: results.sensitivity.curvatureEigenvalues.map(e => Number(e.toFixed(6)))
    },
    adaptiveSamples: results.adaptiveSamples.map(params => ({
      Rsh: Number(params.Rsh.toFixed(3)),
      Ra: Number(params.Ra.toFixed(3)),
      Ca: Number(params.Ca.toExponential(3)),
      Rb: Number(params.Rb.toFixed(3)),
      Cb: Number(params.Cb.toExponential(3)),
      frequency_range: params.frequency_range
    })),
    uncertaintyMap: results.uncertaintyMap.map(point => ({
      params: {
        Rsh: Number(point.params.Rsh.toFixed(3)),
        Ra: Number(point.params.Ra.toFixed(3)),
        Ca: Number(point.params.Ca.toExponential(3)),
        Rb: Number(point.params.Rb.toFixed(3)),
        Cb: Number(point.params.Cb.toExponential(3)),
        frequency_range: point.params.frequency_range
      },
      uncertainty: Number(point.uncertainty.toFixed(6))
    }))
  };

  return JSON.stringify(cleanResults, null, 2);
}

/**
 * Create a comprehensive analysis summary
 */
export function createAnalysisSummary(results: ExportableResults): string {
  const { sensitivity } = results;

  let summary = '';
  summary += '=== DIRECTIONAL ANALYSIS SUMMARY ===\n\n';

  // Overall assessment
  summary += '1. PARAMETER IDENTIFIABILITY\n';
  const conditionNumber = sensitivity.conditionNumber;
  if (conditionNumber < 10) {
    summary += `   ✓ EXCELLENT (Condition Number: ${conditionNumber.toFixed(2)})\n`;
    summary += '   All parameters are well-identifiable from impedance data.\n';
  } else if (conditionNumber < 100) {
    summary += `   ◐ MODERATE (Condition Number: ${conditionNumber.toFixed(2)})\n`;
    summary += '   Most parameters identifiable, some correlation exists.\n';
  } else {
    summary += `   ✗ POOR (Condition Number: ${conditionNumber.toFixed(2)})\n`;
    summary += '   Strong parameter correlations, difficult to identify uniquely.\n';
  }

  summary += '\n2. PRINCIPAL DIRECTIONS OF SENSITIVITY\n';
  sensitivity.principalDirections.forEach((dir, index) => {
    summary += `   Direction ${index + 1}: ${dir.interpretation}\n`;
    summary += `     Sensitivity: ${dir.sensitivity.toFixed(4)}\n`;
    summary += `     Effect: ${dir.spectralEffect}\n`;

    // Find dominant parameters
    const paramNames = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];
    const dominantParams = dir.direction
      .map((val, idx) => ({ param: paramNames[idx], value: Math.abs(val) }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 2); // Top 2 parameters

    summary += `     Dominant: ${dominantParams.map(p => `${p.param}(${(p.value * 100).toFixed(1)}%)`).join(', ')}\n`;
    summary += '\n';
  });

  summary += '3. RECOMMENDATIONS\n';
  if (conditionNumber < 10) {
    summary += '   • Excellent parameter identifiability - proceed with confidence\n';
    summary += '   • Focus measurements on frequencies with highest sensitivity\n';
  } else if (conditionNumber < 100) {
    summary += '   • Consider additional measurement frequencies\n';
    summary += '   • Use regularization in parameter fitting\n';
  } else {
    summary += '   • Strong parameter correlations detected\n';
    summary += '   • Consider constraining some parameters\n';
    summary += '   • Increase measurement frequency range or density\n';
  }

  summary += '\n4. ADAPTIVE SAMPLING\n';
  summary += `   Generated ${results.adaptiveSamples.length} intelligent sampling points\n`;
  summary += '   70% focused on high-sensitivity directions\n';
  summary += '   30% random exploration for robustness\n';

  if (results.uncertaintyMap.length > 0) {
    const avgUncertainty = results.uncertaintyMap.reduce((sum, u) => sum + u.uncertainty, 0) / results.uncertaintyMap.length;
    const maxUncertainty = Math.max(...results.uncertaintyMap.map(u => u.uncertainty));

    summary += '\n5. MODEL UNCERTAINTY\n';
    summary += `   Average uncertainty: ${avgUncertainty.toFixed(4)}\n`;
    summary += `   Maximum uncertainty: ${maxUncertainty.toFixed(4)}\n`;
    summary += '   Use uncertainty map to guide additional measurements\n';
  }

  return summary;
}

/**
 * Download file with given content and filename
 */
export function downloadFile(content: string, filename: string, mimeType: string = 'text/plain'): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();

  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export complete analysis package
 */
export function exportAnalysisPackage(results: ExportableResults, baseName: string = 'directional_analysis'): void {
  const timestamp = new Date().toISOString().split('T')[0]; // YYYY-MM-DD

  // Export CSV
  const csvContent = exportToCSV(results);
  downloadFile(csvContent, `${baseName}_${timestamp}.csv`, 'text/csv');

  // Export JSON
  const jsonContent = exportToJSON(results);
  downloadFile(jsonContent, `${baseName}_${timestamp}.json`, 'application/json');

  // Export summary
  const summaryContent = createAnalysisSummary(results);
  downloadFile(summaryContent, `${baseName}_summary_${timestamp}.txt`, 'text/plain');
}

/**
 * Create NPZ-compatible data structure
 */
export function createNPZData(results: ExportableResults): Record<string, number[][]> {
  const { sensitivity, adaptiveSamples, uncertaintyMap } = results;

  const npzData: Record<string, number[][]> = {};

  // Principal directions matrix (directions × parameters)
  npzData.principal_directions = sensitivity.principalDirections.map(dir => dir.direction);

  // Sensitivity values
  npzData.sensitivities = [sensitivity.principalDirections.map(dir => dir.sensitivity)];

  // Adaptive samples matrix (samples × parameters)
  npzData.adaptive_samples = adaptiveSamples.map(params => [
    params.Rsh, params.Ra, params.Ca, params.Rb, params.Cb
  ]);

  // Uncertainty data
  if (uncertaintyMap.length > 0) {
    npzData.uncertainty_params = uncertaintyMap.map(point => [
      point.params.Rsh, point.params.Ra, point.params.Ca, point.params.Rb, point.params.Cb
    ]);
    npzData.uncertainty_values = [uncertaintyMap.map(point => point.uncertainty)];
  }

  // Metadata as single values
  npzData.condition_number = [[sensitivity.conditionNumber]];
  npzData.num_directions = [[sensitivity.principalDirections.length]];

  return npzData;
}

/**
 * Format NPZ data as JSON for export
 */
export function exportNPZData(results: ExportableResults, filename: string = 'directional_analysis.npz.json'): void {
  const npzData = createNPZData(results);
  const jsonContent = JSON.stringify(npzData, null, 2);
  downloadFile(jsonContent, filename, 'application/json');
}
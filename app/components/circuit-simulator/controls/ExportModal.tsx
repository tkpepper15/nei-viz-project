"use client";

import React, { useState, useCallback } from 'react';
import { SpiderPlot } from '../visualizations/SpiderPlot';
import { ModelSnapshot } from '../types';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  visibleModels: ModelSnapshot[];
  opacityLevel: number;
  chromaEnabled?: boolean;
  opacityIntensity?: number;
  gridSize: number;
}

type ExportFormat = 'svg' | 'png';
type ExportQuality = 'standard' | 'high' | 'print';

const qualitySettings = {
  standard: { width: 800, height: 800, scale: 1 },
  high: { width: 1600, height: 1600, scale: 2 },
  print: { width: 3200, height: 3200, scale: 4 }
};

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  visibleModels,
  opacityLevel,
  chromaEnabled = true,
  opacityIntensity = 1.0,
  gridSize
}) => {
  const [format, setFormat] = useState<ExportFormat>('png');
  const [quality, setQuality] = useState<ExportQuality>('high');
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = useCallback(async () => {
    setIsExporting(true);
    
    try {
      const { width, height } = qualitySettings[quality];
      
      if (format === 'svg') {
        // Try to get SVG element from the preview
        let svgElement = document.querySelector('#export-preview svg') as SVGElement;
        
        if (!svgElement) {
          // Create SVG programmatically from SpiderPlot data
          svgElement = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          svgElement.setAttribute('width', width.toString());
          svgElement.setAttribute('height', height.toString());
          svgElement.setAttribute('viewBox', `0 0 ${width} ${height}`);
          
          // Add background
          const background = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          background.setAttribute('width', width.toString());
          background.setAttribute('height', height.toString());
          background.setAttribute('fill', '#0f172a');
          svgElement.appendChild(background);
          
          // Add placeholder content
          const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          text.setAttribute('x', (width / 2).toString());
          text.setAttribute('y', (height / 2).toString());
          text.setAttribute('text-anchor', 'middle');
          text.setAttribute('fill', '#ffffff');
          text.setAttribute('font-family', 'Arial');
          text.setAttribute('font-size', '24');
          text.textContent = 'Spider Plot Export';
          svgElement.appendChild(text);
        }
        
        // Clone and prepare SVG for export
        const clonedSvg = svgElement.cloneNode(true) as SVGElement;
        clonedSvg.setAttribute('width', width.toString());
        clonedSvg.setAttribute('height', height.toString());
        clonedSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        
        // Create blob and download
        const svgData = new XMLSerializer().serializeToString(clonedSvg);
        const blob = new Blob([svgData], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `spider-plot-${quality}-${Date.now()}.svg`;
        link.click();
        
        URL.revokeObjectURL(url);
      } else {
        // PNG export using canvas - try to get existing canvas first
        let sourceCanvas = document.querySelector('#export-preview canvas') as HTMLCanvasElement;
        
        if (!sourceCanvas) {
          // Create canvas programmatically
          sourceCanvas = document.createElement('canvas');
          sourceCanvas.width = width;
          sourceCanvas.height = height;
          const ctx = sourceCanvas.getContext('2d');
          if (ctx) {
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = '#ffffff';
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Spider Plot Export', width / 2, height / 2);
          }
        }
        
        // Create export canvas
        const exportCanvas = document.createElement('canvas');
        const ctx = exportCanvas.getContext('2d');
        exportCanvas.width = width;
        exportCanvas.height = height;
        
        if (ctx) {
          // Set background
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(0, 0, width, height);
          
          // Draw source canvas scaled to fit
          ctx.drawImage(sourceCanvas, 0, 0, width, height);
          
          exportCanvas.toBlob((blob) => {
            if (blob) {
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `spider-plot-${quality}-${Date.now()}.png`;
              link.click();
              URL.revokeObjectURL(url);
            }
          }, 'image/png');
        }
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed. Please try again or contact support.');
    } finally {
      setIsExporting(false);
    }
  }, [format, quality]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div className="bg-neutral-900 rounded-lg border border-neutral-700 shadow-xl max-w-5xl w-full max-h-[85vh] overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-700">
          <h2 className="text-lg font-medium text-white">Export Spider Plot</h2>
          <button
            onClick={onClose}
            className="p-1.5 text-neutral-400 hover:text-white hover:bg-neutral-800 rounded transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex">
          {/* Settings Panel */}
          <div className="w-64 p-4 border-r border-neutral-700 space-y-4">
            
            {/* Format */}
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2">Format</label>
              <div className="space-y-1">
                <button
                  onClick={() => setFormat('png')}
                  className={`w-full p-2 text-sm rounded border text-left transition-colors ${
                    format === 'png'
                      ? 'border-blue-500 bg-blue-500/10 text-blue-300'
                      : 'border-neutral-600 hover:border-neutral-500 text-neutral-300'
                  }`}
                >
                  PNG (Raster)
                </button>
                <button
                  onClick={() => setFormat('svg')}
                  className={`w-full p-2 text-sm rounded border text-left transition-colors ${
                    format === 'svg'
                      ? 'border-blue-500 bg-blue-500/10 text-blue-300'
                      : 'border-neutral-600 hover:border-neutral-500 text-neutral-300'
                  }`}
                >
                  SVG (Vector)
                </button>
              </div>
            </div>

            {/* Quality */}
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2">Quality</label>
              <div className="space-y-1">
                {Object.entries(qualitySettings).map(([key, settings]) => (
                  <button
                    key={key}
                    onClick={() => setQuality(key as ExportQuality)}
                    className={`w-full p-2 text-sm rounded border text-left transition-colors ${
                      quality === key
                        ? 'border-blue-500 bg-blue-500/10 text-blue-300'
                        : 'border-neutral-600 hover:border-neutral-500 text-neutral-300'
                    }`}
                  >
                    <div className="capitalize">{key}</div>
                    <div className="text-xs text-neutral-400">{settings.width}×{settings.height}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Details */}
            <div className="bg-neutral-800 rounded p-3 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-neutral-400">Models:</span>
                <span className="text-neutral-300">{visibleModels.length}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-neutral-400">Grid Size:</span>
                <span className="text-neutral-300">{gridSize}×{gridSize}×{gridSize}×{gridSize}×{gridSize}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-neutral-400">Mode:</span>
                <span className="text-neutral-300">{chromaEnabled ? 'Chroma' : 'Mono'}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-neutral-400">Opacity:</span>
                <span className="text-neutral-300">{opacityIntensity.toFixed(1)}x</span>
              </div>
            </div>

            {/* Export Button */}
            <button
              onClick={handleExport}
              disabled={isExporting || visibleModels.length === 0}
              className={`w-full py-2.5 px-4 rounded font-medium transition-colors ${
                isExporting || visibleModels.length === 0
                  ? 'bg-neutral-700 text-neutral-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isExporting ? 'Exporting...' : `Export ${format.toUpperCase()}`}
            </button>
          </div>

          {/* Preview Panel */}
          <div className="flex-1 p-4">
            <div className="h-[calc(85vh-8rem)]">
              <div className="text-sm text-neutral-400 mb-2">Preview</div>
              <div 
                id="export-preview"
                className="w-full h-full bg-neutral-800 rounded border border-neutral-700 overflow-hidden"
              >
                <SpiderPlot
                  meshItems={visibleModels}
                  opacityFactor={opacityLevel}
                  maxPolygons={1000000}
                  visualizationMode={chromaEnabled ? 'color' : 'opacity'}
                  gridSize={gridSize}
                  includeLabels={true}
                  backgroundColor="transparent"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 
"use client";

import React, { useState } from 'react';
import { SpiderPlot } from './visualizations/SpiderPlot';
import { ModelSnapshot } from './types';

interface TileRenderingExampleProps {
  meshItems: ModelSnapshot[];
  opacityFactor: number;
  maxPolygons: number;
  visualizationMode?: 'color' | 'opacity';
  opacityIntensity?: number;
}

export const TileRenderingExample: React.FC<TileRenderingExampleProps> = ({
  meshItems,
  opacityFactor,
  maxPolygons,
  visualizationMode = 'color',
  opacityIntensity = 1.0
}) => {
  const [useTileRendering, setUseTileRendering] = useState(false);
  const [exportedImage, setExportedImage] = useState<string | null>(null);

  return (
    <div className="w-full h-full space-y-4">
      {/* Renderer Selection */}
      <div className="flex items-center justify-between bg-neutral-800 p-4 rounded-lg">
        <div>
          <h3 className="text-lg font-semibold text-white mb-1">
            Spider Plot Rendering
          </h3>
          <p className="text-sm text-neutral-400">
            Choose between interactive and high-performance tile-based rendering
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <input
              type="radio"
              id="interactive"
              name="renderer"
              checked={!useTileRendering}
              onChange={() => setUseTileRendering(false)}
              className="w-4 h-4"
            />
            <label htmlFor="interactive" className="text-sm text-neutral-300">
              Interactive (SVG)
            </label>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="radio"
              id="tiled"
              name="renderer"
              checked={useTileRendering}
              onChange={() => setUseTileRendering(true)}
              className="w-4 h-4"
            />
            <label htmlFor="tiled" className="text-sm text-neutral-300">
              Tile-based (Canvas)
            </label>
          </div>
        </div>
      </div>

      {/* Performance Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <div className="bg-neutral-800 p-3 rounded">
          <div className="text-primary font-semibold">Interactive Mode</div>
          <div className="text-neutral-400 mt-1">
            • Real-time interaction<br/>
            • Suitable for {'<'}10k polygons<br/>
            • SVG-based rendering<br/>
            • Memory efficient
          </div>
        </div>
        
        <div className="bg-neutral-800 p-3 rounded">
          <div className="text-amber-400 font-semibold">Tile-based Mode</div>
          <div className="text-neutral-400 mt-1">
            • High-quality static rendering<br/>
            • Handles 100k+ polygons<br/>
            • Multi-threaded processing<br/>
            • Export-optimized
          </div>
        </div>
        
        <div className="bg-neutral-800 p-3 rounded">
          <div className="text-green-400 font-semibold">Performance Tips</div>
          <div className="text-neutral-400 mt-1">
            • Use interactive for exploration<br/>
            • Switch to tiled for final renders<br/>
            • Adjust quality based on use case<br/>
            • Enable tile previews to see progress
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="relative bg-slate-900 rounded-lg" style={{ height: '600px' }}>
        {useTileRendering ? (
          <SpiderPlot
            meshItems={meshItems}
            opacityFactor={opacityFactor}
            maxPolygons={maxPolygons}
            visualizationMode={visualizationMode}
            opacityIntensity={opacityIntensity}
            gridSize={5}
            includeLabels={true}
            backgroundColor="transparent"
          />
        ) : (
          <SpiderPlot
            meshItems={meshItems}
            opacityFactor={opacityFactor}
            maxPolygons={Math.min(maxPolygons, 10000)} // Limit for interactive mode
            visualizationMode={visualizationMode}
            opacityIntensity={opacityIntensity}
            gridSize={5}
            includeLabels={true}
            backgroundColor="transparent"
          />
        )}
        
        {/* Performance Indicator */}
        <div className="absolute top-2 right-2 px-2 py-1 rounded text-xs font-mono bg-black/50 text-white">
          {meshItems.length.toLocaleString()} polygons
        </div>
      </div>

      {/* Export Preview */}
      {exportedImage && (
        <div className="bg-neutral-800 p-4 rounded-lg">
          <h4 className="text-white font-medium mb-2">Exported Image</h4>
          <div className="flex items-start gap-4">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img 
              src={exportedImage} 
              alt="Exported spider plot" 
              className="max-w-xs border border-neutral-600 rounded"
            />
            <div className="text-sm text-neutral-400">
              <p>High-resolution image exported successfully!</p>
              <p className="mt-2">
                The tile-based rendering system allows you to export images 
                at much higher resolutions than what can be displayed interactively.
              </p>
              <button
                onClick={() => setExportedImage(null)}
                className="mt-2 px-3 py-1 bg-neutral-700 hover:bg-neutral-600 text-white rounded text-xs"
              >
                Clear Preview
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 
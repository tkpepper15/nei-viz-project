import React, { useEffect, useMemo, useState } from 'react';
import { ModelSnapshot, ResnormGroup } from '../circuit-simulator/utils/types';
import { BaseSpiderPlot } from '../circuit-simulator';

// Extend ModelSnapshot type for our internal use
interface ModelSnapshotWithGroup extends ModelSnapshot {
  groupIdx?: number; // Current index in the resnormGroups array
  qualityCategory?: number; // Fixed quality category (0=Very Good, 1=Good, 2=Moderate, 3=Poor)
}

type VisualizerTabProps = {
  resnormGroups: ResnormGroup[];
  hiddenGroups: number[];
  maxGridPoints: number;
  opacityLevel: number;
  logScalar: number;
  referenceModelId: string | null;
  referenceModel: ModelSnapshot | null;
  minFreq: number;
  maxFreq: number;
};

// Helper to determine quality category based on group range
const getQualityCategory = (group: ResnormGroup): number => {
  // We can use the color as a reliable indicator of the quality category
  switch (group.color) {
    case '#10B981': return 0; // Very Good - Green
    case '#3B82F6': return 1; // Good - Blue 
    case '#F59E0B': return 2; // Moderate - Amber
    case '#EF4444': return 3; // Poor - Red
    default: return 1; // Default to Good
  }
};

const VisualizerTab: React.FC<VisualizerTabProps> = ({
  resnormGroups,
  hiddenGroups,
  maxGridPoints,
  opacityLevel,
  logScalar,
  referenceModelId,
  referenceModel,
  minFreq,
  maxFreq
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [localOpacity, setLocalOpacity] = useState(opacityLevel);
  const [localLogScale, setLocalLogScale] = useState(logScalar);
  // Track reference visibility separately from referenceModelId
  const [showReference, setShowReference] = useState(referenceModelId !== null);

  // When key data changes, show loading indicator
  useEffect(() => {
    setIsLoading(true);
    const timer = setTimeout(() => setIsLoading(false), 300);
    return () => clearTimeout(timer);
  }, [resnormGroups.length, hiddenGroups.length, maxGridPoints]);
  
  // Update local state when props change
  useEffect(() => {
    setLocalOpacity(opacityLevel);
    setLocalLogScale(logScalar);
  }, [opacityLevel, logScalar]);
  
  // Update reference visibility when referenceModelId changes
  useEffect(() => {
    setShowReference(referenceModelId !== null);
  }, [referenceModelId]);
  
  // Process mesh items (grid points only)
  const gridMeshItems = useMemo(() => {
    // First map all items from all groups with group index AND quality category
    const allGroupItems: ModelSnapshotWithGroup[] = resnormGroups.flatMap((group, groupIdx) => {
      // Determine quality category based on group properties
      const qualityCategory = getQualityCategory(group);
      
      // Get the appropriate color based on quality category
      let color = '#3B82F6'; // Default blue
      if (qualityCategory === 0) color = '#10B981'; // Very Good
      else if (qualityCategory === 1) color = '#3B82F6'; // Good
      else if (qualityCategory === 2) color = '#F59E0B'; // Moderate
      else if (qualityCategory === 3) color = '#EF4444'; // Poor
      
      return group.items.map(item => ({
        ...item,
        groupIdx, // Current array index for filtering
        qualityCategory, // Fixed quality category for consistent colors
        color: color // Ensure consistent color
      }));
    });
    
    // Then filter by hidden status using the current groupIdx
    const visibleItems = allGroupItems.filter(item => 
      item.groupIdx !== undefined && !hiddenGroups.includes(item.groupIdx)
    );
      
    // Return empty array if no items
    if (visibleItems.length === 0) return [];
    
    // Calculate how many items to display based on maxGridPoints
    const itemsToDisplay = Math.min(maxGridPoints, visibleItems.length);
    
    // Process each item with proper opacity
    return visibleItems
      .slice(0, itemsToDisplay)
      .map(item => ({
        ...item,
        opacity: Math.min(1, localOpacity * (item.opacity || 0.7))
      }));
  }, [resnormGroups, hiddenGroups, maxGridPoints, localOpacity]);
  
  // Process reference model separately
  const referenceLayer = useMemo((): ModelSnapshotWithGroup | null => {
    if (!referenceModel || !showReference) return null;
    
    return {
      ...referenceModel,
      opacity: 1.0, // Reference always fully visible
      isVisible: true
    };
  }, [referenceModel, showReference]);
  
  // Combined items for display - don't use this for hasDataToVisualize check
  const combinedItems = useMemo(() => {
    const items = [...gridMeshItems];
    if (referenceLayer) {
      items.unshift(referenceLayer);
    }
    return items;
  }, [gridMeshItems, referenceLayer]);
  
  // Determine if we have any data to visualize
  const hasGridData = gridMeshItems.length > 0;
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full bg-spider-bg/40">
        <div className="animate-pulse flex flex-col items-center">
          <svg className="w-12 h-12 text-primary/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 12l5-5 5 5M7 20l5-5 5 5" />
          </svg>
          <p className="text-xs text-neutral-500 mt-2">Updating visualization...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* Main Visualization Area */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 h-full min-h-[500px]">
        {/* Spider Plot */}
        <div className={`card spider-card md:col-span-2 h-full flex flex-col ${
          maxFreq - minFreq > 500 
            ? 'freq-range-wide' 
            : maxFreq - minFreq > 100 
            ? 'freq-range-moderate' 
            : 'freq-range-narrow'
        }`}>
          <div className="relative flex-1 w-full min-h-[450px]">
            <BaseSpiderPlot
              meshItems={combinedItems}
              referenceId={showReference && referenceModel ? 'dynamic-reference' : undefined}
              opacityFactor={localOpacity}
              logScalar={localLogScale}
            />
            
            {/* Frequency range indicator */}
            <div className="absolute top-2 right-3 bg-black/50 backdrop-blur-sm rounded-md py-1 px-2 text-xs">
              <div className="text-white font-medium mb-1">Frequency Range Impact</div>
              <div className="flex items-center space-x-1">
                <div className={`h-2 w-2 rounded-full ${maxFreq - minFreq > 500 ? 'bg-green-500' : maxFreq - minFreq > 100 ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
                <div className="text-[10px]">
                  {maxFreq - minFreq > 500 
                    ? 'Good parameter separation' 
                    : maxFreq - minFreq > 100 
                    ? 'Moderate separation' 
                    : 'Poor separation (highly underdetermined)'}
                </div>
              </div>
              <div className="text-[10px] text-neutral-300 mt-1">
                {minFreq.toFixed(1)} - {maxFreq.toFixed(1)} Hz
              </div>
            </div>
            
            {/* Data status indicator */}
            {!hasGridData && (
              <div className="absolute top-2 left-3 bg-black/50 backdrop-blur-sm rounded-md py-1 px-2 text-xs text-white">
                <div className="flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5 mr-1.5 text-blue-400">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" />
                  </svg>
                  No Grid Data
                </div>
                <div className="text-[10px] text-neutral-300 mt-1">
                  {referenceLayer ? "Reference model only" : "Compute a grid to visualize parameters"}
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Layer Controls - Takes 1/3 of the width on medium+ screens */}
        <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
          <h4 className="text-xs font-medium text-neutral-300 mb-3 flex items-center">
            <svg className="w-3.5 h-3.5 mr-1.5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            Visualization Layers
          </h4>
          
          <div className="space-y-3">
            {/* Reference Model Toggle - Directly uses local state */}
            <div 
              className={`border rounded-lg p-3 flex flex-col cursor-pointer transition-all ${
                !showReference ? 'opacity-50 border-dashed bg-neutral-800/20' : 'border-solid active'
              }`}
              style={{ borderColor: showReference ? '#FFFFFF' : '#4B5563' }}
              onClick={() => {
                // Toggle local reference visibility first
                setShowReference(!showReference);
                
                // Then trigger global toggle if needed
                if ((!showReference && !referenceModelId) || (showReference && referenceModelId)) {
                  // Only dispatch event if state needs to change
                  if (typeof window !== 'undefined') {
                    const event = new CustomEvent('toggleReferenceModel', {
                      detail: { forceState: !showReference }
                    });
                    window.dispatchEvent(event);
                  }
                }
              }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full mr-2 bg-white" />
                  <span className="text-xs font-medium">Reference Model</span>
                </div>
                <div className="text-neutral-500">
                  {!showReference ? (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                      <path d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
                    </svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                      <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                      <path fillRule="evenodd" d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
              </div>
              <div className="text-xs text-neutral-400 mt-2">
                {referenceModel ? "Current parameter values" : "Not available"}
              </div>
            </div>
            
            {/* Resnorm Group Toggles */}
            {resnormGroups.length > 0 ? (
              <div className="space-y-3 mt-4">
                <h5 className="text-xs font-medium text-neutral-300 mb-2">Grid Point Groups</h5>
                {resnormGroups.map((group, index) => {
                  // Get quality category for consistent labeling
                  const qualityCategory = getQualityCategory(group);
                  
                  return (
                    <div 
                      key={index}
                      className={`border rounded-lg p-3 flex flex-col cursor-pointer transition-all ${
                        hiddenGroups.includes(index) ? 'opacity-50 border-dashed bg-neutral-800/20' : 'border-solid active'
                      }`}
                      style={{ borderColor: hiddenGroups.includes(index) ? '#4B5563' : group.color }}
                      onClick={() => {
                        if (typeof window !== 'undefined') {
                          const event = new CustomEvent('toggleResnormGroup', {
                            detail: { groupIndex: index }
                          });
                          window.dispatchEvent(event);
                        }
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <span 
                            className="w-3 h-3 rounded-full mr-2"
                            style={{ backgroundColor: group.color }}
                          ></span>
                          <span className="text-xs font-medium">
                            {qualityCategory === 0 ? 'Very Good Fit' : 
                             qualityCategory === 1 ? 'Good Fit' : 
                             qualityCategory === 2 ? 'Moderate Fit' : 'Poor Fit'}
                          </span>
                        </div>
                        <span className="text-xs text-neutral-400">{group.items.length}</span>
                      </div>
                      
                      <div className="mt-2 text-[10px] text-neutral-500">
                        Resnorm: {group.range[0].toExponential(1)} - {group.range[1].toExponential(1)}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="mt-4 p-4 bg-neutral-800/20 rounded-lg border border-neutral-700 border-dashed">
                <div className="text-center">
                  <p className="text-sm text-neutral-400 font-medium">No Grid Data</p>
                  <p className="text-xs text-neutral-500 mt-1">
                    Compute a grid to see parameter groups
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Visualization Controls - Moved below the spider plot and layers */}
      <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
        <h4 className="text-xs font-medium text-neutral-300 mb-3 flex items-center">
          <svg className="w-3.5 h-3.5 mr-1.5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          Visualization Controls
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Opacity Slider */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <label className="text-xs text-neutral-400">Opacity Level</label>
              <span className="text-xs font-mono text-neutral-300">{localOpacity.toFixed(2)}</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="1" 
              step="0.05" 
              value={localOpacity}
              onChange={(e) => setLocalOpacity(parseFloat(e.target.value))}
              className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500">
              <span>Low Opacity</span>
              <span>High Opacity</span>
            </div>
          </div>
          
          {/* Scale Factor Slider */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <label className="text-xs text-neutral-400">Axis Scale Factor</label>
              <span className="text-xs font-mono text-neutral-300">{localLogScale.toFixed(1)}x</span>
            </div>
            <input 
              type="range" 
              min="0.5" 
              max="3" 
              step="0.1" 
              value={localLogScale}
              onChange={(e) => setLocalLogScale(parseFloat(e.target.value))}
              className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500">
              <span>Linear</span>
              <span>Logarithmic</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualizerTab; 
/**
 * PyCharm-Style Collapsible Bottom Panel
 * ======================================
 *
 * Provides a collapsible bottom panel similar to PyCharm's terminal/tools area
 * with tabbed interface for impedance data, circuit configuration, and diagnostics.
 */

import React, { useState, useCallback } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import {
  TableCellsIcon,
  CpuChipIcon,
  ArrowDownTrayIcon,
  ChartBarIcon,
  BeakerIcon,
  ChartPieIcon,
  BoltIcon
} from '@heroicons/react/24/outline';
import { ImpedanceComparisonTab } from './tabs/ImpedanceComparisonTab';
import { CircuitConfigurationTab } from './tabs/CircuitConfigurationTab';
import { ExportTab } from './tabs/ExportTab';
import { ResnormRangeTab } from './tabs/ResnormRangeTab';
import { AnalysisTab } from './tabs/AnalysisTab';
import { QuartileAnalysisTab } from './tabs/QuartileAnalysisTab';
import { TERTECAnalysisTab } from './tabs/TERTECAnalysisTab';
import { ModelSnapshot, ResnormGroup } from '../types';
import { CircuitParameters } from '../types/parameters';

export interface BottomPanelTab {
  id: string;
  label: string;
  icon?: React.ComponentType<{ className?: string }>;
  badge?: string | number;
  component: React.ComponentType<BottomPanelTabProps>;
}

export interface BottomPanelTabProps {
  gridResults: ModelSnapshot[];
  topConfigurations: ResnormGroup[];
  currentParameters: CircuitParameters;
  selectedConfigIndex: number;
  onConfigurationSelect: (index: number) => void;
  isVisible: boolean;
  highlightedModelId?: string | null;
  gridSize?: number;

  // Additional props for ResnormRangeTab
  currentResnorm?: number | null;
  onCurrentResnormChange?: (resnorm: number | null) => void;
  selectedResnormRange?: {min: number; max: number} | null;
  onResnormRangeChange?: (min: number, max: number) => void;
  onResnormSelect?: (resnorm: number) => void;
  navigationOffset?: number;
  onNavigationOffsetChange?: (offset: number) => void;
  navigationWindowSize?: number;
  taggedModels?: Map<string, string>;
  tagColors?: string[];

  // TER/TEC filtering props
  onTERTECFilterChange?: (filteredModelIds: string[]) => void;
}

interface CollapsibleBottomPanelProps {
  // Data props
  gridResults: ModelSnapshot[];
  topConfigurations: ResnormGroup[];
  currentParameters: CircuitParameters;
  selectedConfigIndex: number;
  onConfigurationSelect: (index: number) => void;
  highlightedModelId?: string | null;
  gridSize?: number;

  // Panel state props
  isCollapsed: boolean;
  onToggleCollapse: (collapsed: boolean) => void;
  height: number;
  onHeightChange: (height: number) => void;

  // Optional customization
  minHeight?: number;
  maxHeight?: number;
  className?: string;

  // Additional props for ResnormRangeTab
  currentResnorm?: number | null;
  onCurrentResnormChange?: (resnorm: number | null) => void;
  selectedResnormRange?: {min: number; max: number} | null;
  onResnormRangeChange?: (min: number, max: number) => void;
  onResnormSelect?: (resnorm: number) => void;
  navigationOffset?: number;
  onNavigationOffsetChange?: (offset: number) => void;
  navigationWindowSize?: number;
  taggedModels?: Map<string, string>;
  tagColors?: string[];

  // TER/TEC filtering props
  onTERTECFilterChange?: (filteredModelIds: string[]) => void;
}

const PRIMARY_TABS: BottomPanelTab[] = [
  {
    id: 'impedance',
    label: 'Data',
    icon: TableCellsIcon,
    component: ImpedanceComparisonTab
  },
  {
    id: 'configuration',
    label: 'Config',
    icon: CpuChipIcon,
    component: CircuitConfigurationTab
  },
  {
    id: 'analysis',
    label: 'Analysis',
    icon: BeakerIcon,
    component: AnalysisTab
  },
  {
    id: 'tertec',
    label: 'TER/TEC',
    icon: BoltIcon,
    component: TERTECAnalysisTab
  },
  {
    id: 'resnorm',
    label: 'Resnorm',
    icon: ChartBarIcon,
    component: ResnormRangeTab
  },
  {
    id: 'export',
    label: 'Export',
    icon: ArrowDownTrayIcon,
    component: ExportTab
  }
];

const SECONDARY_TABS: BottomPanelTab[] = [
  {
    id: 'quartile',
    label: 'Quartile',
    icon: ChartPieIcon,
    component: QuartileAnalysisTab
  }
];

const DEFAULT_TABS = [...PRIMARY_TABS, ...SECONDARY_TABS];

export const CollapsibleBottomPanel: React.FC<CollapsibleBottomPanelProps> = ({
  gridResults,
  topConfigurations,
  currentParameters,
  selectedConfigIndex,
  onConfigurationSelect,
  highlightedModelId,
  gridSize,
  isCollapsed,
  onToggleCollapse,
  height,
  onHeightChange,
  minHeight = 120,
  maxHeight = 800,
  className = '',
  // Additional props for ResnormRangeTab
  currentResnorm,
  onCurrentResnormChange,
  selectedResnormRange,
  onResnormRangeChange,
  onResnormSelect,
  navigationOffset,
  onNavigationOffsetChange,
  navigationWindowSize = 1000,
  taggedModels,
  tagColors,
  // TER/TEC filtering props
  onTERTECFilterChange
}) => {
  const [activeTab, setActiveTab] = useState('impedance');
  const [isResizing, setIsResizing] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);

  // Keyboard shortcuts removed per user request

  // Mouse drag resize handler
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);

    const startY = e.clientY;
    const startHeight = height;

    const handleMouseMove = (e: MouseEvent) => {
      const deltaY = startY - e.clientY; // Inverted because we're at the bottom
      const newHeight = Math.max(minHeight, Math.min(maxHeight, startHeight + deltaY));
      onHeightChange(newHeight);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [height, minHeight, maxHeight, onHeightChange]);

  const validActiveTab = DEFAULT_TABS.find(tab => tab.id === activeTab) ? activeTab : DEFAULT_TABS[0].id;
  if (validActiveTab !== activeTab) {
    setActiveTab(validActiveTab);
  }

  const sharedTabProps = {
    gridResults,
    topConfigurations,
    currentParameters,
    selectedConfigIndex,
    onConfigurationSelect,
    isVisible: !isCollapsed,
    highlightedModelId,
    gridSize,
    currentResnorm,
    onCurrentResnormChange,
    selectedResnormRange,
    onResnormRangeChange,
    onResnormSelect,
    navigationOffset,
    onNavigationOffsetChange,
    navigationWindowSize,
    taggedModels,
    tagColors,
    onTERTECFilterChange,
  };

  return (
    <div
      className={`bg-neutral-900 border-t border-neutral-800 flex flex-col flex-shrink-0 ${className} ${isCollapsed ? 'hidden' : ''}`}
      style={{
        height: isCollapsed ? '0px' : `${height}px`,
        transition: isCollapsed ? 'none' : 'height 0.2s ease-in-out'
      }}
    >
      {/* Resize Handle */}
      {!isCollapsed && (
        <div
          className={`h-1 bg-neutral-700 hover:bg-neutral-600 cursor-ns-resize transition-colors relative ${isResizing ? 'bg-primary' : ''}`}
          onMouseDown={handleMouseDown}
        >
          <button
            onClick={(e) => { e.stopPropagation(); onToggleCollapse(true); }}
            className="absolute right-4 top-0 bg-neutral-800 border border-neutral-600 rounded-t-md px-3 py-1 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700 transition-colors shadow-lg"
            title="Hide data panel"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      )}

      <Tabs.Root
        value={validActiveTab}
        onValueChange={setActiveTab}
        className="flex flex-col flex-1 min-h-0"
      >
        {/* Tab bar */}
        <div className="flex items-center bg-neutral-800 border-b border-neutral-800 px-2 h-9">
          <Tabs.List className="flex items-center gap-0.5 flex-1" aria-label="Data panel tabs">
            {PRIMARY_TABS.map(tab => {
              const IconComponent = tab.icon;
              return (
                <Tabs.Trigger
                  key={tab.id}
                  value={tab.id}
                  className="relative flex items-center gap-1.5 px-2.5 py-1.5 rounded transition-colors text-neutral-500 hover:text-neutral-200 hover:bg-neutral-700/50 data-[state=active]:bg-neutral-700 data-[state=active]:text-neutral-100 focus:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                >
                  {IconComponent && <IconComponent className="w-3.5 h-3.5 flex-shrink-0" />}
                  <span className="text-[10px] font-medium">{tab.label}</span>
                  {tab.badge && (
                    <span className="bg-red-500 text-white text-[9px] px-1 rounded-full min-w-[14px] text-center">
                      {tab.badge}
                    </span>
                  )}
                </Tabs.Trigger>
              );
            })}

            {/* More — secondary tabs */}
            <div className="relative">
              <button
                onClick={() => setMoreOpen(v => !v)}
                className={`flex items-center gap-1 px-2 py-1.5 rounded text-[10px] font-medium transition-colors ${
                  SECONDARY_TABS.some(t => t.id === validActiveTab)
                    ? 'bg-neutral-700 text-neutral-100'
                    : 'text-neutral-500 hover:text-neutral-200 hover:bg-neutral-700/50'
                }`}
              >
                More
                <svg className={`w-2.5 h-2.5 transition-transform ${moreOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {moreOpen && (
                <div className="absolute left-0 top-full mt-1 z-50 bg-surface border border-border rounded shadow-lg py-1 min-w-[100px]">
                  {SECONDARY_TABS.map(tab => {
                    const IconComponent = tab.icon;
                    return (
                      <button
                        key={tab.id}
                        onClick={() => { setActiveTab(tab.id); setMoreOpen(false); }}
                        className={`w-full flex items-center gap-2 px-3 py-1.5 text-xs transition-colors ${
                          validActiveTab === tab.id
                            ? 'text-neutral-100 bg-neutral-700'
                            : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800'
                        }`}
                      >
                        {IconComponent && <IconComponent className="w-3.5 h-3.5 flex-shrink-0" />}
                        {tab.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </Tabs.List>
        </div>

        {/* Content */}
        {!isCollapsed && (
          <div className="flex-1 overflow-y-auto bg-neutral-900 min-h-0">
            {DEFAULT_TABS.map(tab => (
              <Tabs.Content
                key={tab.id}
                value={tab.id}
                className="h-full focus:outline-none"
                forceMount
                hidden={validActiveTab !== tab.id}
              >
                <tab.component {...sharedTabProps} />
              </Tabs.Content>
            ))}
          </div>
        )}
      </Tabs.Root>
    </div>
  );
};

export default CollapsibleBottomPanel;
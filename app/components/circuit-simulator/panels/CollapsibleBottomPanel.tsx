/**
 * PyCharm-Style Collapsible Bottom Panel
 * ======================================
 *
 * Provides a collapsible bottom panel similar to PyCharm's terminal/tools area
 * with tabbed interface for impedance data, circuit configuration, and diagnostics.
 */

import React, { useState, useCallback } from 'react';
import {
  ChevronDownIcon,
  TableCellsIcon,
  CpuChipIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline';
import { ImpedanceComparisonTab } from './tabs/ImpedanceComparisonTab';
import { CircuitConfigurationTab } from './tabs/CircuitConfigurationTab';
import { ExportTab } from './tabs/ExportTab';
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
}

const DEFAULT_TABS: BottomPanelTab[] = [
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
    id: 'export',
    label: 'Export',
    icon: ArrowDownTrayIcon,
    component: ExportTab
  }
];

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
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<string>('impedance');
  const [isResizing, setIsResizing] = useState(false);

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

  // Ensure active tab exists
  const validActiveTab = DEFAULT_TABS.find(tab => tab.id === activeTab) ? activeTab : DEFAULT_TABS[0].id;
  if (validActiveTab !== activeTab) {
    setActiveTab(validActiveTab);
  }

  const ActiveTabComponent = DEFAULT_TABS.find(tab => tab.id === validActiveTab)?.component;

  return (
    <div
      className={`bg-neutral-900 border-t border-neutral-700 flex flex-col flex-shrink-0 ${className} ${isCollapsed ? 'hidden' : ''}`}
      style={{
        height: isCollapsed ? '0px' : `${height}px`,
        transition: isCollapsed ? 'none' : 'height 0.2s ease-in-out'
      }}
    >
      {/* Resize Handle with Collapse Caret */}
      {!isCollapsed && (
        <div
          className={`h-1 bg-neutral-700 hover:bg-neutral-600 cursor-ns-resize transition-colors relative ${
            isResizing ? 'bg-orange-500' : ''
          }`}
          onMouseDown={handleMouseDown}
        >
          {/* Collapse Caret Button integrated into resize handle */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggleCollapse(true);
            }}
            className="absolute right-4 top-0 bg-neutral-800 border border-neutral-600 rounded-t-md px-3 py-1 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700 transition-colors shadow-lg"
            title="Hide data panel"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      )}

      {/* Header with tabs and controls */}
      <div className="flex items-center justify-between bg-neutral-800 border-b border-neutral-700 px-3 py-1 min-h-[40px]">
        {/* Left side - Icon-based Tabs */}
        <div className="flex items-center space-x-1">
          {DEFAULT_TABS.map(tab => {
            const isActive = tab.id === validActiveTab;
            const IconComponent = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-2 text-xs font-medium rounded transition-colors flex items-center justify-center min-w-[40px] ${
                  isActive
                    ? 'bg-orange-600 text-white'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50'
                }`}
                title={tab.label}
              >
                {IconComponent && <IconComponent className="w-4 h-4" />}
                {tab.badge && (
                  <span className="bg-red-500 text-white text-xs px-1.5 py-0.5 rounded-full min-w-[16px] text-center absolute -top-1 -right-1">
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Right side - Controls */}
        <div className="flex items-center space-x-2">
          {/* Height indicator when not collapsed */}
          {!isCollapsed && (
            <span className="text-xs text-neutral-500 hidden sm:block">
              {height}px
            </span>
          )}

          {/* Collapse button when expanded */}
          {!isCollapsed && (
            <button
              onClick={() => onToggleCollapse(true)}
              className="p-1 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700 rounded transition-colors"
              title="Hide data panel"
            >
              <ChevronDownIcon className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Content area - Enhanced scrollable container */}
      {!isCollapsed && (
        <div className="flex-1 overflow-y-auto bg-neutral-900 min-h-0">
          {ActiveTabComponent && (
            <ActiveTabComponent
              gridResults={gridResults}
              topConfigurations={topConfigurations}
              currentParameters={currentParameters}
              selectedConfigIndex={selectedConfigIndex}
              onConfigurationSelect={onConfigurationSelect}
              isVisible={!isCollapsed}
              highlightedModelId={highlightedModelId}
              gridSize={gridSize}
            />
          )}
        </div>
      )}
    </div>
  );
};

export default CollapsibleBottomPanel;
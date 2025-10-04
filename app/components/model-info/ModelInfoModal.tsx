/**
 * Model Info Modal Component
 * ==========================
 *
 * Displays circuit model equations, methodology, and parameter information.
 * Similar to Settings modal but focused on mathematical documentation.
 */

'use client';

import React, { useState } from 'react';
import {
  XMarkIcon,
  BeakerIcon,
  CalculatorIcon,
  ChartBarIcon,
  BookOpenIcon
} from '@heroicons/react/24/outline';
import { CircuitParameters } from '../circuit-simulator/types/parameters';
import { CircuitModelSection } from '../circuit-simulator/math/CircuitModelSection';
import { CoreEquationsSection } from '../circuit-simulator/math/CoreEquationsSection';
import { ComplexImpedanceSection } from '../circuit-simulator/math/ComplexImpedanceSection';
import { ResnormMethodsSection } from '../circuit-simulator/math/ResnormMethodsSection';

interface ModelInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  parameters?: CircuitParameters;
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

function TabButton({ active, onClick, icon, label }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-3 px-4 py-3 text-sm font-medium rounded-xl transition-all duration-200 w-full ${
        active
          ? 'bg-orange-600 text-white shadow-lg'
          : 'text-neutral-300 hover:text-white hover:bg-neutral-700'
      }`}
    >
      <div className="w-5 h-5">{icon}</div>
      <span>{label}</span>
    </button>
  );
}

export const ModelInfoModal: React.FC<ModelInfoModalProps> = ({
  isOpen,
  onClose
}) => {
  const [activeTab, setActiveTab] = useState<'circuit' | 'equations' | 'impedance' | 'resnorm'>('circuit');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div
        className="bg-neutral-900 rounded-2xl shadow-2xl border border-neutral-700 w-full max-w-6xl max-h-[90vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-700 bg-neutral-800/50">
          <div className="flex items-center space-x-3">
            <BeakerIcon className="w-6 h-6 text-orange-500" />
            <h2 className="text-xl font-bold text-neutral-100">Circuit Model Documentation</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-700 rounded-lg transition-colors"
            title="Close"
          >
            <XMarkIcon className="w-6 h-6 text-neutral-400" />
          </button>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar with tabs */}
          <div className="w-64 bg-neutral-800/30 border-r border-neutral-700 p-4 space-y-2 overflow-y-auto">
            <div className="mb-4">
              <h3 className="text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-3">
                Documentation
              </h3>
            </div>

            <TabButton
              active={activeTab === 'circuit'}
              onClick={() => setActiveTab('circuit')}
              icon={<BeakerIcon className="w-5 h-5" />}
              label="Circuit Topology"
            />

            <TabButton
              active={activeTab === 'equations'}
              onClick={() => setActiveTab('equations')}
              icon={<CalculatorIcon className="w-5 h-5" />}
              label="Core Equations"
            />

            <TabButton
              active={activeTab === 'impedance'}
              onClick={() => setActiveTab('impedance')}
              icon={<ChartBarIcon className="w-5 h-5" />}
              label="Impedance Analysis"
            />

            <TabButton
              active={activeTab === 'resnorm'}
              onClick={() => setActiveTab('resnorm')}
              icon={<BookOpenIcon className="w-5 h-5" />}
              label="Resnorm Methods"
            />

            {/* Info Box */}
            <div className="mt-6 p-4 bg-blue-900/20 border border-blue-700/30 rounded-lg">
              <div className="flex items-start mb-2">
                <svg className="w-4 h-4 mr-2 text-blue-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p className="text-xs font-medium text-blue-200 mb-1">Quick Tip</p>
                  <p className="text-xs text-blue-300/80">
                    Visit the <strong>Data</strong> tab in the bottom panel and click &quot;Show Math&quot; to see step-by-step calculations.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Main content area */}
          <div className="flex-1 overflow-y-auto p-6">
            {activeTab === 'circuit' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-neutral-200 mb-2">Circuit Topology</h3>
                  <p className="text-neutral-400 mb-6">
                    Modified Randles equivalent circuit model for retinal pigment epithelium
                  </p>
                </div>
                <CircuitModelSection />
              </div>
            )}

            {activeTab === 'equations' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-neutral-200 mb-2">Core Equations</h3>
                  <p className="text-neutral-400 mb-6">
                    Fundamental equations governing circuit behavior
                  </p>
                </div>
                <CoreEquationsSection />
              </div>
            )}

            {activeTab === 'impedance' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-neutral-200 mb-2">Complex Impedance Analysis</h3>
                  <p className="text-neutral-400 mb-6">
                    Frequency-dependent impedance calculations
                  </p>
                </div>
                <ComplexImpedanceSection />
              </div>
            )}

            {activeTab === 'resnorm' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-neutral-200 mb-2">Resnorm Calculation Methods</h3>
                  <p className="text-neutral-400 mb-6">
                    Model fitting quality assessment using residual norms
                  </p>
                </div>
                <ResnormMethodsSection />
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-neutral-700 bg-neutral-800/30 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-neutral-700 hover:bg-neutral-600 text-neutral-200 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelInfoModal;

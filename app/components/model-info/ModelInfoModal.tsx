'use client';

import React from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { CircuitModelSection } from '../circuit-simulator/math/CircuitModelSection';
import { CoreEquationsSection } from '../circuit-simulator/math/CoreEquationsSection';
import { ComplexImpedanceSection } from '../circuit-simulator/math/ComplexImpedanceSection';

interface ModelInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ModelInfoModal: React.FC<ModelInfoModalProps> = ({ isOpen, onClose }) => (
  <Dialog.Root open={isOpen} onOpenChange={(open) => !open && onClose()}>
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 bg-black/60 z-50" />
      <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-[#0d0d0f] border border-neutral-800 rounded-lg w-full max-w-3xl max-h-[88vh] flex flex-col overflow-hidden focus:outline-none">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-neutral-800 flex-shrink-0">
          <Dialog.Title className="text-sm font-medium text-neutral-200">
            RPE Epithelium — Circuit Model
          </Dialog.Title>
          <Dialog.Close asChild>
            <button className="w-7 h-7 flex items-center justify-center rounded text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800 transition-colors" title="Close">
              <XMarkIcon className="w-4 h-4" />
            </button>
          </Dialog.Close>
        </div>

        {/* Scrollable physics content */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-8">
          <section>
            <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-widest mb-4">Circuit Topology</h2>
            <CircuitModelSection />
          </section>

          <div className="border-t border-neutral-800"/>

          <section>
            <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-widest mb-4">Core Equations</h2>
            <CoreEquationsSection />
          </section>

          <div className="border-t border-neutral-800"/>

          <section>
            <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-widest mb-4">Impedance Analysis</h2>
            <ComplexImpedanceSection />
          </section>
        </div>

      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>
);

export default ModelInfoModal;

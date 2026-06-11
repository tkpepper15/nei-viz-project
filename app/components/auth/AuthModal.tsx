"use client";

import React from 'react';
import * as Dialog from '@radix-ui/react-dialog';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAuthSuccess?: (user: unknown) => void;
}

// Auth is not needed in self-hosted mode — this is a no-op stub.
export const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose }) => {
  return (
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-neutral-900 border border-neutral-800 rounded-lg p-6 max-w-sm w-full mx-4 focus:outline-none">
          <Dialog.Title className="text-white font-bold text-lg mb-2">Self-Hosted Mode</Dialog.Title>
          <Dialog.Description className="text-neutral-400 text-sm mb-4">
            Authentication is disabled. Running as local user.
          </Dialog.Description>
          <Dialog.Close asChild>
            <button className="w-full py-2 bg-primary hover:bg-primary-dark text-white rounded text-sm font-medium transition-colors">
              Close
            </button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};

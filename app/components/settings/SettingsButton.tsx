'use client';

import React from 'react';
import { CogIcon } from '@heroicons/react/24/outline';

interface SettingsButtonProps {
  onClick: () => void;
}

export function SettingsButton({ onClick }: SettingsButtonProps) {
  return (
    <button
      onClick={onClick}
      className="w-8 h-8 rounded-full bg-neutral-700 hover:bg-neutral-600 flex items-center justify-center text-white transition-all hover:ring-2 hover:ring-white/20"
      title="Settings"
    >
      <CogIcon className="h-4 w-4" />
    </button>
  );
}
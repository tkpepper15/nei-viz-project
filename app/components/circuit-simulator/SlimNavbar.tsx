/**
 * Slim Navbar Component
 * ====================
 *
 * Ultra-compact navigation bar with essential functionality:
 * - Status indicator
 * - Grid computation info
 * - Settings access
 * - User profile
 *
 * Designed to take minimal space while maintaining core functionality.
 */

import React from 'react';
import { UserProfile } from '../auth/UserProfile';
import { User } from '@supabase/supabase-js';

interface SlimNavbarProps {
  statusMessage?: string;
  gridResults?: unknown[];
  gridSize?: number;
  onSettingsOpen: () => void;
  user?: User | null;
  onSignOut?: () => void;
}

export const SlimNavbar: React.FC<SlimNavbarProps> = ({
  statusMessage,
  gridResults = [],
  gridSize = 9,
  onSettingsOpen,
  user,
  onSignOut
}) => {
  const computedCount = gridResults.length;
  const totalPossible = Math.pow(gridSize, 5);

  return (
    <div className="h-12 px-4 flex items-center justify-between bg-neutral-900 flex-shrink-0">
      {/* Left side: Status and grid info */}
      <div className="flex items-center gap-4 text-xs">
        {/* Status and grid info */}
        <div className="flex items-center gap-3">
          {statusMessage && (
            <div className="flex items-center gap-1.5 text-neutral-400">
              <div className="w-1.5 h-1.5 bg-orange-400 rounded-full animate-pulse"></div>
              <span className="truncate max-w-48">{statusMessage}</span>
            </div>
          )}



          {computedCount > 0 && (
            <div className="text-neutral-500 font-mono">
              {computedCount.toLocaleString()} computed
            </div>

          )}
        </div>
      </div>

      {/* Right side: Settings and user */}
      <div className="flex items-center gap-2">
        <button
          onClick={onSettingsOpen}
          className="px-2 py-1 bg-neutral-800 hover:bg-neutral-700 rounded text-neutral-400 hover:text-neutral-200 transition-colors flex items-center"
          title="Settings"
        >
          <svg className="w-4 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>

        {user ? (
          <div className="hover:bg-neutral-700 rounded flex items-center">
            <UserProfile
              user={user}
              onSignOut={onSignOut || (() => {})}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
};
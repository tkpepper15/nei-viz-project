import React from 'react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { SavedProfile } from '../types/savedProfiles';

interface ProfileCardProps {
  profile: SavedProfile;
  isSelected: boolean;
  onDelete: () => void;
  onCompute: () => void;
  onEdit: () => void;
  onEditParameters?: () => void;
  onCopyParams?: () => void;
  isMultiSelectMode?: boolean;
  isSelectedForDelete?: boolean;
  onToggleSelect?: () => void;
  onClick?: () => void;
  isComputing?: boolean;
  onFetchServerData?: () => void;
  serverConnectionStatus?: 'online' | 'offline' | 'checking';
}

export const ProfileCard: React.FC<ProfileCardProps> = ({
  profile,
  isSelected,
  onDelete,
  onCompute,
  onEdit,
  onEditParameters,
  onCopyParams,
  isMultiSelectMode = false,
  isSelectedForDelete = false,
  onToggleSelect,
  onClick,
  isComputing = false
}) => {
  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / 86400000);
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div
      className={`group relative rounded transition-colors cursor-pointer ${
        isSelected
          ? 'bg-neutral-800 border border-neutral-800'
          : 'border border-transparent hover:bg-neutral-800/60'
      } ${isSelectedForDelete ? 'ring-1 ring-primary' : ''} ${
        isComputing ? 'ring-1 ring-primary/40' : ''
      } ${isMultiSelectMode ? 'cursor-pointer' : ''}`}
      onClick={isMultiSelectMode ? onToggleSelect : onClick}
    >
      <div className="flex items-start p-2.5 gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-1">
            <span className="text-xs font-medium text-neutral-200 truncate">{profile.name}</span>
            {isComputing && (
              <span className="text-xs text-neutral-500 shrink-0">computing</span>
            )}
          </div>

          <div className="flex items-center gap-1.5 text-xs text-neutral-500">
            <span className="font-mono">{profile.gridSize}⁵</span>
            <span>·</span>
            <span>{profile.minFreq}–{profile.maxFreq} Hz</span>
            <span>·</span>
            <span>{formatDate(profile.created)}</span>
          </div>

          {profile.description && (
            <p className="text-xs text-neutral-600 truncate mt-0.5">{profile.description}</p>
          )}
        </div>

        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <button
              onClick={(e) => e.stopPropagation()}
              className="w-6 h-6 flex items-center justify-center rounded hover:bg-neutral-700 transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100 focus:outline-none flex-shrink-0 mt-0.5"
            >
              <svg className="w-3.5 h-3.5 text-neutral-400" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
              </svg>
            </button>
          </DropdownMenu.Trigger>

          <DropdownMenu.Portal>
            <DropdownMenu.Content
              className="w-36 bg-surface border border-border rounded shadow-lg py-1 z-50 focus:outline-none"
              side="bottom"
              align="end"
              sideOffset={4}
              onClick={(e) => e.stopPropagation()}
            >
              <DropdownMenu.Item
                onSelect={onCompute}
                className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-neutral-300 hover:bg-neutral-700 hover:text-neutral-100 cursor-pointer focus:outline-none focus:bg-neutral-700"
              >
                <svg className="w-3.5 h-3.5 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Compute
              </DropdownMenu.Item>

              {onEditParameters && (
                <DropdownMenu.Item
                  onSelect={onEditParameters}
                  className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-neutral-300 hover:bg-neutral-700 hover:text-neutral-100 cursor-pointer focus:outline-none focus:bg-neutral-700"
                >
                  <svg className="w-3.5 h-3.5 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                  </svg>
                  Edit Parameters
                </DropdownMenu.Item>
              )}

              <DropdownMenu.Item
                onSelect={onEdit}
                className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-neutral-300 hover:bg-neutral-700 hover:text-neutral-100 cursor-pointer focus:outline-none focus:bg-neutral-700"
              >
                <svg className="w-3.5 h-3.5 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                Rename
              </DropdownMenu.Item>

              {onCopyParams && (
                <DropdownMenu.Item
                  onSelect={onCopyParams}
                  className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-neutral-300 hover:bg-neutral-700 hover:text-neutral-100 cursor-pointer focus:outline-none focus:bg-neutral-700"
                >
                  <svg className="w-3.5 h-3.5 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                  </svg>
                  Copy Params
                </DropdownMenu.Item>
              )}

              <DropdownMenu.Separator className="h-px bg-border my-1" />

              <DropdownMenu.Item
                onSelect={() => {
                  if (typeof window !== 'undefined' && window.confirm(`Delete "${profile.name}"?`)) {
                    onDelete();
                  }
                }}
                className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-danger hover:bg-neutral-700 cursor-pointer focus:outline-none focus:bg-neutral-700"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Portal>
        </DropdownMenu.Root>
      </div>
    </div>
  );
};

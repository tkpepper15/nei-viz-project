import React, { useState, useEffect } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { SavedProfile } from '../types/savedProfiles';

interface EditProfileModalProps {
  profile: SavedProfile | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (profileId: string, name: string, description?: string) => void;
}

export const EditProfileModal: React.FC<EditProfileModalProps> = ({
  profile,
  isOpen,
  onClose,
  onSave
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (profile) {
      setName(profile.name);
      setDescription(profile.description || '');
    }
  }, [profile]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!profile || !name.trim()) return;
    setIsSubmitting(true);
    try {
      onSave(profile.id, name.trim(), description.trim() || undefined);
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog.Root open={isOpen && !!profile} onOpenChange={(open) => !open && !isSubmitting && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-surface border border-border rounded w-full max-w-sm mx-4 focus:outline-none">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <Dialog.Title className="text-sm font-medium text-neutral-200">Edit Circuit</Dialog.Title>
            <Dialog.Close asChild>
              <button
                disabled={isSubmitting}
                className="w-7 h-7 flex items-center justify-center rounded hover:bg-neutral-700 transition-colors text-neutral-500 hover:text-neutral-300 disabled:opacity-40"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </Dialog.Close>
          </div>

          <form onSubmit={handleSubmit} className="p-4 space-y-3">
            <div>
              <label htmlFor="edit-profile-name" className="block text-xs font-medium text-neutral-400 mb-1.5">
                Name
              </label>
              <input
                id="edit-profile-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Circuit name"
                maxLength={50}
                disabled={isSubmitting}
                className="w-full px-3 py-1.5 text-sm bg-neutral-800 border border-neutral-800 rounded text-neutral-200 placeholder-neutral-600 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary disabled:opacity-40 transition-colors"
                required
              />
              <div className="text-xs text-neutral-600 mt-1">{name.length}/50</div>
            </div>

            <div>
              <label htmlFor="edit-profile-description" className="block text-xs font-medium text-neutral-400 mb-1.5">
                Description
              </label>
              <textarea
                id="edit-profile-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Optional description"
                maxLength={200}
                disabled={isSubmitting}
                rows={3}
                className="w-full px-3 py-1.5 text-sm bg-neutral-800 border border-neutral-800 rounded text-neutral-200 placeholder-neutral-600 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary disabled:opacity-40 resize-none transition-colors"
              />
              <div className="text-xs text-neutral-600 mt-1">{description.length}/200</div>
            </div>

            {profile && (
              <div className="bg-neutral-800/50 border border-neutral-800 rounded p-2.5 text-xs text-neutral-500">
                {profile.gridSize}⁵ grid · {profile.minFreq}–{profile.maxFreq} Hz
              </div>
            )}

            <div className="flex gap-2 pt-1">
              <Dialog.Close asChild>
                <button
                  type="button"
                  disabled={isSubmitting}
                  className="flex-1 px-3 py-1.5 text-xs bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded border border-neutral-800 transition-colors disabled:opacity-40"
                >
                  Cancel
                </button>
              </Dialog.Close>
              <button
                type="submit"
                disabled={isSubmitting || !name.trim()}
                className="flex-1 px-3 py-1.5 text-xs bg-primary hover:bg-primary-dark text-white rounded transition-colors disabled:opacity-40 flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-3 h-3 border border-white/30 border-t-white rounded-full animate-spin" />
                    Saving...
                  </>
                ) : 'Save'}
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};

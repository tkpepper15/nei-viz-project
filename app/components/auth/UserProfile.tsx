"use client";

import React from 'react';

interface UserProfileProps {
  user: { id: string; email: string } | null;
  onSignOut?: () => void;
}

export const UserProfile: React.FC<UserProfileProps> = ({ user }) => {
  const initial = user?.email?.[0]?.toUpperCase() ?? 'U';
  return (
    <button
      className="w-8 h-8 rounded-full bg-neutral-700 hover:bg-neutral-600 flex items-center justify-center text-xs text-neutral-200 font-medium transition-colors"
      title={user?.email ?? 'User'}
    >
      {initial}
    </button>
  );
};

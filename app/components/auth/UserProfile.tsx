"use client";

import React, { useState, useRef, useEffect } from 'react';
import { supabase } from '../../../lib/supabase';
import { User } from '@supabase/supabase-js';

interface UserProfileProps {
  user: User;
  onSignOut: () => void;
}

export const UserProfile: React.FC<UserProfileProps> = ({ user, onSignOut }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    onSignOut();
    setIsOpen(false);
  };

  // Get user initials for avatar
  const getInitials = (email: string) => {
    return email.charAt(0).toUpperCase();
  };

  // Generate consistent color from email
  const getAvatarColor = (email: string) => {
    let hash = 0;
    for (let i = 0; i < email.length; i++) {
      hash = email.charCodeAt(i) + ((hash << 5) - hash);
    }
    const colors = [
      'bg-blue-500',
      'bg-green-500', 
      'bg-purple-500',
      'bg-pink-500',
      'bg-indigo-500',
      'bg-yellow-500',
      'bg-red-500',
      'bg-teal-500'
    ];
    return colors[Math.abs(hash) % colors.length];
  };

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Profile circle button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-8 h-8 rounded flex items-center justify-center text-white text-sm font-medium hover:ring-2 hover:ring-white/20 transition-all ${getAvatarColor(user.email || '')}`}
      >
        {getInitials(user.email || '')}
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-neutral-800 border border-neutral-700 rounded-lg shadow-xl z-50">
          <div className="p-3 border-b border-neutral-700">
            <div className="text-sm font-medium text-white truncate">
              {user.email || 'Anonymous User'}
            </div>
            <div className="text-xs text-neutral-400">
              Signed in
            </div>
          </div>
          
          <div className="p-1">
            <button
              onClick={handleSignOut}
              className="w-full px-3 py-2 text-left text-sm text-neutral-300 hover:text-white hover:bg-neutral-700 rounded flex items-center"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
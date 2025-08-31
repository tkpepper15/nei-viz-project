"use client"

import React, { useState } from 'react'
import { useAuth } from './AuthProvider'
// import { useSimpleSessionManagement } from '../../hooks/useSimpleSessionManagement' // Temporarily disabled for deployment
import { CogIcon, PlusIcon, FolderIcon, TagIcon } from '@heroicons/react/24/outline'
import { UserSelector } from './UserSelector'

export const SessionNavbar: React.FC = () => {
  const { user } = useAuth()
  // const { sessionState, actions, isReady } = useSimpleSessionManagement() // Temporarily disabled for deployment
  const sessionState = { 
    sessionName: null, 
    savedProfiles: [], 
    visualizationSettings: { visualizationType: 'spider3d' },
    performanceSettings: { maxWorkers: 4 },
    environment: { memoryLimit: '8GB' },
    taggedModels: []
  }; 
  const isReady = false;
  const actions = {
    createSession: (name: string) => console.log('Session creation disabled for deployment:', name),
    saveConfiguration: (config: Record<string, unknown>) => console.log('Configuration saving disabled for deployment:', config)
  };
  const [showSessionMenu, setShowSessionMenu] = useState(false)

  if (!user) {
    return (
      <div className="bg-gray-900 text-white px-3 py-2 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-1.5 h-1.5 bg-red-500 rounded-full"></div>
          <span className="text-xs">Please log in to access session management</span>
        </div>
      </div>
    )
  }

  if (!isReady) {
    return (
      <div className="bg-gray-900 text-white px-3 py-2 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-1.5 h-1.5 bg-yellow-500 rounded-full animate-pulse"></div>
          <span className="text-xs">Loading session...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 text-white px-3 py-2 flex items-center justify-between border-b border-gray-700">
      {/* Session Status */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <div className="w-1.5 h-1.5 bg-green-500 rounded-full"></div>
          <span className="text-xs font-medium">Connected</span>
        </div>
        
        {/* Session Info */}
        <div className="relative">
          <button
            onClick={() => setShowSessionMenu(!showSessionMenu)}
            className="flex items-center space-x-1.5 hover:bg-gray-800 px-2 py-1.5 rounded-md transition-colors"
          >
            <FolderIcon className="h-3.5 w-3.5" />
            <span className="text-xs font-medium">{sessionState.sessionName}</span>
            <svg 
              className={`w-3 h-3 transition-transform ${showSessionMenu ? 'rotate-180' : ''}`} 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Session Menu */}
          {showSessionMenu && (
            <div className="absolute top-full left-0 mt-1 w-80 bg-gray-800 rounded-md shadow-lg border border-gray-700 z-50">
              <div className="p-4">
                <div className="space-y-3">
                  {/* Session Info */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-300">Current Session</h3>
                    <p className="text-xs text-gray-400 mt-1">{sessionState.sessionName}</p>
                  </div>

                  {/* Saved Profiles Count */}
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>Saved Profiles</span>
                    <span className="bg-gray-700 px-2 py-1 rounded">
                      {sessionState.savedProfiles?.length || 0}
                    </span>
                  </div>

                  {/* Session Settings */}
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Visualization</span>
                      <span className="text-green-400">
                        {sessionState.visualizationSettings?.visualizationType || 'spider3d'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Workers</span>
                      <span className="text-blue-400">
                        {sessionState.performanceSettings?.maxWorkers || 4}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Memory Limit</span>
                      <span className="text-blue-400">
                        {sessionState.environment?.memoryLimit || '8GB'}
                      </span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="pt-2 border-t border-gray-700">
                    <button
                      onClick={() => {
                        actions.createSession(`Session ${new Date().toLocaleDateString()}`)
                        setShowSessionMenu(false)
                      }}
                      className="flex items-center space-x-2 w-full text-left hover:bg-gray-700 px-2 py-2 rounded text-xs"
                    >
                      <PlusIcon className="h-3 w-3" />
                      <span>New Session</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right side - User actions */}
      <div className="flex items-center space-x-3">
        {/* Tagged Models Indicator */}
        <div className="flex items-center space-x-1.5 text-xs text-gray-400">
          <TagIcon className="h-3.5 w-3.5" />
          <span>Tagged: {sessionState.savedProfiles?.length || 0}</span>
        </div>

        {/* Settings */}
        <button className="hover:bg-gray-800 p-1.5 rounded-md transition-colors">
          <CogIcon className="h-3.5 w-3.5" />
        </button>

        {/* User Selector */}
        <UserSelector 
          currentUserId={user?.id}
          onUserChange={(userId) => {
            console.log('Selected user:', userId)
            // Here you could implement user switching logic
            // For demo purposes, we'll just log it
          }}
        />
      </div>
    </div>
  )
}
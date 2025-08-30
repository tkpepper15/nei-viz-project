"use client"

import React, { useState } from 'react'
import { ChevronDownIcon, UserIcon } from '@heroicons/react/24/outline'

interface DemoUser {
  id: string
  name: string
  role: string
  sessions: number
  taggedModels: number
}

const DEMO_USERS: DemoUser[] = [
  {
    id: '11111111-1111-1111-1111-111111111111',
    name: 'Dr. Sarah Chen',
    role: 'Principal Investigator',
    sessions: 3,
    taggedModels: 12
  },
  {
    id: '22222222-2222-2222-2222-222222222222', 
    name: 'Alex Rodriguez',
    role: 'PhD Student',
    sessions: 2,
    taggedModels: 8
  },
  {
    id: '33333333-3333-3333-3333-333333333333',
    name: 'Prof. Maria Gonzalez',
    role: 'Lab Director',
    sessions: 5,
    taggedModels: 24
  }
]

interface UserSelectorProps {
  currentUserId?: string
  onUserChange?: (userId: string) => void
}

export const UserSelector: React.FC<UserSelectorProps> = ({ 
  currentUserId, 
  onUserChange 
}) => {
  const [isOpen, setIsOpen] = useState(false)
  
  const currentUser = DEMO_USERS.find(u => u.id === currentUserId) || DEMO_USERS[0]

  const handleUserSelect = (userId: string) => {
    onUserChange?.(userId)
    setIsOpen(false)
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 hover:bg-gray-800 px-3 py-2 rounded-md transition-colors min-w-[120px]"
      >
        <UserIcon className="h-4 w-4 flex-shrink-0" />
        <div className="flex flex-col items-start min-w-0">
          <span className="text-xs font-medium truncate">{currentUser.name}</span>
          <span className="text-xs text-gray-400 truncate">{currentUser.role}</span>
        </div>
        <ChevronDownIcon 
          className={`h-3 w-3 flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} 
        />
      </button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-1 w-72 bg-gray-800 rounded-md shadow-lg border border-gray-700 z-50">
          <div className="p-2">
            <div className="text-xs text-gray-400 px-2 py-1 border-b border-gray-700 mb-2">
              Select User
            </div>
            {DEMO_USERS.map((user) => (
              <button
                key={user.id}
                onClick={() => handleUserSelect(user.id)}
                className={`w-full text-left p-3 rounded-md transition-colors hover:bg-gray-700 ${
                  user.id === currentUserId ? 'bg-gray-700 border border-blue-500' : ''
                }`}
              >
                <div className="flex items-start space-x-3">
                  <UserIcon className="h-5 w-5 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-white">{user.name}</span>
                      {user.id === currentUserId && (
                        <span className="text-xs bg-blue-600 px-2 py-0.5 rounded-full">Current</span>
                      )}
                    </div>
                    <p className="text-xs text-gray-400 mt-0.5">{user.role}</p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                      <span>{user.sessions} sessions</span>
                      <span>{user.taggedModels} tagged models</span>
                    </div>
                  </div>
                </div>
              </button>
            ))}
            
            <div className="border-t border-gray-700 mt-2 pt-2">
              <button
                onClick={() => setIsOpen(false)}
                className="w-full text-left p-2 rounded-md hover:bg-gray-700 text-xs text-gray-400"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
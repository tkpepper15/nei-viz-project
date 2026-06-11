"use client"

import React, { createContext, useContext, useEffect, useState } from 'react'

// Minimal local user type — matches what the app reads from `user`
export interface LocalUser {
  id: string
  email: string
  is_anonymous: false
}

interface AuthContextType {
  user: LocalUser | null
  loading: boolean
  signInWithEmail: (email: string, password: string) => Promise<{ error: Error | null }>
  signUpWithEmail: (email: string, password: string) => Promise<{ error: Error | null }>
  signOut: () => Promise<{ error: Error | null }>
  signInAnonymously: () => Promise<{ error: Error | null }>
}

const LOCAL_USER: LocalUser = { id: 'local-user', email: 'local@localhost', is_anonymous: false }

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<LocalUser | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Always use the local user — no network calls needed
    setUser(LOCAL_USER)
    setLoading(false)
  }, [])

  const noop = async () => ({ error: null })

  const value: AuthContextType = {
    user,
    loading,
    signInWithEmail: noop,
    signUpWithEmail: noop,
    signOut: async () => { setUser(null); return { error: null } },
    signInAnonymously: async () => { setUser(LOCAL_USER); return { error: null } },
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

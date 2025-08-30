"use client";

import React, { useState } from 'react';
import { DatabaseSetup } from '../../lib/databaseSetup';

interface DiagnosticResult {
  connection: {
    success: boolean;
    error?: string;
    details?: unknown;
  };
  table: {
    exists: boolean;
    canAccess?: boolean;
    error?: string;
  };
  profileTest?: {
    success: boolean;
    error?: string;
    profileId?: string;
  };
  recommendations: string[];
}

export const DatabaseDiagnostics: React.FC = () => {
  const [diagnostics, setDiagnostics] = useState<DiagnosticResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const runDiagnostics = async () => {
    setIsRunning(true);
    try {
      const result = await DatabaseSetup.runDiagnostics();
      setDiagnostics(result);
    } catch (error) {
      console.error('Diagnostics failed:', error);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-neutral-800 border border-neutral-700 rounded-lg p-4 m-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-white">Database Diagnostics</h3>
        <button
          onClick={runDiagnostics}
          disabled={isRunning}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-600 text-white rounded-md text-sm font-medium transition-colors"
        >
          {isRunning ? 'Running...' : 'Run Diagnostics'}
        </button>
      </div>

      {diagnostics && (
        <div className="space-y-4">
          {/* Connection Status */}
          <div className="bg-neutral-900 rounded-lg p-3">
            <h4 className="font-medium text-neutral-200 mb-2">Connection Status</h4>
            <div className={`text-sm ${diagnostics.connection.success ? 'text-green-400' : 'text-red-400'}`}>
              {diagnostics.connection.success ? '✅ Connected' : '❌ Connection Failed'}
            </div>
            {diagnostics.connection.details && (
              <pre className="text-xs text-neutral-400 mt-2 overflow-x-auto">
                {JSON.stringify(diagnostics.connection.details, null, 2)}
              </pre>
            )}
          </div>

          {/* Table Status */}
          <div className="bg-neutral-900 rounded-lg p-3">
            <h4 className="font-medium text-neutral-200 mb-2">Table Status</h4>
            <div className="space-y-1">
              <div className={`text-sm ${diagnostics.table.exists ? 'text-green-400' : 'text-red-400'}`}>
                {diagnostics.table.exists ? '✅ Table Exists' : '❌ Table Missing'}
              </div>
              {diagnostics.table.exists && (
                <div className={`text-sm ${diagnostics.table.canAccess ? 'text-green-400' : 'text-red-400'}`}>
                  {diagnostics.table.canAccess ? '✅ Access Granted' : '❌ Access Denied'}
                </div>
              )}
              {diagnostics.table.error && (
                <div className="text-xs text-red-400 mt-1">
                  Error: {diagnostics.table.error}
                </div>
              )}
            </div>
          </div>

          {/* Profile Creation Test */}
          {diagnostics.profileTest && (
            <div className="bg-neutral-900 rounded-lg p-3">
              <h4 className="font-medium text-neutral-200 mb-2">Profile Test</h4>
              <div className="space-y-1">
                <div className={`text-sm ${diagnostics.profileTest.success ? 'text-green-400' : 'text-red-400'}`}>
                  {diagnostics.profileTest.success ? '✅ Profile Creation Works' : '❌ Profile Creation Failed'}
                </div>
                {diagnostics.profileTest.error && (
                  <div className="text-xs text-red-400 mt-1">
                    Error: {diagnostics.profileTest.error}
                  </div>
                )}
                {diagnostics.profileTest.success && (
                  <div className="text-xs text-green-400 mt-1">
                    Successfully created and deleted test profile
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Recommendations */}
          <div className="bg-neutral-900 rounded-lg p-3">
            <h4 className="font-medium text-neutral-200 mb-2">Recommendations</h4>
            <div className="space-y-1">
              {diagnostics.recommendations.map((rec, index) => (
                <div key={index} className="text-sm text-neutral-300">
                  {rec}
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          {!diagnostics.table.exists && (
            <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-3">
              <h4 className="font-medium text-yellow-400 mb-2">Quick Fix</h4>
              <p className="text-sm text-neutral-300 mb-3">
                The user_profiles table does not exist. Please run this SQL in your Supabase dashboard:
              </p>
              <div className="bg-neutral-950 rounded-md p-2 text-xs text-neutral-300 overflow-x-auto">
                <div className="font-mono">
                  Go to: Supabase Dashboard → SQL Editor → New Query
                  <br />
                  Then paste the contents of: create-user-profiles-table.sql
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
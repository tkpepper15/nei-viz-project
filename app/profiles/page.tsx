"use client";

import { useState } from 'react';
import { useAuth } from '../components/auth/AuthProvider';
import { SavedProfile } from '../components/circuit-simulator/types/savedProfiles';

export default function ProfilesPage() {
  const { user } = useAuth();
  const [savedProfiles] = useState<SavedProfile[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const profilesPerPage = 12;

  const handleLoadProfile = (profile: SavedProfile) => {
    console.log('Load profile:', profile);
  };

  const handleDeleteProfile = (profileId: string) => {
    console.log('Delete profile:', profileId);
  };

  // Pagination logic
  const indexOfLastProfile = currentPage * profilesPerPage;
  const indexOfFirstProfile = indexOfLastProfile - profilesPerPage;
  const currentProfiles = savedProfiles.slice(indexOfFirstProfile, indexOfLastProfile);
  const totalPages = Math.ceil(savedProfiles.length / profilesPerPage);

  return (
    <div className="h-screen overflow-auto">
      <div className="p-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-orange-400">Saved Profiles</h1>
          <p className="text-neutral-400 text-sm mt-1">
            Manage your saved circuit configurations
          </p>
        </div>

        {!user ? (
          <div className="bg-neutral-900 rounded-lg border border-neutral-800 p-12 text-center">
            <h3 className="text-lg font-semibold text-neutral-300 mb-2">
              Sign in to save profiles
            </h3>
            <p className="text-neutral-500">
              You must be signed in to save and manage circuit configurations
            </p>
          </div>
        ) : savedProfiles.length === 0 ? (
          <div className="bg-neutral-900 rounded-lg border border-neutral-800 p-12 text-center">
            <h3 className="text-lg font-semibold text-neutral-300 mb-2">
              No saved profiles yet
            </h3>
            <p className="text-neutral-500">
              Start by running computations in the simulator and saving your configurations
            </p>
          </div>
        ) : (
          <>
            <div className="bg-neutral-900 rounded-lg border border-neutral-800 p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {currentProfiles.map((profile) => (
                  <div
                    key={profile.id}
                    className="bg-neutral-800/50 border border-neutral-700 rounded-lg p-4 hover:border-orange-600 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="font-semibold text-neutral-200">{profile.name}</h3>
                      <button
                        onClick={() => handleDeleteProfile(profile.id)}
                        className="text-red-400 hover:text-red-300 text-sm"
                        title="Delete profile"
                      >
                        ✕
                      </button>
                    </div>
                    {profile.description && (
                      <p className="text-sm text-neutral-400 mb-3">{profile.description}</p>
                    )}
                    <button
                      onClick={() => handleLoadProfile(profile)}
                      className="w-full px-3 py-2 bg-orange-600 hover:bg-orange-700 rounded text-sm transition-colors"
                    >
                      Load Profile
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="flex justify-center items-center gap-4 mt-6">
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-50 disabled:cursor-not-allowed rounded transition-colors"
                >
                  Previous
                </button>
                <span className="text-neutral-400">
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-50 disabled:cursor-not-allowed rounded transition-colors"
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

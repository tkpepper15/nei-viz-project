"use client";

import SimplifiedSettingsModal from '../components/settings/SimplifiedSettingsModal';

export default function SettingsPage() {
  const handleSettingsUpdate = (settings: { useSymmetricGrid: boolean; maxComputationResults: number }) => {
    console.log('Settings updated:', settings);
  };

  return (
    <div className="h-screen overflow-auto">
      <div className="p-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-orange-400">Settings</h1>
          <p className="text-neutral-400 text-sm mt-1">
            Configure computation pipeline and optimization settings
          </p>
        </div>

        <div className="max-w-4xl bg-neutral-900 rounded-lg border border-neutral-800 p-6">
          <SimplifiedSettingsModal
            isOpen={true}
            onClose={() => {}}
            onSettingsChange={handleSettingsUpdate}
            gridSize={9}
            totalPossibleResults={59049}
          />
        </div>
      </div>
    </div>
  );
}

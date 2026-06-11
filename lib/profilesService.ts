import { SavedProfile } from '../app/components/circuit-simulator/types/savedProfiles';
import { CircuitParameters } from '../app/components/circuit-simulator/types/parameters';

export interface UserProfileRow {
  id: string;
  user_id: string;
  username: string | null;
  full_name: string | null;
  avatar_url: string | null;
  default_grid_size: number;
  default_min_freq: number;
  default_max_freq: number;
  default_num_points: number;
  created_at: string;
  updated_at: string;
}

const SETTINGS_KEY = 'nei-viz-user-settings';

interface UserSettings {
  default_grid_size: number;
}

function loadSettings(): UserSettings {
  if (typeof window === 'undefined') return { default_grid_size: 9 };
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? JSON.parse(raw) : { default_grid_size: 9 };
  } catch {
    return { default_grid_size: 9 };
  }
}

function saveSettings(s: UserSettings): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

const PROFILES_KEY = 'nei-viz-saved-profiles';

function loadProfiles(): SavedProfile[] {
  if (typeof window === 'undefined') return [];
  try {
    // Check multiple legacy keys for migration
    for (const key of [PROFILES_KEY, 'nei-viz-profiles', 'savedProfiles']) {
      const raw = localStorage.getItem(key);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) return parsed;
      }
    }
    return [];
  } catch {
    return [];
  }
}

function saveProfiles(profiles: SavedProfile[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(PROFILES_KEY, JSON.stringify(profiles));
}

export class ProfilesService {
  static async getUserDefaultGridSize(_userId: string): Promise<number> {
    return loadSettings().default_grid_size;
  }

  static async createUserProfile(_userId: string, gridSize = 9): Promise<void> {
    const s = loadSettings();
    s.default_grid_size = gridSize;
    saveSettings(s);
  }

  static async getUserProfiles(_userId: string): Promise<SavedProfile[]> {
    return loadProfiles();
  }

  static async createProfile(
    _userId: string,
    name: string,
    parameters: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    description?: string,
  ): Promise<SavedProfile> {
    const profiles = loadProfiles();
    const now = Date.now();
    const newProfile: SavedProfile = {
      id: `profile-${now}-${Math.random().toString(36).slice(2)}`,
      name,
      description,
      groundTruthParams: parameters,
      gridSize,
      minFreq,
      maxFreq,
      numPoints,
      created: now,
      lastModified: now,
      isComputed: false,
    };
    profiles.push(newProfile);
    saveProfiles(profiles);
    return newProfile;
  }

  static async updateProfile(
    profileId: string,
    updates: Partial<SavedProfile>,
  ): Promise<SavedProfile> {
    const profiles = loadProfiles();
    const idx = profiles.findIndex(p => p.id === profileId);
    if (idx === -1) throw new Error(`Profile ${profileId} not found`);
    profiles[idx] = { ...profiles[idx], ...updates, lastModified: Date.now() };
    saveProfiles(profiles);
    return profiles[idx];
  }

  static async deleteProfile(profileId: string): Promise<void> {
    saveProfiles(loadProfiles().filter(p => p.id !== profileId));
  }

  static async deleteMultipleProfiles(profileIds: string[]): Promise<void> {
    const set = new Set(profileIds);
    saveProfiles(loadProfiles().filter(p => !set.has(p.id)));
  }
}

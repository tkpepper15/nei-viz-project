/**
 * Supabase stub — replaced with localStorage-based self-hosted storage.
 * Exists only so dynamic imports of this module don't crash.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const noop = (): any => ({ data: null, error: new Error('Supabase removed') });

export const supabase = {
  auth: {
    getUser: noop,
    getSession: noop,
    signInAnonymously: noop,
    signInWithPassword: noop,
    signUp: noop,
    signOut: noop,
    onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
  },
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  from: (_table: string) => ({
    select: () => ({ eq: () => ({ single: noop, maybeSingle: noop, order: noop }) }),
    insert: () => ({ select: () => ({ single: noop }) }),
    update: () => ({ eq: () => ({ select: () => ({ single: noop }) }) }),
    delete: () => ({ eq: () => ({ in: noop }) }),
  }),
};

export function createServerSupabaseClient() {
  return supabase;
}

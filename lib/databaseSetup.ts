import { supabase } from './supabase';

export class DatabaseSetup {
  /**
   * Test Supabase connection and authentication
   */
  static async testConnection(): Promise<{success: boolean, error?: string, details?: any}> {
    try {
      console.log('🔍 Testing Supabase connection...');
      
      // Test 1: Basic connection using user_profiles table since it exists
      const { data: connectionTest, error: connectionError } = await supabase
        .from('user_profiles')
        .select('count')
        .limit(1);
      
      // If we get a connection error that's not about permissions, it's a real connection issue
      if (connectionError && !['42501', 'PGRST116', 'PGRST301'].includes(connectionError.code || '')) {
        return {
          success: false,
          error: 'Connection failed',
          details: connectionError
        };
      }
      
      console.log('✅ Basic connection successful');
      
      // Test 2: Authentication
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      
      if (userError) {
        console.log('⚠️ Auth error (might be expected):', userError);
      }
      
      // Also check session
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      
      const authStatus = {
        hasUser: !!user,
        hasSession: !!session,
        userId: user?.id || session?.user?.id,
        userRole: user?.role || session?.user?.role,
        isAnonymous: user?.is_anonymous || session?.user?.is_anonymous
      };
      
      console.log('✅ Authentication test:', authStatus);
      
      return {
        success: true, // Connection works even if auth is pending
        details: {
          hasConnection: true,
          ...authStatus,
          connectionTest: connectionError ? 'Permission-based (expected)' : 'Full access'
        }
      };
      
    } catch (error) {
      return {
        success: false,
        error: 'Connection test failed',
        details: error
      };
    }
  }

  /**
   * Check if user_profiles table exists and has correct permissions
   */
  static async checkTableExists(): Promise<{exists: boolean, error?: string, canAccess?: boolean}> {
    try {
      console.log('🗄️ Checking user_profiles table...');
      
      const { data, error } = await supabase
        .from('user_profiles')
        .select('count')
        .limit(1);
      
      if (error) {
        if (error.code === 'PGRST116') {
          return {
            exists: false,
            error: 'Table does not exist'
          };
        }
        
        if (error.code === '42501') {
          return {
            exists: true,
            canAccess: false,
            error: 'Permission denied - check RLS policies'
          };
        }
        
        return {
          exists: false,
          error: error.message
        };
      }
      
      console.log('✅ Table exists and accessible');
      return {
        exists: true,
        canAccess: true
      };
      
    } catch (error) {
      return {
        exists: false,
        error: `Table check failed: ${error}`
      };
    }
  }

  /**
   * Attempt to create the user_profiles table
   * Note: This requires elevated permissions and may fail
   */
  static async createUserProfilesTable(): Promise<{success: boolean, error?: string}> {
    try {
      console.log('🛠️ Attempting to create user_profiles table...');
      
      const createTableSQL = `
        CREATE TABLE IF NOT EXISTS user_profiles (
          id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
          user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
          name TEXT NOT NULL,
          description TEXT,
          parameters JSONB NOT NULL,
          grid_size INTEGER DEFAULT 10,
          min_freq NUMERIC DEFAULT 0.1,
          max_freq NUMERIC DEFAULT 100000,
          num_points INTEGER DEFAULT 100,
          is_computed BOOLEAN DEFAULT false,
          computation_time NUMERIC,
          total_points INTEGER,
          valid_points INTEGER,
          computation_results JSONB,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_profiles_created_at ON user_profiles(created_at);
      `;
      
      // Note: This will likely fail due to permissions
      const { error } = await supabase.rpc('exec_sql', { sql: createTableSQL });
      
      if (error) {
        return {
          success: false,
          error: `Table creation failed: ${error.message}. Please run the SQL manually in Supabase dashboard.`
        };
      }
      
      console.log('✅ Table created successfully');
      return { success: true };
      
    } catch (error) {
      return {
        success: false,
        error: `Table creation failed: ${error}. Please run the SQL manually in Supabase dashboard.`
      };
    }
  }

  /**
   * Test profile creation (if user is authenticated)
   */
  static async testProfileCreation(): Promise<{success: boolean, error?: string, profileId?: string}> {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      
      if (!user) {
        return {
          success: false,
          error: 'No authenticated user for profile test'
        };
      }

      console.log('🧪 Testing profile creation for user:', user.id);

      // Try to create a test profile
      const testProfile = {
        user_id: user.id,
        name: 'Diagnostic Test Profile',
        description: 'Temporary test profile - safe to delete',
        parameters: {
          Rsh: 100,
          Ra: 1000,
          Ca: 1.0e-6,
          Rb: 800,
          Cb: 0.8e-6,
          frequency_range: [0.1, 100000]
        },
        grid_size: 5,
        min_freq: 0.1,
        max_freq: 100000,
        num_points: 50,
        is_computed: false
      };

      const { data, error } = await supabase
        .from('user_profiles')
        .insert(testProfile)
        .select('id')
        .single();

      if (error) {
        return {
          success: false,
          error: `Profile creation failed: ${error.message}`
        };
      }

      // Clean up test profile
      if (data?.id) {
        await supabase
          .from('user_profiles')
          .delete()
          .eq('id', data.id);
      }

      console.log('✅ Profile creation test successful');
      
      return {
        success: true,
        profileId: data?.id
      };

    } catch (error) {
      return {
        success: false,
        error: `Profile test failed: ${error}`
      };
    }
  }

  /**
   * Run comprehensive database diagnostics
   */
  static async runDiagnostics(): Promise<{
    connection: any,
    table: any,
    profileTest?: any,
    recommendations: string[]
  }> {
    console.log('🔬 Running database diagnostics...');
    
    const connection = await this.testConnection();
    const table = await this.checkTableExists();
    const profileTest = await this.testProfileCreation();
    
    const recommendations: string[] = [];
    
    if (!connection.success) {
      recommendations.push('❌ Fix Supabase connection - check environment variables');
    }
    
    if (!connection.details?.hasUser && !connection.details?.hasSession) {
      recommendations.push('❌ Sign in to test profile functionality');
    }
    
    if (!table.exists) {
      recommendations.push('❌ Create user_profiles table - run SQL migration in Supabase dashboard');
      recommendations.push('📋 SQL file location: create-user-profiles-table.sql');
    }
    
    if (table.exists && !table.canAccess) {
      recommendations.push('❌ Fix RLS policies - check user_profiles table permissions');
    }
    
    if (table.exists && table.canAccess && !profileTest.success) {
      recommendations.push(`❌ Profile creation failed: ${profileTest.error}`);
    }
    
    if (table.exists && table.canAccess && profileTest.success) {
      recommendations.push('✅ Database fully functional - profiles can be created and saved!');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('✅ Database setup appears correct');
    }
    
    console.log('📋 Diagnostics complete');
    
    return {
      connection,
      table,
      profileTest,
      recommendations
    };
  }
}
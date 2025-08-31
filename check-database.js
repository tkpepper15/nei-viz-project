#!/usr/bin/env node

/**
 * Database connectivity and schema check script
 */

const fs = require('fs');

async function checkDatabase() {
  try {
    // Load environment variables from .env.local manually
    const envPath = '.env.local';
    let supabaseUrl, supabaseKey;
    
    if (fs.existsSync(envPath)) {
      const envContent = fs.readFileSync(envPath, 'utf8');
      const lines = envContent.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('NEXT_PUBLIC_SUPABASE_URL=')) {
          supabaseUrl = line.split('=')[1]?.trim();
        }
        if (line.startsWith('NEXT_PUBLIC_SUPABASE_ANON_KEY=')) {
          supabaseKey = line.split('=')[1]?.trim();
        }
      }
    }
    
    console.log('🔗 Environment check:');
    console.log('  SUPABASE_URL:', supabaseUrl ? 'Set' : 'Missing');
    console.log('  SUPABASE_KEY:', supabaseKey ? 'Set' : 'Missing');
    
    if (!supabaseUrl || !supabaseKey) {
      console.error('❌ Missing Supabase environment variables');
      process.exit(1);
    }

    // Try to connect to Supabase
    const { createClient } = require('@supabase/supabase-js');
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    console.log('\n📊 Checking database connectivity...');
    
    // Check authentication
    const { data: authData, error: authError } = await supabase.auth.getUser();
    console.log('🔑 Auth status:', authError ? 'No user signed in' : `User: ${authData?.user?.id}`);
    
    // Check if tagged_models table exists
    console.log('\n📋 Checking tagged_models table...');
    const { data, error } = await supabase
      .from('tagged_models')
      .select('*', { count: 'exact', head: true });
      
    if (error) {
      console.error('❌ tagged_models table check failed:', error.message);
      console.error('Full error:', JSON.stringify(error, null, 2));
    } else {
      console.log('✅ tagged_models table exists and is accessible');
      console.log('Current row count:', data);
    }
    
    // Check if user_sessions table exists
    console.log('\n👤 Checking user_sessions table...');
    const { data: sessionsData, error: sessionsError } = await supabase
      .from('user_sessions')
      .select('*', { count: 'exact', head: true });
      
    if (sessionsError) {
      console.error('❌ user_sessions table check failed:', sessionsError.message);
    } else {
      console.log('✅ user_sessions table exists and is accessible');
      console.log('Current row count:', sessionsData);
    }
    
    console.log('\n🎯 Summary:');
    console.log('  Database connection:', error || sessionsError ? '❌ Issues found' : '✅ Working');
    console.log('  Required tables:', (error || sessionsError) ? '❌ Missing or inaccessible' : '✅ Present');
    
    if (error || sessionsError) {
      console.log('\n💡 Next steps:');
      console.log('  1. Run the SQL setup scripts in supabase-sql-simple/ directory');
      console.log('  2. Check your Supabase project settings and RLS policies');
      console.log('  3. Ensure you have the correct database URL and API keys');
    }
    
  } catch (err) {
    console.error('❌ Script failed:', err);
    process.exit(1);
  }
}

checkDatabase();
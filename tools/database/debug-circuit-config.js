#!/usr/bin/env node

// Debug script to check circuit configuration setup
const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// Simple dotenv loader
function loadEnv() {
  const envPath = path.join(__dirname, '.env.local');
  if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf-8');
    const lines = envContent.split('\n');
    lines.forEach(line => {
      if (line.includes('=') && !line.startsWith('#')) {
        const [key, ...valueParts] = line.split('=');
        const value = valueParts.join('=').trim();
        process.env[key.trim()] = value.replace(/^["']|["']$/g, '');
      }
    });
  }
}

loadEnv();

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

console.log('🔍 Circuit Configuration Diagnostics');
console.log('====================================');

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function debugCircuitConfig() {
  console.log('🔄 Testing database tables and relationships...\n');

  try {
    // Test 1: Check if circuit_configurations table exists and is accessible
    console.log('1️⃣ Testing circuit_configurations table access:');
    const { data: configsData, error: configsError } = await supabase
      .from('circuit_configurations')
      .select('*')
      .limit(5);
    
    console.log('   Result:', configsError ? '❌ Failed' : '✅ Success');
    if (configsError) {
      console.log('   Error:', configsError.message, `(${configsError.code})`);
    } else {
      console.log(`   Found ${configsData?.length || 0} circuit configurations`);
    }

    // Test 2: Check if tagged_models table exists with circuit_config_id
    console.log('\n2️⃣ Testing tagged_models table with circuit_config_id:');
    const { data: taggedData, error: taggedError } = await supabase
      .from('tagged_models')
      .select('id, circuit_config_id, model_id, tag_name')
      .limit(5);
    
    console.log('   Result:', taggedError ? '❌ Failed' : '✅ Success');
    if (taggedError) {
      console.log('   Error:', taggedError.message, `(${taggedError.code})`);
    } else {
      console.log(`   Found ${taggedData?.length || 0} tagged models`);
      if (taggedData && taggedData.length > 0) {
        console.log('   Sample:', taggedData[0]);
      }
    }

    // Test 3: Check if user_sessions has current_circuit_config_id field
    console.log('\n3️⃣ Testing user_sessions table with current_circuit_config_id:');
    const { data: sessionsData, error: sessionsError } = await supabase
      .from('user_sessions')
      .select('id, user_id, session_name, current_circuit_config_id')
      .limit(5);
    
    console.log('   Result:', sessionsError ? '❌ Failed' : '✅ Success');
    if (sessionsError) {
      console.log('   Error:', sessionsError.message, `(${sessionsError.code})`);
    } else {
      console.log(`   Found ${sessionsData?.length || 0} user sessions`);
      if (sessionsData && sessionsData.length > 0) {
        console.log('   Sample session:', {
          id: sessionsData[0].id,
          has_circuit_config: !!sessionsData[0].current_circuit_config_id
        });
      }
    }

    // Test 4: Check if user_profiles table has the right structure (user metadata only)
    console.log('\n4️⃣ Testing user_profiles table structure:');
    const { data: profilesData, error: profilesError } = await supabase
      .from('user_profiles')
      .select('id, user_id, username, full_name')
      .limit(3);
    
    console.log('   Result:', profilesError ? '❌ Failed' : '✅ Success');
    if (profilesError) {
      console.log('   Error:', profilesError.message, `(${profilesError.code})`);
    } else {
      console.log(`   Found ${profilesData?.length || 0} user profiles`);
    }

    console.log('\n📊 ANALYSIS:');
    
    if (configsError?.code === 'PGRST205') {
      console.log('❌ CRITICAL: circuit_configurations table is missing!');
      console.log('   Run: database-schemas-updated/02-circuit-configurations-table.sql');
    } else if (!configsError && (!configsData || configsData.length === 0)) {
      console.log('⚠️  WARNING: No circuit configurations exist yet');
      console.log('   This means users can\'t tag models until they create a configuration');
    }
    
    if (taggedError?.message?.includes('circuit_config_id')) {
      console.log('❌ CRITICAL: tagged_models table missing circuit_config_id field!');
      console.log('   Run: database-schemas-updated/03-tagged-models-table.sql');
    }
    
    if (sessionsError?.message?.includes('current_circuit_config_id')) {
      console.log('❌ CRITICAL: user_sessions table missing current_circuit_config_id field!');
      console.log('   Run: database-schemas-updated/04-user-sessions-table.sql');
    }

    console.log('\n💡 RECOMMENDATIONS:');
    console.log('1. If tables are missing, apply the SQL files from database-schemas-updated/');
    console.log('2. If tables exist but are empty, create a default circuit configuration');
    console.log('3. Check browser console for any application-level errors');

  } catch (error) {
    console.error('💥 Unexpected error:', error.message);
  }
}

debugCircuitConfig();
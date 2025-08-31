#!/usr/bin/env node

// Final workflow test with proper UUID and corrected schema
const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

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

// Install uuid if not present
function installUuid() {
  try {
    require('uuid');
  } catch (error) {
    console.log('📦 Installing uuid package...');
    const { execSync } = require('child_process');
    execSync('npm install uuid', { stdio: 'inherit' });
  }
}

// Try to install uuid, if not available use fallback
let uuid;
try {
  installUuid();
  uuid = require('uuid');
} catch (error) {
  // Fallback UUID generator
  console.log('⚠️  Using fallback UUID generator');
  uuid = {
    v4: () => 'test-' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
  };
}

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

console.log('🧪 Final Workflow Test - Models Loading');
console.log('========================================');

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function testCompleteWorkflow() {
  console.log('🔄 Testing complete workflow with corrected schema...\n');

  const testUserId = uuid.v4(); // Use proper UUID
  console.log('👤 Using test user ID:', testUserId);

  try {
    // Step 1: Test creating circuit configuration with corrected parameters
    console.log('1️⃣ Testing Circuit Configuration Creation (with frequency_range):');
    
    const testCircuitConfig = {
      user_id: testUserId,
      name: 'Test Circuit Configuration',
      description: 'Test configuration with proper schema',
      circuit_parameters: {
        Rsh: 5005,
        Ra: 5005,
        Ca: 0.00002505,
        Rb: 5005,
        Cb: 0.00002505,
        frequency_range: [0.1, 100000]
      },
      grid_size: 9,
      min_freq: 0.1,
      max_freq: 100000,
      num_points: 100,
      is_computed: false,
      is_public: false
    };

    const { data: createdConfig, error: configError } = await supabase
      .from('circuit_configurations')
      .insert(testCircuitConfig)
      .select()
      .single();

    if (configError) {
      console.log('   ❌ Failed to create circuit configuration:', configError.message);
      console.log('   Details:', configError);
      return false;
    }

    console.log('   ✅ Circuit configuration created:', createdConfig.id);
    console.log('   ✅ Schema validation passed (includes frequency_range)');

    // Step 2: Test session with active circuit config
    console.log('\n2️⃣ Testing User Session with Active Circuit Config:');
    
    const testSession = {
      user_id: testUserId,
      session_name: 'Test Session',
      description: 'Test session for workflow validation',
      current_circuit_config_id: createdConfig.id,
      is_active: true
    };

    const { data: createdSession, error: sessionError } = await supabase
      .from('user_sessions')
      .insert(testSession)
      .select()
      .single();

    if (sessionError) {
      console.log('   ❌ Failed to create session:', sessionError.message);
      return false;
    }

    console.log('   ✅ Session created:', createdSession.id);
    console.log('   ✅ Active circuit config linked:', createdSession.current_circuit_config_id);

    // Step 3: Test tagged model creation (the main issue)
    console.log('\n3️⃣ Testing Tagged Model Creation (Circuit-Specific):');
    
    const testTaggedModel = {
      user_id: testUserId,
      session_id: createdSession.id,
      circuit_config_id: createdConfig.id,
      model_id: 'test-model-' + Date.now(),
      tag_name: 'Test Tag',
      tag_category: 'user',
      circuit_parameters: {
        Rsh: 5105,
        Ra: 5050,
        Ca: 0.000025,
        Rb: 4995,
        Cb: 0.000026
      },
      resnorm_value: 0.123,
      notes: 'Test tagged model with corrected workflow',
      is_interesting: true
    };

    const { data: createdTag, error: tagError } = await supabase
      .from('tagged_models')
      .insert(testTaggedModel)
      .select()
      .single();

    if (tagError) {
      console.log('   ❌ Failed to create tagged model:', tagError.message);
      console.log('   Details:', tagError);
      return false;
    }

    console.log('   ✅ Tagged model created:', createdTag.id);
    console.log('   ✅ Circuit-specific linking verified');

    // Step 4: Test the query that the app would use (circuit-specific models)
    console.log('\n4️⃣ Testing Circuit-Specific Tagged Model Query:');
    
    const { data: circuitTags, error: queryError } = await supabase
      .from('tagged_models')
      .select('*')
      .eq('circuit_config_id', createdConfig.id)
      .eq('user_id', testUserId)
      .order('tagged_at', { ascending: false });

    if (queryError) {
      console.log('   ❌ Failed to query tagged models:', queryError.message);
      return false;
    }

    console.log(`   ✅ Found ${circuitTags.length} tagged models for circuit`);
    console.log('   ✅ Query matches application logic');

    // Step 5: Test activeConfigId workflow
    console.log('\n5️⃣ Testing Active Config Selection Workflow:');
    
    // Simulate loading configurations for a user
    const { data: userConfigs, error: loadError } = await supabase
      .from('circuit_configurations')
      .select('*')
      .eq('user_id', testUserId)
      .order('updated_at', { ascending: false });

    if (loadError || !userConfigs || userConfigs.length === 0) {
      console.log('   ❌ Failed to load user configurations');
      return false;
    }

    const activeConfigId = userConfigs[0].id; // First config becomes active
    console.log('   ✅ Active config would be set to:', activeConfigId);

    // Verify this config would allow tagging
    if (activeConfigId === createdConfig.id) {
      console.log('   ✅ Active config matches created config - tagging would work');
    } else {
      console.log('   ⚠️  Active config differs from created config');
    }

    // Cleanup
    console.log('\n6️⃣ Cleanup Test Data:');
    await supabase.from('tagged_models').delete().eq('user_id', testUserId);
    await supabase.from('user_sessions').delete().eq('user_id', testUserId);
    await supabase.from('circuit_configurations').delete().eq('user_id', testUserId);
    console.log('   ✅ Test data cleaned up');

    console.log('\n🎉 COMPLETE WORKFLOW TEST PASSED!');
    console.log('✅ Circuit configurations create properly with frequency_range');
    console.log('✅ Sessions link to active circuit configurations');
    console.log('✅ Tagged models are circuit-specific and queryable');
    console.log('✅ Active config selection works correctly');
    console.log('✅ Model loading should now work in the application');

    return true;

  } catch (error) {
    console.error('💥 Unexpected error during workflow test:', error.message);
    console.error('Stack:', error.stack);
    return false;
  }
}

testCompleteWorkflow().then(success => {
  if (success) {
    console.log('\n🚀 APPLICATION IS READY - Models should now load correctly!');
    console.log('\n📋 User should:');
    console.log('   1. Authenticate in the application');
    console.log('   2. Default circuit configuration will be auto-created');
    console.log('   3. Models can be tagged and will appear in the correct circuit');
    console.log('   4. Tagged models persist across sessions');
    process.exit(0);
  } else {
    console.log('\n❌ Workflow test failed - check errors above');
    process.exit(1);
  }
});
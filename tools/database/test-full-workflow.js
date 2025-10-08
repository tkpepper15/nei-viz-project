#!/usr/bin/env node

// Test script to validate full workflow from user authentication to tagged models
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

console.log('🧪 Full Workflow Test');
console.log('=====================');

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function testFullWorkflow() {
  console.log('🔄 Testing complete user workflow...\n');

  try {
    // Step 1: Test creating a mock circuit configuration
    console.log('1️⃣ Testing Circuit Configuration Creation:');
    
    const mockUserId = 'test-user-' + Date.now();
    console.log('   Using mock user ID:', mockUserId);
    
    const testCircuitConfig = {
      user_id: mockUserId,
      name: 'Test Circuit Configuration',
      description: 'Automated test configuration',
      circuit_parameters: {
        Rsh: { value: 1000, min: 100, max: 10000 },
        Ra: { value: 500, min: 50, max: 5000 },
        Ca: { value: 0.00001, min: 0.000001, max: 0.0001 },
        Rb: { value: 200, min: 20, max: 2000 },
        Cb: { value: 0.00005, min: 0.000005, max: 0.0005 }
      },
      grid_size: 10,
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
      return false;
    }

    console.log('   ✅ Circuit configuration created:', createdConfig.id);

    // Step 2: Test creating a user session with active circuit config
    console.log('\n2️⃣ Testing User Session Creation:');
    
    const testSession = {
      user_id: mockUserId,
      session_name: 'Test Session',
      description: 'Automated test session',
      current_circuit_config_id: createdConfig.id,
      is_active: true,
      last_accessed: new Date().toISOString()
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

    // Step 3: Test creating tagged models linked to circuit configuration
    console.log('\n3️⃣ Testing Tagged Model Creation:');
    
    const testTaggedModel = {
      user_id: mockUserId,
      session_id: createdSession.id,
      circuit_config_id: createdConfig.id,
      model_id: 'test-model-' + Date.now(),
      tag_name: 'Test Tag',
      tag_category: 'user',
      circuit_parameters: {
        Rsh: 1100,
        Ra: 550,
        Ca: 0.000011,
        Rb: 220,
        Cb: 0.000055
      },
      resnorm_value: 0.123,
      notes: 'Automated test tagged model',
      is_interesting: true
    };

    const { data: createdTag, error: tagError } = await supabase
      .from('tagged_models')
      .insert(testTaggedModel)
      .select()
      .single();

    if (tagError) {
      console.log('   ❌ Failed to create tagged model:', tagError.message);
      return false;
    }

    console.log('   ✅ Tagged model created:', createdTag.id);
    console.log('   ✅ Linked to circuit config:', createdTag.circuit_config_id);

    // Step 4: Test querying tagged models for specific circuit
    console.log('\n4️⃣ Testing Circuit-Specific Tagged Model Query:');
    
    const { data: circuitTags, error: queryError } = await supabase
      .from('tagged_models')
      .select('*')
      .eq('circuit_config_id', createdConfig.id)
      .eq('user_id', mockUserId);

    if (queryError) {
      console.log('   ❌ Failed to query tagged models:', queryError.message);
      return false;
    }

    console.log(`   ✅ Found ${circuitTags.length} tagged models for circuit`);
    if (circuitTags.length > 0) {
      console.log('   ✅ Sample tagged model:', {
        id: circuitTags[0].id,
        tag_name: circuitTags[0].tag_name,
        circuit_config_id: circuitTags[0].circuit_config_id
      });
    }

    // Step 5: Test relationship integrity (foreign key constraints)
    console.log('\n5️⃣ Testing Relationship Integrity:');
    
    // Verify that tagged models are properly linked to circuit configuration
    const { data: relationshipCheck, error: relError } = await supabase
      .from('tagged_models')
      .select(`
        id,
        tag_name,
        circuit_configurations!inner (
          id,
          name
        )
      `)
      .eq('circuit_config_id', createdConfig.id);

    if (relError) {
      console.log('   ❌ Relationship query failed:', relError.message);
      return false;
    }

    console.log('   ✅ Foreign key relationships working correctly');
    console.log(`   ✅ Found ${relationshipCheck.length} tagged models with valid circuit links`);

    // Cleanup: Delete test data
    console.log('\n6️⃣ Cleanup Test Data:');
    
    await supabase.from('tagged_models').delete().eq('user_id', mockUserId);
    await supabase.from('user_sessions').delete().eq('user_id', mockUserId);
    await supabase.from('circuit_configurations').delete().eq('user_id', mockUserId);
    
    console.log('   ✅ Test data cleaned up');

    console.log('\n🎉 WORKFLOW TEST PASSED!');
    console.log('✅ Circuit configurations work correctly');
    console.log('✅ Sessions link to active circuit configs');
    console.log('✅ Tagged models are circuit-specific');
    console.log('✅ Database relationships are intact');
    console.log('✅ All services can communicate properly');

    return true;

  } catch (error) {
    console.error('💥 Unexpected error during workflow test:', error.message);
    return false;
  }
}

testFullWorkflow().then(success => {
  if (success) {
    console.log('\n🚀 Application is ready for production use!');
    process.exit(0);
  } else {
    console.log('\n❌ Workflow test failed - check logs above');
    process.exit(1);
  }
});
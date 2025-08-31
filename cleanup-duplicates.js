#!/usr/bin/env node

// Cleanup script to remove duplicate default configurations
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

console.log('🧹 Cleaning up duplicate circuit configurations');
console.log('===============================================');

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function cleanupDuplicates() {
  console.log('🔄 Finding duplicate default configurations...\n');

  try {
    // Find all "Default Circuit Configuration" entries
    const { data: duplicateConfigs, error: findError } = await supabase
      .from('circuit_configurations')
      .select('id, user_id, name, created_at')
      .eq('name', 'Default Circuit Configuration')
      .order('created_at', { ascending: true });

    if (findError) {
      console.error('❌ Error finding duplicate configurations:', findError.message);
      return false;
    }

    console.log(`📊 Found ${duplicateConfigs.length} default configurations`);

    if (duplicateConfigs.length <= 1) {
      console.log('✅ No duplicates to clean up!');
      return true;
    }

    // Group by user_id
    const configsByUser = {};
    duplicateConfigs.forEach(config => {
      if (!configsByUser[config.user_id]) {
        configsByUser[config.user_id] = [];
      }
      configsByUser[config.user_id].push(config);
    });

    for (const [userId, userConfigs] of Object.entries(configsByUser)) {
      if (userConfigs.length > 1) {
        console.log(`\n👤 User ${userId} has ${userConfigs.length} default configurations`);
        
        // Keep the oldest (first created) configuration
        const keepConfig = userConfigs[0];
        const deleteConfigs = userConfigs.slice(1);
        
        console.log(`   ✅ Keeping: ${keepConfig.id} (created: ${keepConfig.created_at})`);
        
        for (const deleteConfig of deleteConfigs) {
          console.log(`   🗑️  Deleting: ${deleteConfig.id} (created: ${deleteConfig.created_at})`);
          
          const { error: deleteError } = await supabase
            .from('circuit_configurations')
            .delete()
            .eq('id', deleteConfig.id);
            
          if (deleteError) {
            console.error(`   ❌ Failed to delete ${deleteConfig.id}:`, deleteError.message);
          } else {
            console.log(`   ✅ Deleted ${deleteConfig.id}`);
          }
        }
      }
    }

    // Verify cleanup
    console.log('\n🔍 Verifying cleanup...');
    const { data: remainingConfigs, error: verifyError } = await supabase
      .from('circuit_configurations')
      .select('id, user_id, name')
      .eq('name', 'Default Circuit Configuration');

    if (verifyError) {
      console.error('❌ Error verifying cleanup:', verifyError.message);
      return false;
    }

    console.log(`📊 Remaining default configurations: ${remainingConfigs.length}`);
    
    remainingConfigs.forEach(config => {
      console.log(`   ✅ ${config.id} (user: ${config.user_id})`);
    });

    console.log('\n🎉 Cleanup completed successfully!');
    return true;

  } catch (error) {
    console.error('💥 Unexpected error during cleanup:', error.message);
    return false;
  }
}

cleanupDuplicates().then(success => {
  if (success) {
    console.log('\n✅ Database is now clean of duplicate configurations');
    process.exit(0);
  } else {
    console.log('\n❌ Cleanup failed - check logs above');
    process.exit(1);
  }
});
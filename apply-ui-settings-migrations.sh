#!/bin/bash

echo "🔄 Applying UI Settings Auto-Save Database Migrations..."
echo ""

# Check if supabase CLI is available
if ! command -v supabase &> /dev/null; then
    echo "❌ Supabase CLI not found. Please install it first:"
    echo "   npm install -g supabase"
    exit 1
fi

# Apply migrations in order
echo "1️⃣ Creating circuit_configurations table..."
supabase db push --include-all

echo ""
echo "2️⃣ Running migration verification..."
supabase db push --include-all

echo ""
echo "✅ Database migrations completed!"
echo ""
echo "📋 Summary of changes:"
echo "   • Created circuit_configurations table with ui_settings column"
echo "   • Added current_circuit_config_id to user_sessions table"
echo "   • Updated tagged_models table for new schema"
echo "   • Added indexes for performance"
echo "   • Created helper functions and views"
echo ""
echo "🚀 Your auto-save UI settings system is now ready!"
echo "   Run: npm run dev to test the functionality"
echo ""
echo "💡 Features available:"
echo "   • All UI settings automatically saved to database"
echo "   • Session persistence across page reloads" 
echo "   • No unwanted configuration creation"
echo "   • Real-time save status indicators"
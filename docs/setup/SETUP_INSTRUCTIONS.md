# 🔧 Profile System Setup Instructions

## Current Status
✅ **Authentication working** - Users can sign in  
✅ **Profile system working** - Profiles save to localStorage as fallback  
✅ **Diagnostic tools available** - Check database connection in dev mode  
❌ **Database table missing** - Profiles not saving to Supabase yet  

## Quick Fix: Create Database Table

### Step 1: Access Supabase Dashboard
1. Go to https://supabase.com/dashboard
2. Select your project: `mscpbbmaycoqaikqyvqe`
3. Navigate to **SQL Editor** in the left sidebar

### Step 2: Run Database Migration
1. Click **New query** 
2. Copy the entire contents of `create-user-profiles-table.sql`
3. Paste into the SQL editor
4. Click **Run** to execute

### Step 3: Verify Setup
1. Go back to your app at http://localhost:3003
2. Sign in (you should see the user profile circle in top right)
3. Look for the **Database Diagnostics** panel in the top right (development mode only)
4. Click **Run Diagnostics** to check if the table was created successfully

## How the System Currently Works

### ✅ What's Working Now:
- **Authentication**: Users sign in and get a profile circle
- **Fallback Storage**: Profiles save to localStorage per user
- **Error Handling**: Clear error messages when database is unavailable
- **Diagnostics**: Built-in tools to troubleshoot database issues

### 🔄 Fallback Behavior:
- If Supabase database table doesn't exist → uses localStorage
- If authentication fails → falls back to localStorage  
- If any database error occurs → saves locally with error notification

### 📊 Profile Features Available:
- Create and save circuit parameter profiles
- Edit profile names and descriptions  
- Delete individual or multiple profiles
- Load saved configurations for computation
- Per-user profile isolation

## Testing the Profile System

### Test 1: Basic Profile Creation
1. Sign in to the app
2. Set some circuit parameters (Rsh, Ra, Ca, Rb, Cb)
3. Use "Save Profile" feature in the left sidebar
4. Verify the profile appears in the saved profiles list

### Test 2: Profile Persistence  
1. Create a profile
2. Refresh the page
3. Sign in again
4. Verify the profile still exists

### Test 3: Multi-User Isolation
1. Sign out
2. Sign in with a different email
3. Verify you see different profiles (or none if new user)

## Database Migration SQL

The complete SQL to create the table is in `create-user-profiles-table.sql`. Key features:

- **RLS (Row Level Security)**: Users can only see their own profiles
- **UUID Primary Keys**: Proper database design
- **Foreign Key to auth.users**: Links profiles to authenticated users
- **Indexes**: Optimized queries for user_id and created_at
- **Triggers**: Automatic updated_at timestamp management

## Troubleshooting

### Issue: "Database table not created"
**Solution**: Run the SQL migration in Supabase dashboard

### Issue: "Permission denied"  
**Solution**: Check RLS policies are correctly applied

### Issue: "Authentication required"
**Solution**: Sign out and sign in again

### Issue: Profiles not appearing
**Solution**: 
1. Check browser console for detailed error logs
2. Use Database Diagnostics panel
3. Verify you're signed in with the same user

## Development Mode Features

When running in development (`NODE_ENV=development`), you get:
- **Database Diagnostics Panel**: Shows connection and table status
- **Enhanced Console Logging**: Detailed debug information  
- **Error Details**: Full error objects for troubleshooting

## Next Steps After Database Setup

1. ✅ Run the SQL migration
2. ✅ Test profile creation and loading  
3. ✅ Verify multi-user isolation works
4. 🔄 Remove the temporary diagnostic panel (optional)
5. 🚀 Deploy to production

## Support

If you encounter issues:
1. Check the browser console for detailed error logs
2. Use the Database Diagnostics tool in development mode
3. Verify your Supabase environment variables are correct
4. Ensure you're using the correct Supabase project

The system is designed to work reliably with localStorage fallback, so even if the database setup takes time, users can still save and load their circuit configurations.
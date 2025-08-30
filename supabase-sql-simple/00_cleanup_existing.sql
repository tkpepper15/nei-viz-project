-- 0. Cleanup existing ML tables (run this first if you already ran the full ML version)
DROP TABLE IF EXISTS ml_models CASCADE;
DROP TABLE IF EXISTS ml_training_datasets CASCADE;
DROP TABLE IF EXISTS parameter_optimization_jobs CASCADE;
DROP TABLE IF EXISTS visualization_snapshots CASCADE;
DROP TABLE IF EXISTS parameter_exploration_sessions CASCADE;

-- Keep only essential tables, but recreate them with simplified structure
DROP TABLE IF EXISTS tagged_models CASCADE;
DROP TABLE IF EXISTS user_sessions CASCADE;
-- Auto-generated NPZ dataset registration SQL - PUBLIC TEMPLATE DATASETS
-- Run this in your Supabase SQL editor
-- These datasets will be registered to specific users but marked as public templates
-- Registering to users: b3549b62-6f76-4d69-8453-cada0f6d5976 and 36dfd52f-2e98-411f-adf6-e69bb619d0ae


-- Sample circuit configuration for sample_grid_5_test.npz
INSERT INTO circuit_configurations (
    id,
    user_id,
    name,
    description,
    is_public,
    circuit_parameters,
    grid_size,
    min_freq,
    max_freq,
    num_points,
    is_computed,
    created_at,
    updated_at
) VALUES (
    '36dfd52f-2e98-411f-adf6-e69bb619d0ae',
    'b3549b62-6f76-4d69-8453-cada0f6d5976',  -- User ID for public datasets
    'Sample Grid 5 Dataset',
    'Precomputed 5×5 parameter grid - available to all users',
    true,
    '{"Rsh": 5000, "Ra": 3000, "Ca": 25e-6, "Rb": 4000, "Cb": 30e-6, "frequency_range": [0.10000000149011612, 100000.0]}'::jsonb,
    5,
    0.10000000149011612,
    100000.0,
    50,
    true,
    NOW(),
    NOW()
);

-- NPZ dataset reference for sample_grid_5_test.npz
INSERT INTO circuit_npz_datasets (
    id,
    circuit_config_id,
    user_id,
    npz_filename,
    dataset_name,
    file_path,
    grid_size,
    min_freq,
    max_freq,
    num_points,
    n_parameters,
    file_size_mb,
    storage_location,
    is_available,
    computation_metadata,
    best_resnorm,
    created_at,
    updated_at,
    computed_at
) VALUES (
    'efc5e04f-d87c-42aa-a1c5-7d457663968a',
    '36dfd52f-2e98-411f-adf6-e69bb619d0ae',  -- Link to configuration above
    'b3549b62-6f76-4d69-8453-cada0f6d5976',  -- User ID for public datasets
    'sample_grid_5_test.npz',
    'Sample precomputed dataset - available to all users',
    'data/npz/precomputed/sample_grid_5_test.npz',
    5,
    0.10000000149011612,
    100000.0,
    50,
    125,
    0.08628368377685547,
    'local',
    true,
    '{"grid_size": 5, "n_parameters": 125, "n_frequencies": 50, "freq_min": 0.10000000149011612, "freq_max": 100000.0, "auto_discovered": true, "discovery_timestamp": 1757518660.4956741}'::jsonb,
    0.36956319212913513,
    NOW(),
    NOW(),
    NOW()
);


-- Sample circuit configuration for grid_15_freq_0.1_100000_fixed.npz
INSERT INTO circuit_configurations (
    id,
    user_id,
    name,
    description,
    is_public,
    circuit_parameters,
    grid_size,
    min_freq,
    max_freq,
    num_points,
    is_computed,
    created_at,
    updated_at
) VALUES (
    '1f315f60-42ad-49a4-963a-5ccae2405409',
    'b3549b62-6f76-4d69-8453-cada0f6d5976',  -- User ID for public datasets
    'Performance Test Grid 15',
    'Large 15×15 parameter grid - available to all users',
    true,
    '{"Rsh": 5000, "Ra": 3000, "Ca": 25e-6, "Rb": 4000, "Cb": 30e-6, "frequency_range": [0.1, 100000.0]}'::jsonb,
    15,
    0.1,
    100000.0,
    100,
    true,
    NOW(),
    NOW()
);

-- NPZ dataset reference for grid_15_freq_0.1_100000_fixed.npz
INSERT INTO circuit_npz_datasets (
    id,
    circuit_config_id,
    user_id,
    npz_filename,
    dataset_name,
    file_path,
    grid_size,
    min_freq,
    max_freq,
    num_points,
    n_parameters,
    file_size_mb,
    storage_location,
    is_available,
    computation_metadata,
    best_resnorm,
    created_at,
    updated_at,
    computed_at
) VALUES (
    'a5796783-495c-42f1-ac4d-4e71ca0572fb',
    '1f315f60-42ad-49a4-963a-5ccae2405409',  -- Link to configuration above or replace with existing config_id
    'b3549b62-6f76-4d69-8453-cada0f6d5976',  -- User ID for public datasets
    'grid_15_freq_0.1_100000_fixed.npz',
    'Large grid 15 parameter sweep - available to all users',
    'data/npz/user_generated/grid_15_freq_0.1_100000_fixed.npz',
    15,
    0.1,
    100000.0,
    100,
    5000,
    6.57452392578125,
    'local',
    true,
    '{"grid_size": 15, "n_parameters": 5000, "n_frequencies": 100, "freq_min": 0.1, "freq_max": 100000.0, "auto_discovered": true, "discovery_timestamp": 1757518660.495693}'::jsonb,
    385.7760009765625,
    NOW(),
    NOW(),
    NOW()
);


-- DUPLICATE ENTRIES FOR SECOND USER --
-- This ensures both specified users can access these public templates

-- Sample circuit configuration for sample_grid_5_test.npz (User 2)
INSERT INTO circuit_configurations (
    id,
    user_id,
    name,
    description,
    is_public,
    circuit_parameters,
    grid_size,
    min_freq,
    max_freq,
    num_points,
    is_computed,
    created_at,
    updated_at
) VALUES (
    '36ffde11-3644-472a-a17e-10f8664fffd7',
    '36dfd52f-2e98-411f-adf6-e69bb619d0ae',  -- Second user ID
    'Sample Grid 5 Dataset',
    'Precomputed 5×5 parameter grid - available to all users',
    true,
    '{"Rsh": 5000, "Ra": 3000, "Ca": 25e-6, "Rb": 4000, "Cb": 30e-6, "frequency_range": [0.10000000149011612, 100000.0]}'::jsonb,
    5,
    0.10000000149011612,
    100000.0,
    50,
    true,
    NOW(),
    NOW()
);

-- NPZ dataset reference for sample_grid_5_test.npz (User 2)
INSERT INTO circuit_npz_datasets (
    id,
    circuit_config_id,
    user_id,
    npz_filename,
    dataset_name,
    file_path,
    grid_size,
    min_freq,
    max_freq,
    num_points,
    n_parameters,
    file_size_mb,
    storage_location,
    is_available,
    computation_metadata,
    best_resnorm,
    created_at,
    updated_at,
    computed_at
) VALUES (
    'efc5e04f-d87c-42aa-a1c5-7d457663968b',
    '36ffde11-3644-472a-a17e-10f8664fffd7',
    '36dfd52f-2e98-411f-adf6-e69bb619d0ae',  -- Second user ID
    'sample_grid_5_test.npz',
    'Sample precomputed dataset - available to all users',
    'data/npz/precomputed/sample_grid_5_test.npz',
    5,
    0.10000000149011612,
    100000.0,
    50,
    125,
    0.08628368377685547,
    'local',
    true,
    '{"grid_size": 5, "n_parameters": 125, "n_frequencies": 50, "freq_min": 0.10000000149011612, "freq_max": 100000.0, "auto_discovered": true, "discovery_timestamp": 1757518660.4956741}'::jsonb,
    0.36956319212913513,
    NOW(),
    NOW(),
    NOW()
);

-- Performance Test Grid 15 configuration (User 2)
INSERT INTO circuit_configurations (
    id,
    user_id,
    name,
    description,
    is_public,
    circuit_parameters,
    grid_size,
    min_freq,
    max_freq,
    num_points,
    is_computed,
    created_at,
    updated_at
) VALUES (
    '1f315f60-42ad-49a4-963a-5ccae2405410',
    '36dfd52f-2e98-411f-adf6-e69bb619d0ae',  -- Second user ID
    'Performance Test Grid 15',
    'Large 15×15 parameter grid - available to all users',
    true,
    '{"Rsh": 5000, "Ra": 3000, "Ca": 25e-6, "Rb": 4000, "Cb": 30e-6, "frequency_range": [0.1, 100000.0]}'::jsonb,
    15,
    0.1,
    100000.0,
    100,
    true,
    NOW(),
    NOW()
);

-- NPZ dataset reference for grid_15_freq_0.1_100000_fixed.npz (User 2)  
INSERT INTO circuit_npz_datasets (
    id,
    circuit_config_id,
    user_id,
    npz_filename,
    dataset_name,
    file_path,
    grid_size,
    min_freq,
    max_freq,
    num_points,
    n_parameters,
    file_size_mb,
    storage_location,
    is_available,
    computation_metadata,
    best_resnorm,
    created_at,
    updated_at,
    computed_at
) VALUES (
    'a5796783-495c-42f1-ac4d-4e71ca0572fd',
    '1f315f60-42ad-49a4-963a-5ccae2405410',
    '36dfd52f-2e98-411f-adf6-e69bb619d0ae',  -- Second user ID
    'grid_15_freq_0.1_100000_fixed.npz',
    'Large grid 15 parameter sweep - available to all users',
    'data/npz/user_generated/grid_15_freq_0.1_100000_fixed.npz',
    15,
    0.1,
    100000.0,
    100,
    5000,
    6.57452392578125,
    'local',
    true,
    '{"grid_size": 15, "n_parameters": 5000, "n_frequencies": 100, "freq_min": 0.1, "freq_max": 100000.0, "auto_discovered": true, "discovery_timestamp": 1757518660.495693}'::jsonb,
    385.7760009765625,
    NOW(),
    NOW(),
    NOW()
);

-- Verify the inserted data
SELECT 
    cc.name as config_name,
    cc.user_id,
    npz.dataset_name,
    npz.npz_filename,
    npz.n_parameters,
    npz.file_size_mb,
    npz.storage_location,
    npz.is_available,
    npz.storage_location as storage_type
FROM circuit_configurations cc
JOIN circuit_npz_datasets npz ON npz.circuit_config_id = cc.id
WHERE npz.computation_metadata->>'auto_discovered' = 'true'
ORDER BY cc.user_id, npz.created_at DESC;
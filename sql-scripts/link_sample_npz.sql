-- Link Sample NPZ to Circuit Configuration
-- Run this in your Supabase SQL editor after replacing user_id


-- Insert sample circuit configuration
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
    'a56f4a06-06ca-41d0-b41b-e3b0d6871f01',
    '00000000-0000-0000-0000-000000000000',  -- Replace with actual user_id
    'Sample Grid 5 Configuration',
    'Sample precomputed dataset for testing NPZ integration - 5x5x5x5x5 parameter grid',
    true,
    '{"Rsh": 5000.0, "Ra": 3000.0, "Ca": 2.5e-05, "Rb": 4000.0, "Cb": 3e-05, "frequency_range": [0.1, 100000]}'::jsonb,
    5,
    0.1,
    100000,
    50,
    true,
    NOW(),
    NOW()
);


-- Insert NPZ dataset reference
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
    '427d7723-945d-4794-bdd9-e9f931cbb120',
    'a56f4a06-06ca-41d0-b41b-e3b0d6871f01',
    '00000000-0000-0000-0000-000000000000',  -- Replace with actual user_id
    'sample_grid_5_test.npz',
    'Sample Grid 5 Test Dataset',
    'data/npz/precomputed/sample_grid_5_test.npz',
    5,
    0.10000000149011612,
    100000.0,
    50,
    125,
    0.08627986907958984,
    'precomputed',
    true,
    '{"grid_size": 5, "n_parameter_sets": 125, "n_frequencies": 50, "freq_min": 0.10000000149011612, "freq_max": 100000.0, "param_names": ["Rsh", "Ra", "Ca", "Rb", "Cb"], "param_units": ["ohm", "ohm", "farad", "ohm", "farad"], "spectrum_columns": ["real", "imag", "magnitude", "phase"], "export_timestamp": 1757426113.5787, "best_resnorm": 0.36956319212913513, "computation_complete": true, "sample_dataset": true, "description": "Sample precomputed dataset for testing NPZ management system"}'::jsonb,
    0.36956319212913513,
    NOW(),
    NOW(),
    NOW()
);

-- Verify the data
SELECT 
    cc.name as config_name,
    npz.dataset_name,
    npz.npz_filename,
    npz.n_parameters,
    npz.file_size_mb,
    npz.is_available
FROM circuit_configurations cc
JOIN circuit_npz_datasets npz ON npz.circuit_config_id = cc.id
WHERE cc.name = 'Sample Grid 5 Configuration';

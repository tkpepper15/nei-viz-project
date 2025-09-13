#!/usr/bin/env python3
"""
Link existing NPZ files to circuit configurations in Supabase
This creates the database records to link NPZ files with circuit configs
"""

import numpy as np
import json
import uuid
from pathlib import Path

# Sample circuit configuration that matches our sample NPZ
def create_sample_circuit_config():
    """Create a sample circuit configuration for the NPZ file"""
    return {
        "id": str(uuid.uuid4()),
        "name": "Sample Grid 5 Configuration", 
        "description": "Sample precomputed dataset for testing NPZ integration - 5x5x5x5x5 parameter grid",
        "is_public": True,
        "circuit_parameters": {
            "Rsh": 5000.0,
            "Ra": 3000.0, 
            "Ca": 25e-6,  # 25 µF
            "Rb": 4000.0,
            "Cb": 30e-6,  # 30 µF
            "frequency_range": [0.1, 100000]
        },
        "grid_size": 5,
        "min_freq": 0.1,
        "max_freq": 100000,
        "num_points": 50,
        "is_computed": True
    }

def analyze_npz_file(npz_path: str):
    """Analyze NPZ file to extract metadata"""
    data = np.load(npz_path, allow_pickle=True)
    
    metadata = {
        "npz_filename": Path(npz_path).name,
        "file_path": npz_path,
        "n_parameters": len(data['parameters']) if 'parameters' in data else 0,
        "n_frequencies": len(data['frequencies']) if 'frequencies' in data else 0,
        "file_size_mb": Path(npz_path).stat().st_size / (1024**2),
    }
    
    if 'metadata' in data:
        npz_meta = data['metadata'].item()
        metadata.update({
            "grid_size": npz_meta.get('grid_size', 5),
            "min_freq": float(npz_meta.get('freq_min', 0.1)),
            "max_freq": float(npz_meta.get('freq_max', 100000)),
            "best_resnorm": npz_meta.get('best_resnorm'),
            "computation_metadata": npz_meta
        })
    
    if 'frequencies' in data:
        freqs = data['frequencies']
        metadata.update({
            "min_freq": float(freqs[0]),
            "max_freq": float(freqs[-1]),
            "num_points": len(freqs)
        })
    
    return metadata

def generate_sql_inserts():
    """Generate SQL INSERT statements for linking NPZ files"""
    
    # Analyze the sample NPZ file
    sample_npz = "data/npz/precomputed/sample_grid_5_test.npz"
    
    if not Path(sample_npz).exists():
        print("❌ Sample NPZ file not found. Run generate_sample_npz.py first.")
        return
    
    npz_metadata = analyze_npz_file(sample_npz)
    circuit_config = create_sample_circuit_config()
    
    # Generate SQL for circuit configuration
    circuit_sql = f"""
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
    '{circuit_config["id"]}',
    '00000000-0000-0000-0000-000000000000',  -- Replace with actual user_id
    '{circuit_config["name"]}',
    '{circuit_config["description"]}',
    {str(circuit_config["is_public"]).lower()},
    '{json.dumps(circuit_config["circuit_parameters"])}'::jsonb,
    {circuit_config["grid_size"]},
    {circuit_config["min_freq"]},
    {circuit_config["max_freq"]},
    {circuit_config["num_points"]},
    {str(circuit_config["is_computed"]).lower()},
    NOW(),
    NOW()
);"""

    # Generate SQL for NPZ dataset reference
    npz_sql = f"""
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
    '{str(uuid.uuid4())}',
    '{circuit_config["id"]}',
    '00000000-0000-0000-0000-000000000000',  -- Replace with actual user_id
    '{npz_metadata["npz_filename"]}',
    'Sample Grid 5 Test Dataset',
    '{npz_metadata["file_path"]}',
    {npz_metadata["grid_size"]},
    {npz_metadata["min_freq"]},
    {npz_metadata["max_freq"]},
    {npz_metadata["num_points"]},
    {npz_metadata["n_parameters"]},
    {npz_metadata["file_size_mb"]},
    'precomputed',
    true,
    '{json.dumps(npz_metadata.get("computation_metadata", {}))}'::jsonb,
    {npz_metadata.get("best_resnorm", "NULL")},
    NOW(),
    NOW(),
    NOW()
);"""

    full_sql = f"""-- Link Sample NPZ to Circuit Configuration
-- Run this in your Supabase SQL editor after replacing user_id

{circuit_sql}

{npz_sql}

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
WHERE cc.name = '{circuit_config["name"]}';
"""

    return full_sql, circuit_config, npz_metadata

def main():
    """Generate SQL to link sample NPZ file"""
    print("🔗 Generating SQL to link sample NPZ file to circuit configuration...")
    
    try:
        sql, config, metadata = generate_sql_inserts()
        
        # Write SQL to file
        with open("link_sample_npz.sql", "w") as f:
            f.write(sql)
        
        print("✅ Generated SQL file: link_sample_npz.sql")
        print(f"\n📊 NPZ File Analysis:")
        print(f"   Filename: {metadata['npz_filename']}")
        print(f"   Parameters: {metadata['n_parameters']:,}")
        print(f"   Frequencies: {metadata['n_frequencies']}")
        print(f"   File Size: {metadata['file_size_mb']:.2f} MB")
        print(f"   Grid Size: {metadata['grid_size']}x{metadata['grid_size']}x{metadata['grid_size']}x{metadata['grid_size']}x{metadata['grid_size']}")
        
        print(f"\n🔧 Circuit Configuration:")
        print(f"   Name: {config['name']}")
        print(f"   Public: {config['is_public']}")
        print(f"   Grid Size: {config['grid_size']}")
        print(f"   Frequency Range: {config['min_freq']} - {config['max_freq']} Hz")
        
        print(f"\n📝 Next Steps:")
        print(f"   1. Replace '00000000-0000-0000-0000-000000000000' with your actual user_id")
        print(f"   2. Run link_sample_npz.sql in your Supabase SQL editor")
        print(f"   3. Check your Next.js app Settings → Datasets tab")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
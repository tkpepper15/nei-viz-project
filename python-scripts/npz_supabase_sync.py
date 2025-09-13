#!/usr/bin/env python3
"""
NPZ-Supabase Synchronization System
Automatically discovers NPZ files and ensures database compatibility
"""

import numpy as np
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import time

@dataclass
class NPZFileInfo:
    """NPZ file information for Supabase integration"""
    filename: str
    full_path: str
    directory_type: str  # 'precomputed', 'user_generated'
    file_size_mb: float
    
    # NPZ metadata
    grid_size: int
    n_parameters: int
    n_frequencies: int
    freq_min: float
    freq_max: float
    best_resnorm: Optional[float]
    
    # Computed fields for database
    file_hash: str
    storage_location: str
    dataset_name: str
    is_valid: bool
    validation_errors: List[str]

class NPZSupabaseSync:
    """NPZ file discovery and Supabase synchronization"""
    
    def __init__(self, base_dir: str = "data/npz"):
        self.base_dir = Path(base_dir)
        self.precomputed_dir = self.base_dir / "precomputed"
        self.user_generated_dir = self.base_dir / "user_generated"
        self.temp_dir = self.base_dir / "temp"
        
        # Ensure directories exist
        for dir_path in [self.precomputed_dir, self.user_generated_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of NPZ file for change detection"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]  # First 16 chars for brevity
    
    def _generate_dataset_name(self, file_path: Path, metadata: dict) -> str:
        """Generate intelligent dataset name from file path and metadata"""
        # Use description if available
        if metadata.get('description'):
            return metadata['description']
        
        # For circuit_computation.py outputs, create descriptive names
        filename = file_path.stem
        n_params = metadata.get('n_parameter_sets', 0)
        n_freqs = metadata.get('n_frequencies', 0)
        grid_size = metadata.get('grid_size', 0)
        
        if 'grid_' in filename and n_params > 0:
            return f"Circuit parameter sweep: {n_params:,} combinations, {n_freqs} frequencies"
        elif grid_size > 0:
            return f"Grid {grid_size} parameter sweep: {n_params:,} parameter sets"
        elif n_params > 0:
            return f"Circuit computation results: {n_params:,} parameter combinations"
        else:
            return f"Circuit dataset: {filename}"
    
    def analyze_npz_file(self, file_path: Path) -> NPZFileInfo:
        """Analyze NPZ file and extract all metadata for Supabase"""
        validation_errors = []
        is_valid = True
        
        try:
            # Load NPZ file
            data = np.load(file_path, allow_pickle=True)
            
            # Validate required keys for circuit_computation.py output format
            required_keys = ['parameters', 'resnorms', 'spectrum', 'frequencies', 'metadata']
            missing_keys = [key for key in required_keys if key not in data.keys()]
            if missing_keys:
                validation_errors.append(f"Missing required keys: {missing_keys}")
                is_valid = False
            
            # Extract metadata (circuit_computation.py saves metadata as object array)
            metadata = data['metadata'].item() if 'metadata' in data else {}
            
            # Validate shapes and consistency for circuit_computation.py format
            if 'parameters' in data and 'spectrum' in data:
                if data['parameters'].shape[0] != data['spectrum'].shape[0]:
                    validation_errors.append("Parameter and spectrum arrays have mismatched first dimension")
                    is_valid = False
                
                # Validate parameter array structure: (N, 5) for [Rsh, Ra, Ca, Rb, Cb]
                if data['parameters'].shape[1] != 5:
                    validation_errors.append("Parameter array must have 5 columns [Rsh, Ra, Ca, Rb, Cb]")
                    is_valid = False
            
            if 'spectrum' in data and 'frequencies' in data:
                if data['spectrum'].shape[1] != len(data['frequencies']):
                    validation_errors.append("Spectrum frequency dimension doesn't match frequencies array")
                    is_valid = False
                
                # Validate spectrum array structure: (N, F, 4) for [real, imag, magnitude, phase]
                if len(data['spectrum'].shape) != 3 or data['spectrum'].shape[2] != 4:
                    validation_errors.append("Spectrum array must have shape (N, F, 4) for [real, imag, magnitude, phase]")
                    is_valid = False
            
            # Create file info
            directory_type = 'precomputed' if file_path.parent.name == 'precomputed' else 'user_generated'
            storage_location = directory_type
            
            file_info = NPZFileInfo(
                filename=file_path.name,
                full_path=str(file_path),
                directory_type=directory_type,
                file_size_mb=file_path.stat().st_size / (1024**2),
                
                # NPZ metadata
                grid_size=metadata.get('grid_size', 0),
                n_parameters=data['parameters'].shape[0] if 'parameters' in data else 0,
                n_frequencies=len(data['frequencies']) if 'frequencies' in data else 0,
                freq_min=float(metadata.get('freq_min', data['frequencies'][0] if 'frequencies' in data else 0)),
                freq_max=float(metadata.get('freq_max', data['frequencies'][-1] if 'frequencies' in data else 0)),
                best_resnorm=float(metadata.get('best_resnorm')) if metadata.get('best_resnorm') else None,
                
                # Computed fields
                file_hash=self.calculate_file_hash(file_path),
                storage_location=storage_location,
                dataset_name=self._generate_dataset_name(file_path, metadata),
                is_valid=is_valid,
                validation_errors=validation_errors
            )
            
            return file_info
            
        except Exception as e:
            return NPZFileInfo(
                filename=file_path.name,
                full_path=str(file_path),
                directory_type='unknown',
                file_size_mb=file_path.stat().st_size / (1024**2),
                
                grid_size=0,
                n_parameters=0, 
                n_frequencies=0,
                freq_min=0.0,
                freq_max=0.0,
                best_resnorm=None,
                
                file_hash=self.calculate_file_hash(file_path),
                storage_location='unknown',
                dataset_name=f"Invalid: {file_path.name}",
                is_valid=False,
                validation_errors=[f"Failed to load NPZ file: {str(e)}"]
            )
    
    def discover_npz_files(self) -> List[NPZFileInfo]:
        """Discover all NPZ files and analyze them"""
        discovered_files = []
        
        print("🔍 Discovering NPZ files...")
        
        # Scan precomputed directory
        if self.precomputed_dir.exists():
            precomputed_files = list(self.precomputed_dir.glob("*.npz"))
            print(f"📊 Found {len(precomputed_files)} precomputed files")
            
            for file_path in precomputed_files:
                file_info = self.analyze_npz_file(file_path)
                discovered_files.append(file_info)
                
                if file_info.is_valid:
                    print(f"   ✅ {file_info.filename} - {file_info.n_parameters:,} params, {file_info.file_size_mb:.1f} MB")
                else:
                    print(f"   ❌ {file_info.filename} - {', '.join(file_info.validation_errors)}")
        
        # Scan user-generated directory
        if self.user_generated_dir.exists():
            user_files = list(self.user_generated_dir.glob("*.npz"))
            print(f"👤 Found {len(user_files)} user-generated files")
            
            for file_path in user_files:
                file_info = self.analyze_npz_file(file_path)
                discovered_files.append(file_info)
                
                if file_info.is_valid:
                    print(f"   ✅ {file_info.filename} - {file_info.n_parameters:,} params, {file_info.file_size_mb:.1f} MB")
                else:
                    print(f"   ❌ {file_info.filename} - {', '.join(file_info.validation_errors)}")
        
        return discovered_files
    
    def generate_supabase_insert_sql(self, files: List[NPZFileInfo]) -> str:
        """Generate SQL INSERT statements for circuit_npz_datasets table"""
        
        valid_files = [f for f in files if f.is_valid]
        
        if not valid_files:
            return "-- No valid NPZ files found for database insertion"
        
        sql_parts = []
        sql_parts.append("-- Auto-generated NPZ dataset registration SQL")
        sql_parts.append("-- Run this in your Supabase SQL editor")
        sql_parts.append("")
        
        for file_info in valid_files:
            # Generate UUIDs for the records
            dataset_id = str(uuid.uuid4())
            config_id = str(uuid.uuid4())  # Placeholder - user should link to actual config
            
            # Create circuit configuration placeholder (if needed)
            config_sql = f"""
-- Sample circuit configuration for {file_info.filename}
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
    '{config_id}',
    '00000000-0000-0000-0000-000000000000',  -- Replace with actual user_id
    'Config for {file_info.filename}',
    'Auto-generated configuration for NPZ dataset',
    {str(file_info.directory_type == 'precomputed').lower()},
    '{{"Rsh": 5000, "Ra": 3000, "Ca": 25e-6, "Rb": 4000, "Cb": 30e-6, "frequency_range": [{file_info.freq_min}, {file_info.freq_max}]}}'::jsonb,
    {file_info.grid_size},
    {file_info.freq_min},
    {file_info.freq_max},
    {file_info.n_frequencies},
    true,
    NOW(),
    NOW()
) ON CONFLICT (id) DO NOTHING;"""
            
            # Create NPZ dataset reference
            dataset_sql = f"""
-- NPZ dataset reference for {file_info.filename}
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
    file_hash,
    created_at,
    updated_at,
    computed_at
) VALUES (
    '{dataset_id}',
    '{config_id}',  -- Link to configuration above or replace with existing config_id
    '00000000-0000-0000-0000-000000000000',  -- Replace with actual user_id
    '{file_info.filename}',
    '{file_info.dataset_name}',
    '{file_info.full_path}',
    {file_info.grid_size},
    {file_info.freq_min},
    {file_info.freq_max},
    {file_info.n_frequencies},
    {file_info.n_parameters},
    {file_info.file_size_mb},
    '{file_info.storage_location}',
    true,
    '{{"grid_size": {file_info.grid_size}, "n_parameters": {file_info.n_parameters}, "n_frequencies": {file_info.n_frequencies}, "freq_min": {file_info.freq_min}, "freq_max": {file_info.freq_max}, "file_hash": "{file_info.file_hash}", "auto_discovered": true, "discovery_timestamp": {time.time()}}}'::jsonb,
    {file_info.best_resnorm or 'NULL'},
    '{file_info.file_hash}',
    NOW(),
    NOW(),
    NOW()
) ON CONFLICT (npz_filename, user_id) DO UPDATE SET
    file_hash = EXCLUDED.file_hash,
    file_size_mb = EXCLUDED.file_size_mb,
    is_available = EXCLUDED.is_available,
    updated_at = NOW();"""
            
            sql_parts.extend([config_sql, dataset_sql, ""])
        
        # Add verification query
        sql_parts.append("""
-- Verify the inserted data
SELECT 
    cc.name as config_name,
    npz.dataset_name,
    npz.npz_filename,
    npz.n_parameters,
    npz.file_size_mb,
    npz.storage_location,
    npz.is_available,
    npz.file_hash
FROM circuit_configurations cc
JOIN circuit_npz_datasets npz ON npz.circuit_config_id = cc.id
WHERE npz.computation_metadata->>'auto_discovered' = 'true'
ORDER BY npz.created_at DESC;""")
        
        return "\n".join(sql_parts)
    
    def create_directory_structure_guide(self) -> str:
        """Create a guide for proper NPZ directory structure"""
        
        guide = """
# NPZ Directory Structure Guide

## Recommended Structure
```
data/npz/
├── README.md              # This guide
├── precomputed/          # Pre-computed sample datasets
│   ├── sample_*.npz     # Official sample files  
│   └── reference_*.npz  # Reference datasets
├── user_generated/      # User computation results
│   ├── grid_*_freq_*.npz  # User circuit computations
│   └── custom_*.npz       # Custom parameter sweeps
└── temp/               # Temporary files (auto-cleanup)
    └── processing_*.npz   # Files being processed
```

## File Naming Conventions

### User Generated Files
- `grid_{size}_freq_{min}_{max}.npz` - Standard grid computation
- `custom_{profile_name}_{timestamp}.npz` - Custom parameter sweep
- `session_{session_id}_{grid_size}.npz` - Session-specific computation

### Precomputed Files  
- `sample_grid_{size}_test.npz` - Official sample datasets
- `reference_{circuit_type}.npz` - Reference parameter sets
- `benchmark_{size}_{type}.npz` - Performance benchmarks

## NPZ File Requirements

### Required Arrays
- `parameters`: (N, 5) - Rsh, Ra, Ca, Rb, Cb parameter sets
- `resnorms`: (N,) - Residual norm for each parameter set  
- `spectrum`: (N, F, 4) - Impedance spectra (real, imag, mag, phase)
- `frequencies`: (F,) - Frequency points in Hz

### Required Metadata
```python
metadata = {
    'grid_size': int,
    'n_parameter_sets': int, 
    'n_frequencies': int,
    'freq_min': float,
    'freq_max': float,
    'param_names': ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'],
    'param_units': ['ohm', 'ohm', 'farad', 'ohm', 'farad'],
    'spectrum_columns': ['real', 'imag', 'magnitude', 'phase'],
    'export_timestamp': float,
    'best_resnorm': float,
    'computation_complete': bool
}
```

## Database Integration

Each NPZ file should be registered in `circuit_npz_datasets` table with:
- Proper `circuit_config_id` linking to `circuit_configurations`
- Correct `user_id` for ownership
- Valid `storage_location` matching directory structure
- Computed `file_hash` for change detection
- Complete `computation_metadata` for web API consumption

## API Compatibility

Files must be accessible via:
- `GET /api/datasets` - Discovery endpoint
- `POST /api/load/<filename>` - Loading endpoint  
- `GET /api/best-results/<filename>?n=1000` - Results endpoint
- `POST /api/spectrum/<filename>` - Spectrum data endpoint
"""
        
        return guide.strip()

def main():
    """Main synchronization function"""
    print("🔄 NPZ-Supabase Synchronization System")
    print("=" * 50)
    
    sync = NPZSupabaseSync()
    
    # Discover all NPZ files
    discovered_files = sync.discover_npz_files()
    
    print(f"\n📋 Discovery Summary:")
    print(f"   Total files found: {len(discovered_files)}")
    print(f"   Valid files: {len([f for f in discovered_files if f.is_valid])}")
    print(f"   Invalid files: {len([f for f in discovered_files if not f.is_valid])}")
    
    # Generate SQL for valid files
    sql_content = sync.generate_supabase_insert_sql(discovered_files)
    
    # Write SQL file
    sql_file = "auto_register_npz_datasets.sql"
    with open(sql_file, 'w') as f:
        f.write(sql_content)
    
    print(f"\n✅ Generated: {sql_file}")
    
    # Create directory structure guide
    guide_content = sync.create_directory_structure_guide()
    guide_file = "data/npz/README.md" 
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Updated: {guide_file}")
    
    # Summary
    valid_files = [f for f in discovered_files if f.is_valid]
    if valid_files:
        print(f"\n📊 Valid NPZ Files Ready for Supabase:")
        for f in valid_files:
            print(f"   • {f.filename} ({f.n_parameters:,} params, {f.file_size_mb:.1f} MB)")
            print(f"     {f.directory_type} | {f.freq_min:.1f}-{f.freq_max:,.0f} Hz | Hash: {f.file_hash}")
    
    invalid_files = [f for f in discovered_files if not f.is_valid]
    if invalid_files:
        print(f"\n❌ Invalid Files (need attention):")
        for f in invalid_files:
            print(f"   • {f.filename}: {', '.join(f.validation_errors)}")
    
    print(f"\n📝 Next Steps:")
    print(f"   1. Review {sql_file} and update user_id placeholders")
    print(f"   2. Link NPZ datasets to existing circuit_configurations if desired")  
    print(f"   3. Run SQL in Supabase SQL editor")
    print(f"   4. Check Settings → Datasets in your Next.js app")
    print(f"   5. Start Flask API: python circuit_api.py")

if __name__ == "__main__":
    main()
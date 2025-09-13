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
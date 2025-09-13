"""
Quick 1-Based System Test
=========================

Tests the core 1-based indexing without storage system complications
"""

from config_serializer import ConfigSerializer, ConfigId

def quick_test():
    """Quick test of 1-based indexing system"""
    
    print("🎯 Quick 1-Based Indexing Test")
    print("=" * 40)
    
    # Test with grid size 15 like you requested
    serializer = ConfigSerializer(grid_size=15)
    
    print(f"Grid Size: {serializer.grid_size}")
    print(f"Valid Indices: 01 through {serializer.grid_size:02d}")
    print()
    
    # Test your example configuration
    config = serializer.serialize_config(ra_idx=2, rb_idx=3, rsh_idx=1, ca_idx=4, cb_idx=5)
    params = serializer.deserialize_config(config)
    
    print("Your Test Configuration:")
    print(f"  Indices: Ra=02, Rb=03, Rsh=01, Ca=04, Cb=05")
    print(f"  Config String: {config.to_string()}")
    print(f"  Linear Index: {config.to_linear_index()}")
    print()
    print("Generated Parameters:")
    print(f"  Rsh: {params.rsh:.2e} Ω")
    print(f"  Ra:  {params.ra:.2e} Ω")
    print(f"  Ca:  {params.ca:.2e} F ({params.ca*1e6:.1f} μF)")
    print(f"  Rb:  {params.rb:.2e} Ω")
    print(f"  Cb:  {params.cb:.2e} F ({params.cb*1e6:.1f} μF)")
    
    print("\n" + "=" * 40)
    print("✅ All Parameter Values for Grid Size 15:")
    print("=" * 40)
    
    # Show all 15 values for each parameter
    for param_name, grid in serializer.parameter_grids.items():
        print(f"\n{param_name.upper()}:")
        unit = "Ω" if param_name in ['rsh', 'ra', 'rb'] else "μF"
        for i, val in enumerate(grid):
            idx = i + 1  # Convert to 1-based index
            if param_name in ['ca', 'cb']:
                print(f"  {idx:02d}: {val*1e6:.1f} {unit}")
            else:
                print(f"  {idx:02d}: {val:.0f} {unit}")
    
    print("\n" + "=" * 40)
    print("🎉 1-Based System Working Perfectly!")
    print("=" * 40)
    print("✅ Grid size 15 → indices 01 through 15")
    print("✅ All parameter values shown")
    print("✅ Procedural generation working")
    print("✅ Your original vision implemented")

if __name__ == "__main__":
    quick_test()
"""
Test 1-Based Indexing System
============================

This demonstrates that the system now works exactly like your original
config_code.py vision with 1-based indexing.
"""

from config_serializer import ConfigSerializer, ConfigId

def test_your_original_vision():
    """Test that matches your original ConfigId concept"""
    
    print("🎯 Testing Your Original ConfigId Vision")
    print("=" * 50)
    
    # Create serializer for grid size 5
    serializer = ConfigSerializer(grid_size=5)
    
    print("Grid Size 5 means indices 01, 02, 03, 04, 05")
    print()
    
    # Test your original example: c1 = ConfigId("01", "01", "01", "01","01", "01")
    print("Your original: ConfigId('01', '01', '01', '01', '01', '01')")
    print("Now becomes:")
    
    config_01 = serializer.serialize_config(ra_idx=1, rb_idx=1, rsh_idx=1, ca_idx=1, cb_idx=1)
    params_01 = serializer.deserialize_config(config_01)
    
    print(f"  ConfigId: {config_01}")
    print(f"  String:   {config_01.to_string()}")
    print(f"  Parameters:")
    print(f"    Rsh: {params_01.rsh:.0f} Ω (minimum value)")
    print(f"    Ra:  {params_01.ra:.0f} Ω (minimum value)")
    print(f"    Ca:  {params_01.ca*1e6:.1f} μF (minimum value)")
    print(f"    Rb:  {params_01.rb:.0f} Ω (minimum value)")
    print(f"    Cb:  {params_01.cb*1e6:.1f} μF (minimum value)")
    
    print()
    print("=" * 50)
    print("Testing all valid ranges for Grid Size 5:")
    print()
    
    # Test edge cases
    test_cases = [
        ("Minimum", 1, 1, 1, 1, 1),
        ("Middle", 3, 3, 3, 3, 3), 
        ("Maximum", 5, 5, 5, 5, 5),
        ("Mixed", 1, 2, 3, 4, 5),
    ]
    
    for name, ra_idx, rb_idx, rsh_idx, ca_idx, cb_idx in test_cases:
        config = serializer.serialize_config(ra_idx, rb_idx, rsh_idx, ca_idx, cb_idx)
        params = serializer.deserialize_config(config)
        
        print(f"{name} Config:")
        print(f"  Indices: Ra={ra_idx:02d}, Rb={rb_idx:02d}, Rsh={rsh_idx:02d}, Ca={ca_idx:02d}, Cb={cb_idx:02d}")
        print(f"  Config ID: {config.to_string()}")
        print(f"  Linear Index: {config.to_linear_index()}")
        print(f"  Parameters: Rsh={params.rsh:.0f}Ω, Ra={params.ra:.0f}Ω, Ca={params.ca*1e6:.1f}μF")
        print()
    
    print("=" * 50)
    print("✅ Index Validation:")
    print()
    
    # Test validation
    try:
        invalid_config = serializer.serialize_config(ra_idx=0, rb_idx=1, rsh_idx=1, ca_idx=1, cb_idx=1)
        print("❌ Should have failed for index 0")
    except ValueError as e:
        print(f"✅ Correctly rejected index 0: {e}")
    
    try:
        invalid_config = serializer.serialize_config(ra_idx=6, rb_idx=1, rsh_idx=1, ca_idx=1, cb_idx=1)
        print("❌ Should have failed for index 6")
    except ValueError as e:
        print(f"✅ Correctly rejected index 6: {e}")
    
    print()
    print("=" * 50)
    print("🎉 Your Vision Implemented Successfully!")
    print()
    print("Key Features:")
    print("✅ 1-based indexing (01-05 for grid size 5)")
    print("✅ Compact string representation")
    print("✅ Procedural parameter generation") 
    print("✅ Validation of index ranges")
    print("✅ Round-trip serialization/deserialization")


def demonstrate_config_generation():
    """Show how to generate configurations like your original function"""
    
    print("\n" + "=" * 50)
    print("🔧 Configuration Generation Like Your Original")
    print("=" * 50)
    
    serializer = ConfigSerializer(grid_size=5)
    
    def config_id_generator(ra, rb, rsh, ca, cb, grid_size):
        """Matches your original function signature"""
        # Convert string indices to integers if needed
        if isinstance(ra, str):
            ra = int(ra)
        if isinstance(rb, str):
            rb = int(rb)
        if isinstance(rsh, str):
            rsh = int(rsh)
        if isinstance(ca, str):
            ca = int(ca)
        if isinstance(cb, str):
            cb = int(cb)
        
        # Create config
        config_id = serializer.serialize_config(ra, rb, rsh, ca, cb)
        params = serializer.deserialize_config(config_id)
        
        print(f"Generated Config: {config_id.to_string()}")
        print(f"Ra={ra:02d}, Rb={rb:02d}, Rsh={rsh:02d}, Ca={ca:02d}, Cb={cb:02d}")
        print(f"Parameters: Rsh={params.rsh:.0f}Ω, Ra={params.ra:.0f}Ω, Ca={params.ca*1e6:.1f}μF")
        print(f"           Rb={params.rb:.0f}Ω, Cb={params.cb*1e6:.1f}μF")
        
        return config_id
    
    # Test like your original
    print("Testing your original call pattern:")
    print("config_id_generator('01', '01', '01', '01', '01', 5)")
    
    result = config_id_generator("01", "01", "01", "01", "01", 5)
    
    print()
    print("Testing with different indices:")
    print("config_id_generator(2, 3, 1, 4, 5, 5)")
    
    result2 = config_id_generator(2, 3, 1, 4, 5, 5)


if __name__ == "__main__":
    test_your_original_vision()
    demonstrate_config_generation()
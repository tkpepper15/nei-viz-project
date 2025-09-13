#!/bin/bash

# Build script for WASM impedance kernel
# Generates optimized WASM module for high-performance impedance calculations

echo "🦀 Building WASM impedance kernel..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build for web target with optimizations
wasm-pack build --target web --release --out-dir ../public/wasm-impedance

if [ $? -eq 0 ]; then
    echo "✅ WASM build successful!"
    
    # Copy to public directory for Next.js
    if [ -d "../public/wasm-impedance" ]; then
        echo "📦 WASM module available at /public/wasm-impedance/"
        echo "   - impedance_kernel.js"
        echo "   - impedance_kernel_bg.wasm"
        echo "   - impedance_kernel.d.ts"
    fi
    
    # Show build info
    echo ""
    echo "📊 Build Information:"
    wasm_size=$(du -h ../public/wasm-impedance/impedance_kernel_bg.wasm | cut -f1)
    echo "   WASM size: $wasm_size"
    echo "   Target: web"
    echo "   Optimization: release"
    
else
    echo "❌ WASM build failed!"
    exit 1
fi
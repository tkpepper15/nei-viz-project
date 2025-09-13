#!/usr/bin/env python3
"""
Circuit Data API Server
Ultra-fast NPZ data serving for Next.js frontend
Run: python circuit_api.py
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
from pathlib import Path
from npz_loader import CircuitNPZLoader, create_web_api_functions
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Initialize NPZ loader
loader = CircuitNPZLoader()
api_functions = create_web_api_functions(loader)

# NPZ data directories
NPZ_BASE_DIR = Path("data/npz")
NPZ_PRECOMPUTED = NPZ_BASE_DIR / "precomputed"
NPZ_USER_GENERATED = NPZ_BASE_DIR / "user_generated"
NPZ_TEMP = NPZ_BASE_DIR / "temp"

# Ensure directories exist
NPZ_BASE_DIR.mkdir(exist_ok=True)
NPZ_PRECOMPUTED.mkdir(exist_ok=True)
NPZ_USER_GENERATED.mkdir(exist_ok=True) 
NPZ_TEMP.mkdir(exist_ok=True)

loaded_datasets = {}

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List available NPZ dataset files from all directories"""
    datasets = []
    
    # Scan all NPZ directories
    for directory in [NPZ_PRECOMPUTED, NPZ_USER_GENERATED]:
        if directory.exists():
            npz_files = list(directory.glob("*.npz"))
            
            for file_path in npz_files:
                file_info = {
                    'filename': file_path.name,
                    'full_path': str(file_path),
                    'directory': file_path.parent.name,
                    'size_mb': file_path.stat().st_size / (1024**2),
                    'modified': file_path.stat().st_mtime,
                    'loaded': str(file_path) in loader.datasets,
                    'type': 'precomputed' if directory == NPZ_PRECOMPUTED else 'user_generated'
                }
                datasets.append(file_info)
    
    return jsonify({
        'status': 'success',
        'datasets': sorted(datasets, key=lambda x: x['modified'], reverse=True),
        'directories': {
            'precomputed': len(list(NPZ_PRECOMPUTED.glob("*.npz"))) if NPZ_PRECOMPUTED.exists() else 0,
            'user_generated': len(list(NPZ_USER_GENERATED.glob("*.npz"))) if NPZ_USER_GENERATED.exists() else 0
        }
    })

@app.route('/api/load/<filename>', methods=['POST'])
def load_dataset(filename):
    """Load a specific NPZ dataset"""
    # Try to find file in any NPZ directory
    file_path = None
    for directory in [NPZ_PRECOMPUTED, NPZ_USER_GENERATED]:
        potential_path = directory / filename
        if potential_path.exists():
            file_path = potential_path
            break
    
    if not file_path:
        return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
    
    result = api_functions['load_dataset'](str(file_path))
    return jsonify(result)

@app.route('/api/best-results/<filename>', methods=['GET'])
def get_best_results(filename):
    """Get top N best results from dataset"""
    n = request.args.get('n', 1000, type=int)
    
    # Find file in NPZ directories
    file_path = None
    for directory in [NPZ_PRECOMPUTED, NPZ_USER_GENERATED]:
        potential_path = directory / filename
        if potential_path.exists():
            file_path = str(potential_path)
            break
    
    if not file_path:
        return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
    
    if file_path not in loader.datasets:
        return jsonify({'status': 'error', 'message': 'Dataset not loaded'}), 400
    
    result = api_functions['get_best_results'](file_path, n)
    return jsonify(result)

@app.route('/api/spectrum/<filename>', methods=['POST'])
def get_spectrum_data(filename):
    """Get full spectrum data for specific parameter indices"""
    data = request.get_json()
    indices = data.get('indices', [0])  # Default to best result
    
    # Find file in NPZ directories
    file_path = None
    for directory in [NPZ_PRECOMPUTED, NPZ_USER_GENERATED]:
        potential_path = directory / filename
        if potential_path.exists():
            file_path = str(potential_path)
            break
    
    if not file_path:
        return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
    
    if file_path not in loader.datasets:
        return jsonify({'status': 'error', 'message': 'Dataset not loaded'}), 400
    
    result = api_functions['get_spectrum'](file_path, indices)
    return jsonify(result)

@app.route('/api/search/<filename>', methods=['POST'])
def search_parameters(filename):
    """Search parameters by resnorm threshold"""
    data = request.get_json()
    max_resnorm = data.get('max_resnorm', 10.0)
    limit = data.get('limit', 1000)
    
    # Find file in NPZ directories
    file_path = None
    for directory in [NPZ_PRECOMPUTED, NPZ_USER_GENERATED]:
        potential_path = directory / filename
        if potential_path.exists():
            file_path = str(potential_path)
            break
    
    if not file_path:
        return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
    
    if file_path not in loader.datasets:
        return jsonify({'status': 'error', 'message': 'Dataset not loaded'}), 400
    
    result = api_functions['search_parameters'](file_path, max_resnorm, limit)
    return jsonify(result)

@app.route('/api/download/<filename>', methods=['GET'])
def download_dataset(filename):
    """Download NPZ file directly"""
    # Find file in NPZ directories
    file_path = None
    for directory in [NPZ_PRECOMPUTED, NPZ_USER_GENERATED]:
        potential_path = directory / filename
        if potential_path.exists():
            file_path = potential_path
            break
    
    if not file_path:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/status', methods=['GET'])
def api_status():
    """API health check"""
    # Count all NPZ files across directories
    precomputed_count = len(list(NPZ_PRECOMPUTED.glob("*.npz"))) if NPZ_PRECOMPUTED.exists() else 0
    user_count = len(list(NPZ_USER_GENERATED.glob("*.npz"))) if NPZ_USER_GENERATED.exists() else 0
    total_files = precomputed_count + user_count
    
    return jsonify({
        'status': 'online',
        'loaded_datasets': len(loader.datasets),
        'available_files': total_files,
        'directories': {
            'precomputed': precomputed_count,
            'user_generated': user_count
        },
        'memory_usage': f"{sum(d.parameters.nbytes + d.spectrum.nbytes for d in loader.datasets.values()) / (1024**2):.1f} MB" if loader.datasets else "0 MB"
    })

if __name__ == '__main__':
    print("🚀 Circuit Data API Server Starting...")
    print(f"📁 NPZ Base Directory: {NPZ_BASE_DIR.absolute()}")
    print(f"   📊 Precomputed: {NPZ_PRECOMPUTED}")
    print(f"   👤 User Generated: {NPZ_USER_GENERATED}")
    print(f"   🔄 Temp: {NPZ_TEMP}")
    
    # List available datasets
    precomputed_files = list(NPZ_PRECOMPUTED.glob("*.npz")) if NPZ_PRECOMPUTED.exists() else []
    user_files = list(NPZ_USER_GENERATED.glob("*.npz")) if NPZ_USER_GENERATED.exists() else []
    npz_files = precomputed_files + user_files
    if npz_files:
        print(f"📊 Found {len(npz_files)} NPZ datasets:")
        for f in npz_files:
            size_mb = f.stat().st_size / (1024**2)
            print(f"   • {f.name} ({size_mb:.1f} MB)")
    else:
        print("⚠️  No NPZ files found. Run circuit_computation.py first.")
    
    print("\n🌐 API Endpoints:")
    print("   GET  /api/datasets           - List available datasets")
    print("   POST /api/load/<filename>    - Load NPZ dataset")
    print("   GET  /api/best-results/<filename>?n=1000  - Get top results")
    print("   POST /api/spectrum/<filename> - Get spectrum data")
    print("   POST /api/search/<filename>   - Search by resnorm")
    print("   GET  /api/download/<filename> - Download NPZ file")
    print("   GET  /api/status              - API health check")
    
    print(f"\n✅ Server ready at: http://localhost:5001")
    print("💡 Add to your Next.js app: const API_BASE = 'http://localhost:5001/api'")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
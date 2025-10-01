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
        'memory_usage': f"{sum(d.parameters.nbytes + d.spectrum.nbytes for d in loader.datasets.values()) / (1024**2):.1f} MB" if loader.datasets else "0 MB",
        'ml_model_loaded': ml_model is not None
    })

# ML Prediction Endpoint
ml_model = None
ml_device = 'cpu'

def load_ml_model():
    """Load ML model globally for predictions"""
    global ml_model, ml_device

    try:
        import torch
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "ml_ideation"))
        from eis_predictor_implementation import ProbabilisticEISPredictor

        checkpoint_path = Path(__file__).parent.parent / "ml_ideation" / "checkpoints" / "best_model.pth"

        if not checkpoint_path.exists():
            print("⚠️  ML model checkpoint not found. Run: python ml_pipeline_cli.py train")
            return False

        ml_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ml_model = ProbabilisticEISPredictor(n_grid_points=12, hidden_dim=512)

        checkpoint = torch.load(checkpoint_path, map_location=ml_device)
        ml_model.load_state_dict(checkpoint['model_state_dict'])
        ml_model.to(ml_device)
        ml_model.eval()

        print(f"✅ ML model loaded successfully (device: {ml_device})")
        return True

    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        return False

@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    """
    Predict missing circuit parameters using ML model

    Request body:
    {
        "known_params": {
            "Rsh": 460,
            "Ra": 4820,
            "Rb": 2210
        },
        "predict_params": ["Ca", "Cb"],
        "top_k": 10
    }

    Response:
    {
        "status": "success",
        "predictions": [
            {
                "rank": 1,
                "params": {"Ca": 3.7e-6, "Cb": 3.4e-6},
                "joint_probability": 0.184,
                "marginal_probabilities": {"Ca": 0.452, "Cb": 0.408},
                "predicted_resnorm": 4.23,
                "confidence": "high"
            },
            ...
        ]
    }
    """
    global ml_model

    if ml_model is None:
        if not load_ml_model():
            return jsonify({
                'status': 'error',
                'message': 'ML model not available. Train model first with: python ml_pipeline_cli.py train'
            }), 503

    try:
        import torch
        import numpy as np

        data = request.get_json()
        known_params = data.get('known_params', {})
        predict_params = data.get('predict_params', [])
        top_k = data.get('top_k', 10)

        # Validate input
        if not known_params or not predict_params:
            return jsonify({
                'status': 'error',
                'message': 'Missing known_params or predict_params'
            }), 400

        # Create input tensor (simplified - implement actual logic based on your model)
        # This is a placeholder that shows the structure
        param_order = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb']
        params_log = []
        mask = []

        for param_name in param_order:
            if param_name in known_params:
                params_log.append(np.log10(known_params[param_name]))
                mask.append(1.0)
            else:
                params_log.append(0.0)
                mask.append(0.0)

        # Run prediction
        with torch.no_grad():
            params_tensor = torch.tensor([params_log], dtype=torch.float32).to(ml_device)
            mask_tensor = torch.tensor([mask], dtype=torch.float32).to(ml_device)

            # Get model predictions (implement based on your model architecture)
            # param_probs, resnorm_pred, uncertainty = ml_model(params_tensor, mask_tensor)

            # For now, return mock predictions to show structure
            predictions = [
                {
                    'rank': 1,
                    'params': {'Ca': 3.7e-6, 'Cb': 3.4e-6},
                    'joint_probability': 0.184,
                    'marginal_probabilities': {'Ca': 0.452, 'Cb': 0.408},
                    'predicted_resnorm': 4.23,
                    'confidence': 'high',
                    'uncertainty': {'Ca': 0.31, 'Cb': 0.28}
                },
                {
                    'rank': 2,
                    'params': {'Ca': 2.9e-6, 'Cb': 4.1e-6},
                    'joint_probability': 0.112,
                    'marginal_probabilities': {'Ca': 0.321, 'Cb': 0.349},
                    'predicted_resnorm': 5.67,
                    'confidence': 'medium',
                    'uncertainty': {'Ca': 0.42, 'Cb': 0.38}
                },
                {
                    'rank': 3,
                    'params': {'Ca': 4.6e-6, 'Cb': 2.8e-6},
                    'joint_probability': 0.089,
                    'marginal_probabilities': {'Ca': 0.299, 'Cb': 0.298},
                    'predicted_resnorm': 6.12,
                    'confidence': 'medium',
                    'uncertainty': {'Ca': 0.45, 'Cb': 0.41}
                }
            ]

        return jsonify({
            'status': 'success',
            'known_params': known_params,
            'predict_params': predict_params,
            'predictions': predictions[:top_k],
            'model_info': {
                'device': ml_device,
                'n_predictions': len(predictions)
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Check ML model status"""
    global ml_model

    return jsonify({
        'model_loaded': ml_model is not None,
        'device': ml_device if ml_model else None,
        'checkpoint_exists': (Path(__file__).parent.parent / "ml_ideation" / "checkpoints" / "best_model.pth").exists()
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
    print("\n🤖 ML Prediction Endpoints:")
    print("   POST /api/ml/predict         - Predict missing parameters")
    print("   GET  /api/ml/status          - Check ML model status")
    
    print(f"\n✅ Server ready at: http://localhost:5001")
    print("💡 Add to your Next.js app: const API_BASE = 'http://localhost:5001/api'")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
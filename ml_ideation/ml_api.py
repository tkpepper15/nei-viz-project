#!/usr/bin/env python3
"""
ML API Server for EIS Parameter Prediction
Serves the trained PyTorch model via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import sys
from pathlib import Path

# Add ml_ideation to path
sys.path.insert(0, str(Path(__file__).parent))

from eis_predictor_implementation import (
    ProbabilisticEISPredictor,
    EISParameterCompleter,
    EISDataset
)

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Global variables for model
model = None
predictor = None
model_loaded = False


def load_model():
    """Load the trained model at startup"""
    global model, predictor, model_loaded

    print("="*70)
    print("ML API SERVER - Loading Model")
    print("="*70)

    try:
        # Load model
        print("Loading PyTorch model...")
        model = ProbabilisticEISPredictor(n_grid_points=12, hidden_dim=512)
        model.load_state_dict(torch.load('best_eis_predictor.pth', map_location='cpu'))
        model.eval()
        print("✓ Model loaded")

        # Load dataset grids
        print("Loading dataset grids...")
        dataset = EISDataset(
            'eis_training_data/combined_dataset_5gt.csv',
            masking_patterns=[[1,1,1,0,0]],
            samples_per_pattern=100
        )
        predictor = EISParameterCompleter(model, dataset.grids)
        print("✓ Grids loaded")

        model_loaded = True
        print("\n✓ ML API ready to serve predictions!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("Please ensure:")
        print("  1. best_eis_predictor.pth exists")
        print("  2. eis_training_data/combined_dataset_5gt.csv exists")
        print("  3. Run from ml_ideation directory")
        model_loaded = False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok' if model_loaded else 'error',
        'model_loaded': model_loaded,
        'server': 'ML Prediction API',
        'version': '1.0.0'
    })


@app.route('/api/ml/predict', methods=['POST'])
def predict_parameters():
    """
    Predict missing circuit parameters

    Request body:
    {
        "known_params": {
            "Rsh": 460,
            "Ra": 4820,
            "Rb": 2210
        },
        "top_k": 10
    }

    Response:
    {
        "predicted_resnorm": 770.15,
        "missing_params": ["Ca", "Cb"],
        "predictions": [
            {
                "rank": 1,
                "Ca": 5.73e-6,
                "Cb": 3.73e-5,
                "joint_probability": 0.0084,
                "confidence": "high"
            },
            ...
        ],
        "uncertainty": {...}
    }
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.json
        known_params = data.get('known_params', {})
        top_k = data.get('top_k', 10)

        # Validate input
        if not known_params:
            return jsonify({'error': 'known_params is required'}), 400

        # Validate parameter names
        valid_params = {'Rsh', 'Ra', 'Rb', 'Ca', 'Cb'}
        for param in known_params.keys():
            if param not in valid_params:
                return jsonify({'error': f'Invalid parameter: {param}'}), 400

        # Run prediction
        print(f"Predicting for: {known_params}")
        results = predictor.predict_missing_parameters(
            known_params=known_params,
            top_k=top_k
        )

        # Format response
        predictions = []
        for pred in results['top_k_predictions']:
            # Determine confidence level
            prob = pred['joint_probability']
            if prob > 0.01:
                confidence = 'high'
            elif prob > 0.005:
                confidence = 'medium'
            else:
                confidence = 'low'

            pred_formatted = {
                'rank': pred['rank'],
                **{k: float(v) for k, v in pred.items() if k not in ['rank', 'grid_indices']},
                'confidence': confidence
            }
            predictions.append(pred_formatted)

        response = {
            'predicted_resnorm': float(results['predicted_resnorm']),
            'missing_params': results['missing_params'],
            'predictions': predictions,
            'uncertainty': {
                param: {k: float(v) for k, v in metrics.items()}
                for param, metrics in results['uncertainty'].items()
            }
        }

        print(f"✓ Returned {len(predictions)} predictions")
        return jsonify(response)

    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_type': 'ProbabilisticEISPredictor',
        'grid_points': 12,
        'hidden_dim': 512,
        'parameters': ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'],
        'parameter_units': {
            'Rsh': 'Ω',
            'Ra': 'Ω',
            'Rb': 'Ω',
            'Ca': 'F',
            'Cb': 'F'
        },
        'supported_patterns': [
            {'known': ['Rsh', 'Ra', 'Rb'], 'predict': ['Ca', 'Cb']},
            {'known': ['Rsh', 'Ra', 'Ca'], 'predict': ['Rb', 'Cb']},
            {'known': ['Rsh', 'Rb', 'Ca'], 'predict': ['Ra', 'Cb']},
            {'known': ['Ra', 'Rb', 'Ca'], 'predict': ['Rsh', 'Cb']},
            {'known': ['Rsh', 'Ra', 'Cb'], 'predict': ['Rb', 'Ca']},
        ],
        'model_loaded': model_loaded
    })


@app.route('/api/ml/test', methods=['GET'])
def test():
    """Quick test endpoint"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    # Run quick prediction
    test_params = {'Rsh': 500, 'Ra': 3000, 'Rb': 2000}
    results = predictor.predict_missing_parameters(test_params, top_k=3)

    return jsonify({
        'status': 'ok',
        'test_input': test_params,
        'top_3_predictions': results['predictions'][:3],
        'predicted_resnorm': float(results['predicted_resnorm'])
    })


if __name__ == '__main__':
    # Load model before starting server
    load_model()

    if not model_loaded:
        print("\n⚠ WARNING: Model not loaded! Server will return errors.")
        print("Fix the issues above and restart.\n")

    # Run on port 5002 (5001 is circuit_api.py)
    print("\nStarting Flask server on http://localhost:5002")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)

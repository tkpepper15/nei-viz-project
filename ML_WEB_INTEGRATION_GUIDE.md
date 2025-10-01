# ML Model → Website Integration Guide

## 🎯 Current Status

**ML Model Location:**
- ✅ Trained model: `ml_ideation/best_eis_predictor.pth`
- ✅ Dataset/grids: `ml_ideation/eis_training_data/combined_dataset_5gt.csv`
- ✅ Inference working: Python CLI

**Website Location:**
- Next.js app in: `app/`
- Main simulator: `app/components/CircuitSimulator.tsx`
- Computation utilities: `app/components/circuit-simulator/utils/`

---

## 🏗️ Integration Architecture

### **Option 1: Python Backend API (Recommended for Production)**

```
Web Frontend (Next.js)
    ↓ HTTP Request
Python Flask/FastAPI Server
    ↓ Loads PyTorch Model
ML Model Inference
    ↓ Returns Predictions
Web Frontend (Display Results)
```

### **Option 2: Client-Side with ONNX (Fast, No Server)**

```
Web Frontend (Next.js)
    ↓ Loads ONNX Model
ONNX.js Runtime (Browser)
    ↓ Runs Inference
Display Results (Instant)
```

---

## 🚀 Implementation: Option 1 (Python API)

### **Step 1: Create Flask API Server**

Create `ml_ideation/ml_api.py`:

```python
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

# Load model at startup (once)
print("Loading ML model...")
model = ProbabilisticEISPredictor(n_grid_points=12, hidden_dim=512)
model.load_state_dict(torch.load('best_eis_predictor.pth', map_location='cpu'))
model.eval()

# Load dataset grids
dataset = EISDataset(
    'eis_training_data/combined_dataset_5gt.csv',
    masking_patterns=[[1,1,1,0,0]],
    samples_per_pattern=100
)
predictor = EISParameterCompleter(model, dataset.grids)

print("✓ ML model loaded and ready!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': True})


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
        ]
    }
    """
    try:
        data = request.json
        known_params = data.get('known_params', {})
        top_k = data.get('top_k', 10)

        # Validate input
        if not known_params:
            return jsonify({'error': 'known_params is required'}), 400

        # Run prediction
        results = predictor.predict_missing_parameters(
            known_params=known_params,
            top_k=top_k
        )

        # Format response
        predictions = []
        for pred in results['top_k_predictions']:
            pred_formatted = {
                'rank': pred['rank'],
                **{k: float(v) for k, v in pred.items() if k != 'rank'},
                'confidence': 'high' if pred['joint_probability'] > 0.01 else 'medium' if pred['joint_probability'] > 0.005 else 'low'
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

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_type': 'ProbabilisticEISPredictor',
        'grid_points': 12,
        'hidden_dim': 512,
        'parameters': ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'],
        'supported_patterns': [
            {'known': ['Rsh', 'Ra', 'Rb'], 'predict': ['Ca', 'Cb']},
            {'known': ['Rsh', 'Ra', 'Ca'], 'predict': ['Rb', 'Cb']},
            # Add more patterns as needed
        ]
    })


if __name__ == '__main__':
    # Run on port 5002 (5001 is already used by circuit_api.py)
    app.run(host='0.0.0.0', port=5002, debug=True)
```

Install dependencies:
```bash
pip install flask flask-cors
```

Start the API server:
```bash
cd ml_ideation
python ml_api.py
```

Server will run on: `http://localhost:5002`

---

### **Step 2: Create TypeScript API Client**

Create `app/lib/mlApiClient.ts`:

```typescript
/**
 * ML API Client for Circuit Parameter Prediction
 */

export interface KnownParameters {
  Rsh?: number;
  Ra?: number;
  Rb?: number;
  Ca?: number;
  Cb?: number;
}

export interface MLPrediction {
  rank: number;
  Ca?: number;
  Cb?: number;
  Rsh?: number;
  Ra?: number;
  Rb?: number;
  joint_probability: number;
  marginal_prob_1: number;
  marginal_prob_2: number;
  confidence: 'high' | 'medium' | 'low';
}

export interface UncertaintyMetrics {
  entropy: number;
  max_probability: number;
  top_5_mass: number;
  normalized_entropy: number;
}

export interface MLPredictionResponse {
  predicted_resnorm: number;
  missing_params: string[];
  predictions: MLPrediction[];
  uncertainty: Record<string, UncertaintyMetrics>;
}

export interface ModelInfo {
  model_type: string;
  grid_points: number;
  hidden_dim: number;
  parameters: string[];
  supported_patterns: Array<{
    known: string[];
    predict: string[];
  }>;
}

class MLApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:5002') {
    this.baseUrl = baseUrl;
  }

  /**
   * Check if ML API is available
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      return data.status === 'ok' && data.model_loaded === true;
    } catch (error) {
      console.error('ML API health check failed:', error);
      return false;
    }
  }

  /**
   * Predict missing circuit parameters
   */
  async predictParameters(
    knownParams: KnownParameters,
    topK: number = 10
  ): Promise<MLPredictionResponse> {
    const response = await fetch(`${this.baseUrl}/api/ml/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        known_params: knownParams,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'ML prediction failed');
    }

    return await response.json();
  }

  /**
   * Get model information
   */
  async getModelInfo(): Promise<ModelInfo> {
    const response = await fetch(`${this.baseUrl}/api/ml/model-info`);
    if (!response.ok) {
      throw new Error('Failed to fetch model info');
    }
    return await response.json();
  }
}

// Export singleton instance
export const mlApi = new MLApiClient();
```

---

### **Step 3: Create React Component for ML Predictions**

Create `app/components/circuit-simulator/ml/MLParameterPredictor.tsx`:

```typescript
'use client';

import React, { useState } from 'react';
import { mlApi, MLPrediction, KnownParameters } from '@/lib/mlApiClient';

interface MLParameterPredictorProps {
  currentParameters: {
    Rsh: number;
    Ra: number;
    Rb: number;
    Ca: number;
    Cb: number;
  };
  onApplyPrediction?: (params: Partial<KnownParameters>) => void;
}

export const MLParameterPredictor: React.FC<MLParameterPredictorProps> = ({
  currentParameters,
  onApplyPrediction,
}) => {
  const [predictions, setPredictions] = useState<MLPrediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictedResnorm, setPredictedResnorm] = useState<number | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      // Predict Ca and Cb given Rsh, Ra, Rb
      const knownParams: KnownParameters = {
        Rsh: currentParameters.Rsh,
        Ra: currentParameters.Ra,
        Rb: currentParameters.Rb,
      };

      const result = await mlApi.predictParameters(knownParams, 10);

      setPredictions(result.predictions);
      setPredictedResnorm(result.predicted_resnorm);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyPrediction = (prediction: MLPrediction) => {
    if (onApplyPrediction) {
      onApplyPrediction({
        Ca: prediction.Ca,
        Cb: prediction.Cb,
      });
    }
  };

  return (
    <div className="ml-parameter-predictor bg-gray-900 p-6 rounded-lg">
      <h2 className="text-xl font-bold text-white mb-4">
        🤖 ML Parameter Prediction
      </h2>

      <div className="mb-4">
        <p className="text-gray-300 text-sm mb-2">
          Given: Rsh={currentParameters.Rsh.toFixed(1)}Ω,
          Ra={currentParameters.Ra.toFixed(1)}Ω,
          Rb={currentParameters.Rb.toFixed(1)}Ω
        </p>
        <p className="text-gray-400 text-sm">
          Predict: Ca and Cb
        </p>
      </div>

      <button
        onClick={handlePredict}
        disabled={loading}
        className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg mb-4 disabled:opacity-50"
      >
        {loading ? 'Predicting...' : 'Predict Parameters'}
      </button>

      {error && (
        <div className="bg-red-900/30 border border-red-600 text-red-300 p-3 rounded mb-4">
          Error: {error}
        </div>
      )}

      {predictedResnorm !== null && (
        <div className="bg-blue-900/30 border border-blue-600 text-blue-300 p-3 rounded mb-4">
          Predicted Resnorm: {predictedResnorm.toFixed(2)}
        </div>
      )}

      {predictions.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-white font-semibold mb-2">Top 10 Predictions:</h3>
          {predictions.map((pred) => (
            <div
              key={pred.rank}
              className="bg-gray-800 p-3 rounded border border-gray-700 hover:border-blue-500 transition-colors"
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="text-white font-medium">
                    Rank #{pred.rank}
                  </div>
                  <div className="text-gray-300 text-sm mt-1">
                    Ca = {(pred.Ca! * 1e6).toFixed(2)} µF
                    ({pred.Ca!.toExponential(2)} F)
                  </div>
                  <div className="text-gray-300 text-sm">
                    Cb = {(pred.Cb! * 1e6).toFixed(2)} µF
                    ({pred.Cb!.toExponential(2)} F)
                  </div>
                  <div className="text-gray-400 text-xs mt-1">
                    Probability: {(pred.joint_probability * 100).toFixed(2)}%
                    <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                      pred.confidence === 'high' ? 'bg-green-900/50 text-green-300' :
                      pred.confidence === 'medium' ? 'bg-yellow-900/50 text-yellow-300' :
                      'bg-gray-700 text-gray-400'
                    }`}>
                      {pred.confidence}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => handleApplyPrediction(pred)}
                  className="ml-4 bg-green-600 hover:bg-green-700 text-white text-sm px-3 py-1 rounded"
                >
                  Apply
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
```

---

### **Step 4: Integrate into CircuitSimulator**

Update `app/components/CircuitSimulator.tsx`:

```typescript
import { MLParameterPredictor } from './circuit-simulator/ml/MLParameterPredictor';

// Inside your CircuitSimulator component, add:

const handleApplyMLPrediction = (params: Partial<KnownParameters>) => {
  if (params.Ca) {
    setCircuitParams(prev => ({ ...prev, Ca: params.Ca! }));
  }
  if (params.Cb) {
    setCircuitParams(prev => ({ ...prev, Cb: params.Cb! }));
  }
  // Trigger recomputation
  handleComputeClick();
};

// Add to your JSX (in a new tab or panel):
<MLParameterPredictor
  currentParameters={circuitParams}
  onApplyPrediction={handleApplyMLPrediction}
/>
```

---

## 🎯 Usage Workflow

### **1. Start Backend Servers:**

```bash
# Terminal 1: Circuit API (already running)
cd python-scripts
python circuit_api.py

# Terminal 2: ML API (NEW)
cd ml_ideation
python ml_api.py

# Terminal 3: Next.js Frontend
npm run dev
```

### **2. In Your Website:**

1. User sets Rsh, Ra, Rb values
2. Click "Predict Parameters" button
3. ML API returns top 10 predictions for Ca, Cb
4. User clicks "Apply" on desired prediction
5. Website updates circuit parameters
6. Run computation with new parameters

---

## 🚀 Quick Setup Script

Create `start_ml_services.sh`:

```bash
#!/bin/bash

echo "Starting ML-Enhanced EIS Platform..."

# Start Circuit API
cd python-scripts
python circuit_api.py &
CIRCUIT_PID=$!

# Start ML API
cd ../ml_ideation
python ml_api.py &
ML_PID=$!

# Start Next.js
cd ..
npm run dev &
NEXT_PID=$!

echo "Services started:"
echo "  Circuit API: http://localhost:5001 (PID: $CIRCUIT_PID)"
echo "  ML API: http://localhost:5002 (PID: $ML_PID)"
echo "  Next.js: http://localhost:3000 (PID: $NEXT_PID)"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $CIRCUIT_PID $ML_PID $NEXT_PID; exit" INT
wait
```

```bash
chmod +x start_ml_services.sh
./start_ml_services.sh
```

---

## 📊 Testing the Integration

### **Test ML API:**

```bash
# Health check
curl http://localhost:5002/health

# Predict parameters
curl -X POST http://localhost:5002/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "known_params": {"Rsh": 460, "Ra": 4820, "Rb": 2210},
    "top_k": 5
  }'
```

### **Test in Browser Console:**

```javascript
// Test ML API from browser
fetch('http://localhost:5002/api/ml/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    known_params: {Rsh: 460, Ra: 4820, Rb: 2210},
    top_k: 5
  })
})
.then(r => r.json())
.then(console.log);
```

---

## 🎨 UI Enhancement Ideas

1. **Add ML Tab** in your existing tabs (Visualizer, Math Details, NPZ Manager)
2. **Inline Suggestions** - Show ML predictions next to parameter sliders
3. **Confidence Indicators** - Color-code predictions by confidence
4. **History Tracking** - Save applied ML predictions
5. **Comparison View** - Compare ML prediction vs manual parameters

---

## ✅ Summary

**What You Have Now:**
- ✅ Trained ML model: `best_eis_predictor.pth`
- ✅ Python inference: Working

**What to Add:**
1. Create `ml_api.py` (Flask server)
2. Create `mlApiClient.ts` (TypeScript client)
3. Create `MLParameterPredictor.tsx` (React component)
4. Integrate into `CircuitSimulator.tsx`

**Result:**
Users can click "Predict Parameters" and get ML-powered suggestions! 🚀

Need help with any of these steps?

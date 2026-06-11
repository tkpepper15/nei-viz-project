#!/bin/bash

# Start Physics Transformer API
# Port: 5001

echo "Starting Physics Transformer API on port 5001..."

cd "$(dirname "$0")"

# Check if model exists
if [ ! -d "models/physics_informed" ]; then
    echo "Error: Model not found at models/physics_informed/"
    echo "Please train the model first using: python 03_train_physics.py"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "Warning: Virtual environment not found. Using system Python."
    python3 backend_api.py
else
    echo "Using virtual environment..."
    source ../.venv/bin/activate
    python backend_api.py
fi

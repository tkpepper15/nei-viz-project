#!/bin/bash

# Kill any existing backend server process
pkill -f "python app/components/compute.py" || true
echo "Cleaning up any existing backend processes..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
fi

# Install required dependencies if needed
echo "Checking dependencies..."
pip install fastapi uvicorn numpy scipy pydantic --quiet

# Start the backend server with retry logic
echo "Starting backend server..."
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  python app/components/compute.py
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Backend server stopped normally."
    break
  else
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
      echo "Backend server crashed. Retrying ($RETRY_COUNT/$MAX_RETRIES)..."
      sleep 2
    else
      echo "Backend server failed to start after $MAX_RETRIES attempts."
    fi
  fi
done

echo "Backend server stopped." 
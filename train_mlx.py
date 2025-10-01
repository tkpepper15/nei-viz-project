#!/usr/bin/env python3
"""
Convenience wrapper for MLX training pipeline
Runs from project root directory
"""

import sys
import os
from pathlib import Path

# Add ml_ideation to path
ml_ideation_path = Path(__file__).parent / 'ml_ideation'
sys.path.insert(0, str(ml_ideation_path))

# Change to ml_ideation directory for relative paths
os.chdir(ml_ideation_path)

# Import and run the main pipeline
from complete_pipeline_mlx import main

if __name__ == "__main__":
    main()

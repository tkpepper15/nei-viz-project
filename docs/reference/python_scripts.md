 # Install dependencies
  pip install -r scripts/requirements.txt

  # Run analysis (interactive)
  python scripts/circuit_analysis.py

  # Or batch mode with defaults
  python scripts/circuit_analysis.py --batch

  # Generate visualizations
  python scripts/create_visualizations.py

  # List available analyses
  python scripts/create_visualizations.py --list

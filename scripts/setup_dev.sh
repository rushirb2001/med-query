#!/bin/bash
# Development environment setup script

set -e

echo "Setting up med-query development environment..."

# Install dependencies
poetry install

# Verify hybrid-flow is accessible
python -c "import hybridflow; print(f'HybridFlow version: {hybridflow.__version__ if hasattr(hybridflow, \"__version__\") else \"installed\"}')"

echo "Setup complete!"

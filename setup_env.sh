#!/usr/bin/env bash
# Script to set up the environment for mlx-distributed-inference
set -euo pipefail

# Install system packages
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip git openmpi-bin libopenmpi-dev
else
    echo "Please install python3, git, and OpenMPI using your system package manager." >&2
fi

# Create Python virtual environment
VENV_DIR="mlx-env"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install Python dependencies
pip install --upgrade pip
pip install mlx mlx_lm huggingface-hub

# Optionally configure SSH keys for MPI
if [ ! -f "$HOME/.ssh/id_rsa" ]; then
    ssh-keygen -t rsa -N '' -f "$HOME/.ssh/id_rsa"
fi

# Copy hosts.json to standard location if needed
mkdir -p "$HOME/.mlx"
if [ -f hosts.json ] && [ ! -f "$HOME/.mlx/hosts.json" ]; then
    cp hosts.json "$HOME/.mlx/hosts.json"
fi

echo "Environment setup complete. Activate with: source $VENV_DIR/bin/activate"

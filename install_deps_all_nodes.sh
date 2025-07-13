#!/bin/bash

# Script to install all dependencies in mlx-env on all nodes

echo "=================================================="
echo "Installing Dependencies on All Nodes"
echo "=================================================="

# Python environment path
PYTHON_ENV="/Users/mm/mlx-project/mlx-env"
PYTHON_EXEC="$PYTHON_ENV/bin/python"
PIP_EXEC="$PYTHON_ENV/bin/pip"

# Check if mlx-env exists
if [ ! -d "$PYTHON_ENV" ]; then
    echo "Error: mlx-env not found at $PYTHON_ENV"
    exit 1
fi

# Create requirements file with all dependencies
cat > /tmp/mlx_requirements.txt << EOF
pyyaml
mlx
mlx-lm
numpy
tqdm
huggingface-hub
transformers
matplotlib
mpi4py
safetensors
sentencepiece
protobuf
EOF

echo "Installing dependencies locally..."
$PIP_EXEC install -r /tmp/mlx_requirements.txt

# Get list of remote nodes from hostfile
if [ -f "hostfile" ]; then
    NODES=$(grep -v '^#' hostfile | grep -v '^$' | awk '{print $1}' | grep -v '10.85.100.220')
    
    echo ""
    echo "Installing dependencies on remote nodes..."
    for node in $NODES; do
        echo ""
        echo "Installing on $node..."
        
        # Copy requirements file to remote node
        scp /tmp/mlx_requirements.txt $node:/tmp/
        
        # Install dependencies on remote node
        ssh $node "$PIP_EXEC install -r /tmp/mlx_requirements.txt"
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully installed dependencies on $node"
        else
            echo "❌ Failed to install dependencies on $node"
        fi
    done
fi

# Clean up
rm /tmp/mlx_requirements.txt

echo ""
echo "=================================================="
echo "✅ Dependency installation complete!"
echo "=================================================="

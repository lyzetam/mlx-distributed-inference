#!/bin/bash

# Setup script for distributed MLX training nodes
# This script installs all required dependencies on all nodes

echo "=================================================="
echo "MLX Distributed Training - Node Setup"
echo "=================================================="

# Check if hostfile exists
if [ ! -f "hostfile" ]; then
    echo "Error: hostfile not found!"
    echo "Please ensure hostfile is configured with your node IPs"
    exit 1
fi

# Count number of nodes
NUM_NODES=$(grep -v '^#' hostfile | grep -v '^$' | wc -l | tr -d ' ')
echo "Setting up $NUM_NODES nodes..."

# Display hosts
echo "Nodes to setup:"
grep -v '^#' hostfile | grep -v '^$' | awk '{print "  - " $1}'

# Detect the Python environment
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
    PIP_EXEC="$CONDA_PREFIX/bin/pip"
    echo "Using conda environment: $CONDA_PREFIX"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
    PIP_EXEC="$VIRTUAL_ENV/bin/pip"
    echo "Using virtual environment: $VIRTUAL_ENV"
else
    # Try to find mlx-env specifically
    if [ -d "$HOME/mlx-env" ]; then
        PYTHON_EXEC="$HOME/mlx-env/bin/python"
        PIP_EXEC="$HOME/mlx-env/bin/pip"
        echo "Using mlx-env at: $HOME/mlx-env"
    elif [ -d "/Users/mm/mlx-env" ]; then
        PYTHON_EXEC="/Users/mm/mlx-env/bin/python"
        PIP_EXEC="/Users/mm/mlx-env/bin/pip"
        echo "Using mlx-env at: /Users/mm/mlx-env"
    elif [ -d "/Users/mm/mlx-project/mlx-env" ]; then
        PYTHON_EXEC="/Users/mm/mlx-project/mlx-env/bin/python"
        PIP_EXEC="/Users/mm/mlx-project/mlx-env/bin/pip"
        echo "Using mlx-env at: /Users/mm/mlx-project/mlx-env"
    else
        PYTHON_EXEC="python3"
        PIP_EXEC="pip3"
        echo "Warning: No mlx-env found, using system python3"
    fi
fi

echo ""
echo "Step 1: Checking Python version on all nodes..."
echo "--------------------------------------------------"
mpirun --hostfile hostfile -np $NUM_NODES $PYTHON_EXEC --version

echo ""
echo "Step 2: Installing required Python packages..."
echo "--------------------------------------------------"
echo "Installing: PyYAML, MLX, and other dependencies"

# Create a temporary requirements file with all dependencies
cat > temp_requirements.txt << EOF
pyyaml
mlx
mlx-lm
numpy
tqdm
huggingface-hub
transformers
matplotlib
mpi4py
EOF

# Install packages on all nodes
echo "Running pip install on all nodes..."
mpirun --hostfile hostfile -np $NUM_NODES $PIP_EXEC install -r temp_requirements.txt

# Clean up
rm temp_requirements.txt

echo ""
echo "Step 3: Verifying installations..."
echo "--------------------------------------------------"
# Check if key packages are installed
echo "Checking PyYAML..."
mpirun --hostfile hostfile -np $NUM_NODES $PYTHON_EXEC -c "import yaml; print(f'PyYAML {yaml.__version__} installed')"

echo ""
echo "Checking MLX..."
mpirun --hostfile hostfile -np $NUM_NODES $PYTHON_EXEC -c "import mlx; print('MLX installed')"

echo ""
echo "Step 4: Synchronizing code to all nodes..."
echo "--------------------------------------------------"
CURRENT_DIR=$(pwd)
echo "Syncing from: $CURRENT_DIR"

# Get list of nodes from hostfile
NODES=$(grep -v '^#' hostfile | grep -v '^$' | awk '{print $1}')

# Sync to each node (skip if it's localhost)
for node in $NODES; do
    if [[ "$node" != "localhost" ]] && [[ "$node" != "127.0.0.1" ]]; do
        echo "Syncing to $node..."
        # Create directory if it doesn't exist
        ssh $node "mkdir -p $CURRENT_DIR"
        # Sync files
        rsync -av --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
              --exclude='logs' --exclude='adapters' --exclude='*.safetensors' \
              $CURRENT_DIR/ $node:$CURRENT_DIR/
    else
        echo "Skipping localhost"
    fi
done

echo ""
echo "Step 5: Final verification..."
echo "--------------------------------------------------"
# Test import of training module on all nodes
echo "Testing training module import..."
mpirun --hostfile hostfile -np $NUM_NODES -wdir $CURRENT_DIR \
    $PYTHON_EXEC -c "from utils.training_core import train_model; print('Training module OK')"

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "You can now run distributed training with:"
echo "  ./run_distributed_training.sh config/homeassistant_training_config.yaml"
echo ""
echo "To verify the setup:"
echo "  mpirun --hostfile hostfile -np $NUM_NODES $PYTHON_EXEC -c \"import yaml, mlx; print('All OK')\""

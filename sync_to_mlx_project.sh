#!/bin/bash

# Script to sync MLX distributed inference code to mlx-project directory on all nodes

echo "=================================================="
echo "Syncing MLX Code to All Nodes"
echo "=================================================="

# Source and destination paths
SOURCE_DIR="/Users/mm/Github/mlx-distributed-inference"
DEST_DIR="/Users/mm/mlx-project"

# Check if hostfile exists
if [ ! -f "hostfile" ]; then
    echo "Error: hostfile not found!"
    echo "Please ensure hostfile is configured with your node IPs"
    exit 1
fi

# Get list of nodes from hostfile
NODES=$(grep -v '^#' hostfile | grep -v '^$' | awk '{print $1}')
NUM_NODES=$(grep -v '^#' hostfile | grep -v '^$' | wc -l | tr -d ' ')

echo "Source directory: $SOURCE_DIR"
echo "Destination directory: $DEST_DIR"
echo "Number of nodes to sync: $NUM_NODES"
echo ""
echo "Nodes:"
for node in $NODES; do
    echo "  - $node"
done

echo ""
echo "Starting synchronization..."
echo "=================================================="

# First, sync to local mlx-project if needed
if [ -d "$SOURCE_DIR" ]; then
    echo "Syncing to local mlx-project directory..."
    mkdir -p "$DEST_DIR"
    rsync -av --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
          --exclude='logs' --exclude='adapters' --exclude='*.safetensors' \
          --exclude='mlx-env' --exclude='venv' --exclude='.env' \
          "$SOURCE_DIR/" "$DEST_DIR/"
    echo "✅ Local sync complete"
else
    echo "❌ Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

echo ""
echo "Syncing to remote nodes..."
echo "--------------------------------------------------"

# Sync to each remote node
for node in $NODES; do
    # Skip localhost entries
    if [[ "$node" == "localhost" ]] || [[ "$node" == "127.0.0.1" ]] || [[ "$node" == "10.85.100.220" ]]; then
        echo "Skipping local node: $node"
        continue
    fi
    
    echo ""
    echo "Syncing to $node..."
    
    # Create destination directory on remote node
    echo "  Creating directory structure..."
    ssh "$node" "mkdir -p $DEST_DIR" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "  ❌ Failed to connect to $node - skipping"
        continue
    fi
    
    # Sync files
    echo "  Copying files..."
    rsync -av --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
          --exclude='logs' --exclude='adapters' --exclude='*.safetensors' \
          --exclude='mlx-env' --exclude='venv' --exclude='.env' \
          "$SOURCE_DIR/" "$node:$DEST_DIR/"
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully synced to $node"
    else
        echo "  ❌ Failed to sync to $node"
    fi
done

echo ""
echo "=================================================="
echo "Sync Summary"
echo "=================================================="

# Verify the sync on all nodes
echo "Verifying sync on all nodes..."
for node in $NODES; do
    if [[ "$node" == "localhost" ]] || [[ "$node" == "127.0.0.1" ]]; then
        echo "$node: $(ls -la $DEST_DIR/train_homeassistant.py 2>/dev/null | wc -l) files found"
    else
        echo -n "$node: "
        ssh "$node" "ls $DEST_DIR/train_homeassistant.py 2>/dev/null && echo 'OK' || echo 'NOT FOUND'" 2>/dev/null || echo "UNREACHABLE"
    fi
done

echo ""
echo "=================================================="
echo "✅ Sync process complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Ensure mlx-env is available at /Users/mm/mlx-project/mlx-env on all nodes"
echo "2. Run distributed training with:"
echo "   cd $DEST_DIR"
echo "   ./run_distributed_training.sh config/homeassistant_training_config.yaml"

#!/bin/bash

echo "===================================================="
echo "Syncing MLX Distributed Inference to All Nodes"
echo "===================================================="

# Define the nodes
NODES=("10.85.100.221" "10.85.100.222")
SOURCE_DIR="/Users/mm/Github/mlx-distributed-inference"
TARGET_DIR="/Users/mm/Github/mlx-distributed-inference"

# Create target directory on remote nodes if it doesn't exist
for NODE in "${NODES[@]}"; do
    echo "Creating directory on $NODE..."
    ssh $NODE "mkdir -p $TARGET_DIR"
done

# Sync the entire repository to each node
for NODE in "${NODES[@]}"; do
    echo ""
    echo "Syncing to $NODE..."
    rsync -av --exclude 'mlx-env' --exclude '__pycache__' --exclude '.git' \
        --exclude '*.pyc' --exclude '.DS_Store' \
        $SOURCE_DIR/ $NODE:$TARGET_DIR/
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully synced to $NODE"
    else
        echo "❌ Failed to sync to $NODE"
    fi
done

echo ""
echo "===================================================="
echo "Sync Complete!"
echo "===================================================="
echo ""
echo "To verify, you can run:"
echo "ssh 10.85.100.221 'ls -la $TARGET_DIR'"
echo "ssh 10.85.100.222 'ls -la $TARGET_DIR'"

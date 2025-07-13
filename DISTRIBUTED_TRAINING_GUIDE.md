# Distributed Training Guide

## Current Status
✅ Distributed training is now **ENABLED** in the configuration files
✅ 3 nodes are configured in `hosts.json`
✅ Training script is ready for distributed execution
✅ All dependencies installed on all nodes in mlx-env
✅ Code synced to `/Users/mm/mlx-project` on all nodes

## How to Run Distributed Training

### Option 1: Using the Shell Script (Recommended)
```bash
# Navigate to the mlx-project directory
cd /Users/mm/mlx-project

# For general training with TinyLlama
./run_distributed_training.sh

# For Home Assistant training with Gemma 9B
./run_distributed_training.sh config/homeassistant_training_config.yaml
```

### Option 2: Direct MPI Command
```bash
# Run with 3 nodes
mpirun --hostfile hostfile -np 3 python3 train_homeassistant.py --distributed

# Or for the notebook
mpirun --hostfile hostfile -np 3 jupyter notebook distributed_training.ipynb
```

### Option 3: Using Python Script with Distributed Flag
```bash
# This will use the distributed setting from the config file (now enabled)
python3 train_homeassistant.py
```

## Configuration Changes Made

1. **config/training_config.yaml**
   - `distributed.enabled: true`

2. **config/homeassistant_training_config.yaml**
   - `distributed.enabled: true`

## Verify Distributed Setup

Before running training, verify your MPI setup:
```bash
# Check if all nodes are accessible
mpirun --hostfile hostfile -np 3 hostname

# Check Python on all nodes
mpirun --hostfile hostfile -np 3 python3 --version
```

## Hostfile Format

The MPI hostfile should be in the following format:
```
10.85.100.220 slots=1
10.85.100.221 slots=1
10.85.100.222 slots=1
```

Note: The script will automatically convert `hosts.json` to the proper MPI format if needed.

## Important Notes

- The training was previously running on a single node
- Now it will automatically use all 3 nodes configured in `hosts.json`
- Each node will process a portion of the batch
- Gradients will be synchronized across nodes
- Training should be ~3x faster with 3 nodes

## Troubleshooting

If you encounter issues:

1. **SSH Access**: Ensure passwordless SSH is set up between nodes
2. **Code Sync**: Make sure all nodes have the same code version
   - The code must be in the same path on all nodes
   - Use rsync or git to synchronize: `rsync -av /Users/mm/Github/mlx-distributed-inference/ user@node:/path/to/mlx-distributed-inference/`
3. **Dependencies**: Install MLX and other dependencies on all nodes
   - Run on each node: `pip install -r requirements.txt`
4. **Network**: Check network connectivity between nodes (preferably Thunderbolt)
5. **Working Directory**: Ensure the working directory exists on all nodes

### Quick Setup for Remote Nodes

```bash
# 1. Sync code to mlx-project directory on all nodes
./sync_to_mlx_project.sh

# 2. Install all dependencies in mlx-env on all nodes
./install_deps_all_nodes.sh

# 3. Run distributed training
cd /Users/mm/mlx-project
./run_distributed_training.sh config/homeassistant_training_config.yaml
```

### Complete Setup Process

1. **Enable distributed training in config files**
   - Set `distributed.enabled: true` in YAML configs

2. **Create proper MPI hostfile**
   ```
   10.85.100.220 slots=1
   10.85.100.221 slots=1
   10.85.100.222 slots=1
   ```

3. **Sync code to all nodes**
   - Use `sync_to_mlx_project.sh` to copy files to `/Users/mm/mlx-project`

4. **Install dependencies**
   - Use `install_deps_all_nodes.sh` to install all Python packages in mlx-env

5. **Run training**
   - Execute from `/Users/mm/mlx-project` directory
   - Script automatically detects and uses mlx-env

## Monitor Training

The training will show progress from all nodes:
- `[Rank 0/3]` - Main node
- `[Rank 1/3]` - Second node  
- `[Rank 2/3]` - Third node

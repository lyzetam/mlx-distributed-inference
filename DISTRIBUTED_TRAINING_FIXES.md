# Distributed Training Fixes Summary

## Issues Encountered and Solutions

### 1. Configuration Issues
**Problem**: Distributed training was disabled in config files
**Solution**: Set `distributed.enabled: true` in both:
- `config/training_config.yaml`
- `config/homeassistant_training_config.yaml`

### 2. MPI Hostfile Format
**Problem**: Hostfile had incorrect format for MPI
**Solution**: Created proper MPI hostfile with format:
```
10.85.100.220 slots=1
10.85.100.221 slots=1
10.85.100.222 slots=1
```

### 3. Environment Path Issues
**Problem**: Scripts couldn't find mlx-env
**Solution**: Modified `run_distributed_training.sh` to:
- Detect mlx-env at `/Users/mm/mlx-project/mlx-env`
- Use the correct Python executable from mlx-env

### 4. Code Synchronization
**Problem**: Code needed to be in same location on all nodes
**Solution**: 
- Created `sync_to_mlx_project.sh` to sync code from GitHub directory to `/Users/mm/mlx-project`
- Synced to all nodes with proper directory structure

### 5. Missing Dependencies
**Problem**: matplotlib was missing on all nodes
**Solution**: 
- Installed matplotlib in mlx-env: `pip install matplotlib`
- Created `install_deps_all_nodes.sh` to install all dependencies on all nodes

### 6. Interactive Input in Distributed Mode
**Problem**: Script waited for user input ("Press Enter") which fails in MPI
**Solution**: Modified `train_homeassistant.py` to skip input prompt when:
```python
if not (args.distributed or config['distributed']['enabled']):
    input("\nPress Enter to start training...")
```

### 7. Missing wandb Attribute
**Problem**: Training expected `wandb` attribute in args
**Solution**: Added `"wandb": False` to defaults in `training_core.py`

### 8. Sentencepiece Library Issues
**Problem**: sentencepiece Python package couldn't link to system library
**Solution**:
1. Installed system dependencies: `brew install sentencepiece protobuf cmake`
2. Reinstalled Python package with proper linking

## Final Working Setup

1. All code in `/Users/mm/mlx-project` on all nodes
2. Using mlx-env virtual environment
3. All dependencies installed (including matplotlib, mpi4py, sentencepiece)
4. Distributed training enabled in configs
5. Proper MPI hostfile format
6. No interactive prompts in distributed mode
7. wandb disabled by default

## Running Distributed Training

```bash
cd /Users/mm/mlx-project
./run_distributed_training.sh config/homeassistant_training_config.yaml
```

This will:
- Use MPI to launch training on all 3 nodes
- Load Gemma 9B model on each node
- Distribute training data across nodes
- Synchronize gradients during training
- Provide ~3x speedup over single-node training

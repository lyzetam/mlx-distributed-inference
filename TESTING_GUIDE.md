# MLX Distributed Inference Testing Guide

## Quick Start

### 1. Local Inference (Single Node)
```bash
# Basic test with default model (TinyLlama)
python distributed_inference_mlx.py --max-tokens 50 "What is machine learning?"

# Test with different models
python distributed_inference_mlx.py --model-name mlx-community/Llama-3.2-1B-Instruct-4bit --max-tokens 30 "Hello"

# Adjust temperature for creativity
python distributed_inference_mlx.py --temperature 0.9 --max-tokens 100 "Write a story"
```

### 2. Run Test Script
```bash
# Make executable and run
chmod +x test_inference.sh
./test_inference.sh
```

### 3. Distributed Inference (Multi-Node)
```bash
# First, sync to all nodes
./sync_to_nodes.sh

# Then run distributed inference using the helper script
./run_distributed_inference.sh "Your prompt" 50

# Or run directly with mlx.launch
mlx.launch --hostfile hosts.json --backend mpi python distributed_inference_mlx.py --max-tokens 50 "Your prompt"

# Or specify hosts directly
mlx.launch --hosts "10.85.100.220,10.85.100.221,10.85.100.222" --backend mpi python distributed_inference_mlx.py --max-tokens 50 "Your prompt"
```

### Using the Helper Script
```bash
# Basic usage
./run_distributed_inference.sh "What is AI?"

# With custom token limit
./run_distributed_inference.sh "Explain machine learning" 100

# With specific model
./run_distributed_inference.sh "Hello" 50 mlx-community/Llama-3.2-1B-Instruct-4bit
```

## Available Models

Some popular models that work well:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (default, fast)
- `mlx-community/Llama-3.2-1B-Instruct-4bit`
- `mlx-community/Phi-3.5-mini-instruct-4bit`
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

## Command Line Options

- `--model-name`: Specify the model to use
- `--max-tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Control randomness (0.0-1.0, default: 0.7)
- `--top-p`: Nucleus sampling parameter (default: 0.9)
- `--seed`: Random seed for reproducibility

## Troubleshooting

### If distributed inference hangs:
1. Check network connectivity: `ping 10.85.100.221`
2. Verify SSH access: `ssh 10.85.100.221 hostname`
3. Check if MPI is installed on all nodes
4. Try with fewer nodes first

### If model loading fails:
1. The model might need to be downloaded first
2. Check available disk space
3. Try a smaller model like TinyLlama

### Performance Tips:
1. Use 4-bit quantized models for better memory efficiency
2. Start with smaller batch sizes
3. Monitor memory usage with Activity Monitor

## Example Outputs

Local inference typically shows:
- Model loading time
- Generated text
- Performance metrics (tokens/sec)
- Memory usage

Distributed inference will show similar metrics but distributed across nodes.

#!/bin/bash

echo "===================================================="
echo "MLX Distributed Inference"
echo "===================================================="

# Check if prompt is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Your prompt here\" [max_tokens] [model_name]"
    echo "Example: $0 \"What is AI?\" 50"
    echo "Example: $0 \"Explain quantum computing\" 100 mlx-community/Llama-3.2-1B-Instruct-4bit"
    exit 1
fi

PROMPT="$1"
MAX_TOKENS="${2:-50}"  # Default to 50 tokens
MODEL_NAME="${3:-}"    # Optional model name

# Build the command
CMD="mlx.launch --hostfile hosts.json --backend mpi python distributed_inference_mlx.py --max-tokens $MAX_TOKENS"

# Add model name if provided
if [ -n "$MODEL_NAME" ]; then
    CMD="$CMD --model-name $MODEL_NAME"
fi

# Add the prompt
CMD="$CMD \"$PROMPT\""

echo "Running distributed inference across nodes:"
echo "  - 10.85.100.220"
echo "  - 10.85.100.221" 
echo "  - 10.85.100.222"
echo ""
echo "Model: ${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default)}"
echo "Max tokens: $MAX_TOKENS"
echo "Prompt: $PROMPT"
echo "===================================================="

# Execute the command
eval $CMD

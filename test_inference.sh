#!/bin/bash

echo "Testing MLX Distributed Inference Setup"
echo "======================================="

# Activate the virtual environment
source mlx-env/bin/activate

echo "1. Testing local inference..."
python distributed_inference_mlx.py --max-tokens 30 "What is artificial intelligence?"

echo -e "\n2. Testing with custom model..."
python distributed_inference_mlx.py --model-name mlx-community/gemma-2-2b-it-4bit --max-tokens 20 "Hello, how are you?"

echo -e "\nInference tests completed!"

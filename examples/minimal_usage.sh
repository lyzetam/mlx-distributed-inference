#!/bin/bash
# Example: Minimal command line usage with configuration file

echo "=== MLX Distributed Inference - Minimal Usage Examples ==="
echo ""
echo "With config.yaml, your commands become much simpler:"
echo ""

echo "1. Single node inference (uses all defaults from config.yaml):"
echo "   python3 distributed_inference_mlx.py \"What is the meaning of life?\""
echo ""

echo "2. Distributed inference with hosts:"
echo "   mlx.launch --hosts 10.85.100.220,10.85.100.221,10.85.100.222 \\"
echo "     python3 distributed_inference_mlx.py \"What is the meaning of life?\""
echo ""

echo "3. Distributed inference with hostfile:"
echo "   mlx.launch --hostfile hosts.json --backend ring \\"
echo "     python3 distributed_inference_mlx.py \"What is the meaning of life?\""
echo ""

echo "Compare to the old way without config.yaml:"
echo "   mlx.launch --hosts 10.85.100.220,10.85.100.221,10.85.100.222 \\"
echo "     python3 distributed_inference_mlx.py \\"
echo "       --max-tokens 128 \\"
echo "       --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\"
echo "       --log-level INFO \\"
echo "       \"What is the meaning of life?\""
echo ""

echo "The config.yaml file handles all the default settings!"

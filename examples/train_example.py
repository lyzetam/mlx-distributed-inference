#!/usr/bin/env python3
"""
Example script for distributed training with MLX.

Usage:
    # Single node training
    python examples/train_example.py
    
    # Distributed training
    mpirun --hostfile hosts.json -np 2 python examples/train_example.py --distributed
"""

import argparse
import json
import time
from pathlib import Path

from utils.model_cache import ModelCacheManager
from utils.logging import setup_logging
from utils.data_preparation import prepare_dataset, create_sample_dataset
from utils.training_core import train_model
from utils.training_callbacks import DistributedTrainingCallback


def main():
    parser = argparse.ArgumentParser(description="MLX Training Example")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model name or path")
    parser.add_argument("--data", type=str, default="data/sample_dataset.jsonl",
                        help="Path to dataset")
    parser.add_argument("--output-dir", type=str, default="adapters",
                        help="Output directory for adapters")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create a sample dataset")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    print("=" * 60)
    print("MLX Training Example")
    print("=" * 60)
    
    # Create sample dataset if requested
    if args.create_sample or not Path(args.data).exists():
        print(f"Creating sample dataset at {args.data}...")
        Path(args.data).parent.mkdir(exist_ok=True)
        create_sample_dataset(args.data, num_examples=200)
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset_info = prepare_dataset(
        file_path=args.data,
        output_dir="data",
        val_ratio=0.2,
        format_type="chat"
    )
    
    print(f"Training examples: {dataset_info['train_size']}")
    print(f"Validation examples: {dataset_info['valid_size']}")
    
    # Training configuration
    training_config = {
        "fine_tune_type": "lora",
        "num_layers": 8,
        "lora_parameters": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "scale": 10.0
        },
        "batch_size": args.batch_size,
        "iters": args.iterations,
        "learning_rate": args.learning_rate,
        "val_batches": 25,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "save_every": 100,
        "adapter_path": args.output_dir,
        "max_seq_length": 2048
    }
    
    # Start training
    print(f"\nStarting {'distributed' if args.distributed else 'single-node'} training...")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run training
    results = train_model(
        model_name=args.model,
        data_path="data",
        training_config=training_config,
        distributed=args.distributed
    )
    
    training_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Training completed in {training_time:.2f} seconds")
    
    if results.get('success'):
        print("✅ Training successful!")
        print(f"Adapter saved to: {results.get('adapter_path')}")
        if 'final_train_loss' in results:
            print(f"Final training loss: {results['final_train_loss']:.4f}")
        if 'final_val_loss' in results:
            print(f"Final validation loss: {results['final_val_loss']:.4f}")
        
        # Save results
        results_file = Path("training_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "model": args.model,
                "training_time": training_time,
                "results": results,
                "args": vars(args)
            }, f, indent=2)
        print(f"\nResults saved to {results_file}")
        
        # Print inference command
        print("\nTo use the trained adapter:")
        print(f"mlx_lm.generate --model {args.model} --adapter-path {results.get('adapter_path')} --prompt 'Your prompt here'")
    else:
        print(f"❌ Training failed: {results.get('error')}")


if __name__ == "__main__":
    main()

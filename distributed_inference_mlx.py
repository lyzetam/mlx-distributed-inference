#!/usr/bin/env python3
"""
Adapted distributed inference script for MLX using pipeline parallelism.

Run with either:
1. mlx.launch --hostfile hosts.json --backend mpi distributed_inference_mlx.py "Your prompt here"
2. mpirun --hostfile hostfile -np 3 python distributed_inference_mlx.py "Your prompt here"
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.utils import load_model, load_tokenizer


def download(repo: str, allow_patterns: list[str]) -> Path:
    """Download model files from HuggingFace hub."""
    return Path(
        snapshot_download(
            repo,
            allow_patterns=allow_patterns,
        )
    )


def shard_and_load(model_name: str):
    """Load and shard model across distributed hosts."""
    # Get model path with everything but weight safetensors
    model_path = download(
        model_name,
        allow_patterns=["*.json", "*.py", "tokenizer.model", "*.tiktoken", "*.txt"],
    )
    
    # Initialize distributed group
    # Let MLX auto-detect the backend based on how it was launched
    group = mx.distributed.init()
    rank = group.rank()
    size = group.size()
    
    print(f"[Rank {rank}/{size}] Loading model {model_name}...")
    start_time = time.time()
    
    # For smaller models that don't need sharding, we can load normally
    # Check if model has safetensors index (indicates it's large enough to shard)
    index_path = model_path / "model.safetensors.index.json"
    
    if index_path.exists() and size > 1:
        # Large model - use pipeline parallelism
        print(f"[Rank {rank}] Using pipeline parallelism for large model")
        
        # Lazy load to figure out which weights we need
        model, _ = load_model(model_path, lazy=True, strict=False)
        # Shard the layers for this rank
        if hasattr(model, "pipeline"):
            model.pipeline(group)
        elif hasattr(model, "model") and hasattr(model.model, "pipeline"):
            model.model.pipeline(group)
        
        # Figure out which files we need for the local shard
        with open(index_path, "r") as fid:
            weight_index = json.load(fid)["weight_map"]
        
        local_files = set()
        for k, _ in tree_flatten(model.parameters()):
            if k in weight_index:
                local_files.add(weight_index[k])
        
        # Download weights for local shard
        download(model_name, allow_patterns=local_files)
        
        # Load and shard the model
        tokenizer = load_tokenizer(model_path)
        model, _ = load_model(model_path, lazy=True, strict=False)
        if hasattr(model, "pipeline"):
            model.pipeline(group)
        elif hasattr(model, "model") and hasattr(model.model, "pipeline"):
            model.model.pipeline(group)
        mx.eval(model.parameters())
    else:
        # Small model - each rank loads the full model
        print(f"[Rank {rank}] Loading full model (no sharding)")
        
        # Download all model files
        download(model_name, allow_patterns=["*.safetensors", "*.bin"])
        
        # Load model normally
        tokenizer = load_tokenizer(model_path)
        model, _ = load_model(model_path)
        mx.eval(model.parameters())
    
    load_time = time.time() - start_time
    print(f"[Rank {rank}] Model loaded in {load_time:.2f}s")
    
    # Synchronize processes before generation
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    
    return model, tokenizer, group


def main():
    parser = argparse.ArgumentParser(
        description="Distributed prompt runner via mlx.launch"
    )
    parser.add_argument(
        'prompt', nargs='+', 
        help="The prompt text to send to the model"
    )
    parser.add_argument(
        '--max-tokens', type=int, default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        '--model-name', type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model identifier to load"
    )
    parser.add_argument(
        '--chat-template', type=str,
        help=(
            'Optional chat template string or path to a file containing the '
            'template. If provided, it will override any template shipped with '
            'the tokenizer.'
        ),
    )
    # parser.add_argument(
    #     '--temperature', type=float, default=0.7,
    #     help="Sampling temperature"
    # )
    args = parser.parse_args()

    # Combine words into full prompt
    prompt = ' '.join(args.prompt)

    chat_template = None
    if args.chat_template:
        template_path = Path(args.chat_template)
        if template_path.exists():
            chat_template = template_path.read_text()
        else:
            chat_template = args.chat_template
    
    # Load and shard model
    model, tokenizer, group = shard_and_load(args.model_name)
    if chat_template:
        tokenizer.chat_template = chat_template
    rank = group.rank()
    size = group.size()
    
    # Only rank 0 prints output (to avoid duplicates)
    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)
    
    # Format prompt for chat models
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    # Generate response
    rprint(f"\nProcessing prompt: '{prompt}'")
    rprint("=" * 60)
    
    start_gen = time.time()
    
    # Use stream_generate for better performance with distributed setup
    response = None
    for response in stream_generate(
        model, 
        tokenizer, 
        formatted_prompt,
        max_tokens=args.max_tokens
        # temperature=args.temperature
    ):
        rprint(response.text, end="", flush=True)
    
    gen_time = time.time() - start_gen
    
    # Print statistics
    if response:
        rprint("\n" + "=" * 60)
        rprint(f"\nGeneration time: {gen_time:.2f}s")
        rprint(
            f"Prompt: {response.prompt_tokens} tokens, "
            f"{response.prompt_tps:.3f} tokens-per-sec"
        )
        rprint(
            f"Generation: {response.generation_tokens} tokens, "
            f"{response.generation_tps:.3f} tokens-per-sec"
        )
        rprint(f"Peak memory: {response.peak_memory:.3f} GB")
    
    rprint(f"\n[Distributed across {size} hosts] Done!")


if __name__ == "__main__":
    main()
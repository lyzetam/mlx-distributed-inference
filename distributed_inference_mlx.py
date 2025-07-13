#!/usr/bin/env python3
"""
Adapted distributed inference script for MLX using pipeline parallelism.

Run with either:
1. mlx.launch --hostfile hosts.json --backend mpi distributed_inference_mlx.py "Your prompt here"
2. mpirun --hostfile hostfile -np 3 python distributed_inference_mlx.py "Your prompt here"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.utils import load_model, load_tokenizer

from utils.errors import (
    ModelDownloadError, ModelLoadError, DistributedSetupError, InferenceError
)
from utils.logging import setup_logging, get_logger
from utils.model_cache import ModelCacheManager

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def shard_and_load(
    model_name: str,
    cache_manager: ModelCacheManager,
    force_download: bool = False
) -> Tuple:
    """
    Load and shard model across distributed hosts.
    
    Args:
        model_name: Model identifier or local path
        cache_manager: Model cache manager instance
        force_download: Force re-download even if cached
        
    Returns:
        Tuple of (model, tokenizer, group)
        
    Raises:
        DistributedSetupError: If distributed setup fails
        ModelLoadError: If model loading fails
    """
    logger = get_logger()
    
    try:
        # Initialize distributed group
        group = mx.distributed.init()
        rank = group.rank()
        size = group.size()
    except Exception as e:
        raise DistributedSetupError(f"Failed to initialize distributed group: {e}")
    
    # Update logger with rank information
    logger = setup_logging(rank=rank, size=size)
    
    logger.info(f"Loading model {model_name}...")
    start_time = time.time()
    
    try:
        # Get model path (download if necessary)
        # First download metadata files
        model_path = cache_manager.download_model_files(
            model_name,
            file_patterns=["*.json", "*.py", "tokenizer.model", "*.tiktoken", "*.txt"],
            force_download=force_download
        )
        
        # For small models, also download the weights immediately
        # Check if model has safetensors index (indicates it's large enough to shard)
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.exists():
            # Small model - download all weights now
            model_path = cache_manager.download_model_files(
                model_name,
                file_patterns=["*.safetensors", "*.bin"],
                force_download=False  # Don't force since we just downloaded metadata
            )
        
        # Check if model has safetensors index (indicates it's large enough to shard)
        index_path = model_path / "model.safetensors.index.json"
        
        if index_path.exists() and size > 1:
            # Large model - use pipeline parallelism
            logger.info("Using pipeline parallelism for large model")
            
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
            if local_files:
                logger.debug(f"Downloading {len(local_files)} weight files for local shard")
                cache_manager.download_model_files(
                    model_name,
                    file_patterns=list(local_files),
                    force_download=force_download
                )
            
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
            logger.info("Loading full model (no sharding)")
            
            # Download all model files
            model_path = cache_manager.download_model_files(
                model_name,
                file_patterns=["*.safetensors", "*.bin"],
                force_download=force_download
            )
            
            # Load model normally
            tokenizer = load_tokenizer(model_path)
            model, _ = load_model(model_path)
            mx.eval(model.parameters())
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        # Synchronize processes before generation
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
        
        return model, tokenizer, group
        
    except ModelDownloadError:
        raise
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}")


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    rank: int = 0
):
    """
    Generate response from the model.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        rank: Process rank (for output control)
        
    Returns:
        Generation statistics
        
    Raises:
        InferenceError: If generation fails
    """
    logger = get_logger()
    
    # Only rank 0 prints output
    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)
    
    try:
        # Format prompt for chat models
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
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
        
        # Use stream_generate for better performance
        response = None
        for response in stream_generate(
            model, 
            tokenizer, 
            formatted_prompt,
            max_tokens=max_tokens
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
            
            return {
                "generation_time": gen_time,
                "prompt_tokens": response.prompt_tokens,
                "generation_tokens": response.generation_tokens,
                "peak_memory": response.peak_memory
            }
        
    except Exception as e:
        raise InferenceError(f"Failed to generate response: {e}")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in current directory.
        
    Returns:
        Dictionary with configuration values
    """
    if not YAML_AVAILABLE:
        return {}
    
    if config_path is None:
        # Look for default config files
        for default_path in ["config.yaml", "config.yml", ".mlx-config.yaml"]:
            if Path(default_path).exists():
                config_path = default_path
                break
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                return config
        except Exception as e:
            logger = get_logger()
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return {}


def main():
    # First, create a parser just for the config file argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, help='Path to configuration file')
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load configuration
    config = load_config(pre_args.config)
    
    # Extract default values from config (ensure they're always dicts)
    model_config = config.get('model') or {}
    gen_config = config.get('generation') or {}
    cache_config = config.get('cache') or {}
    log_config = config.get('logging') or {}
    
    parser = argparse.ArgumentParser(
        description="Distributed prompt runner via mlx.launch"
    )
    
    # Prompt arguments
    parser.add_argument(
        'prompt', nargs='*', 
        help="The prompt text to send to the model"
    )
    
    # Configuration file
    parser.add_argument(
        '--config', type=str,
        help='Path to configuration file (YAML format)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name', type=str,
        default=model_config.get('name', "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        help="Model identifier to load"
    )
    parser.add_argument(
        '--model-path', type=str,
        default=model_config.get('path'),
        help="Local path to model (overrides model-name)"
    )
    
    # Cache arguments
    parser.add_argument(
        '--cache-dir', type=str,
        default=cache_config.get('dir'),
        help="Directory to cache downloaded models"
    )
    parser.add_argument(
        '--force-download', action='store_true',
        default=cache_config.get('force_download', False),
        help="Force re-download even if model is cached"
    )
    parser.add_argument(
        '--clear-cache', action='store_true',
        default=cache_config.get('clear_cache', False),
        help="Clear model cache before running"
    )
    parser.add_argument(
        '--list-cached', action='store_true',
        help="List cached models and exit"
    )
    
    # Generation arguments
    parser.add_argument(
        '--max-tokens', type=int, 
        default=gen_config.get('max_tokens', 100),
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        '--chat-template', type=str,
        default=gen_config.get('chat_template'),
        help=(
            'Optional chat template string or path to a file containing the '
            'template. If provided, it will override any template shipped with '
            'the tokenizer.'
        ),
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level', type=str, 
        default=log_config.get('level', 'INFO'),
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Logging level"
    )
    parser.add_argument(
        '--log-file', type=str,
        default=log_config.get('file'),
        help="Log file path (will append rank for multi-process)"
    )
    
    args = parser.parse_args()
    
    # Setup initial logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    # Initialize cache manager
    cache_dir = None
    if args.cache_dir:
        cache_dir = Path(os.path.expanduser(args.cache_dir))
    cache_manager = ModelCacheManager(cache_dir)
    
    # Handle cache operations
    if args.list_cached:
        cached_models = cache_manager.list_cached_models()
        if cached_models:
            print("Cached models:")
            sizes = cache_manager.get_cache_size()
            for model in cached_models:
                size = sizes.get(model, 0)
                print(f"  - {model} ({size:.2f} GB)")
        else:
            print("No cached models found")
        return
    
    if args.clear_cache:
        logger.info("Clearing model cache...")
        cache_manager.clear_cache()
    
    # Check if we need a prompt
    if not args.prompt:
        parser.error("Prompt is required unless using --list-cached")
    
    # Determine model to use
    model_name = args.model_path if args.model_path else args.model_name
    
    # Combine prompt words
    prompt = ' '.join(args.prompt)
    
    # Load chat template if provided
    chat_template = None
    if args.chat_template:
        template_path = Path(args.chat_template)
        if template_path.exists():
            chat_template = template_path.read_text()
            logger.debug(f"Loaded chat template from {template_path}")
        else:
            chat_template = args.chat_template
    
    try:
        # Load and shard model
        model, tokenizer, group = shard_and_load(
            model_name,
            cache_manager,
            force_download=args.force_download
        )
        
        if chat_template:
            tokenizer.chat_template = chat_template
        
        rank = group.rank()
        size = group.size()
        
        # Generate response
        stats = generate_response(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            rank=rank
        )
        
        if rank == 0:
            print(f"\n[Distributed across {size} hosts] Done!")
            
    except (ModelDownloadError, ModelLoadError, DistributedSetupError, InferenceError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

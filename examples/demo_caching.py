#!/usr/bin/env python3
"""
Demo script showing the new caching and logging features.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_cache import ModelCacheManager
from utils.logging import setup_logging

def main():
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Create cache manager
    cache_manager = ModelCacheManager()
    
    logger.info("Model Cache Demo")
    logger.info("=" * 50)
    
    # List cached models
    cached_models = cache_manager.list_cached_models()
    if cached_models:
        logger.info(f"Found {len(cached_models)} cached models:")
        sizes = cache_manager.get_cache_size()
        for model in cached_models:
            size = sizes.get(model, 0)
            logger.info(f"  - {model} ({size:.2f} GB)")
    else:
        logger.info("No cached models found")
    
    # Check if a specific model is cached
    test_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if cache_manager.is_cached(test_model):
        logger.info(f"\n{test_model} is already cached")
    else:
        logger.info(f"\n{test_model} is not cached")
    
    logger.info("\nCache directory: " + str(cache_manager.cache_dir))
    
    # Show how to use different log levels
    logger.debug("This is a debug message (won't show with INFO level)")
    logger.warning("This is a warning message")
    logger.error("This is an error message (not a real error, just a demo)")

if __name__ == "__main__":
    main()

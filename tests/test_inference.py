import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import distributed_inference_mlx as dim
from utils.errors import ModelDownloadError, ModelLoadError, DistributedSetupError
from utils.model_cache import ModelCacheManager


class TestModelCacheManager:
    """Test the ModelCacheManager class."""
    
    def test_cache_manager_init(self, tmp_path):
        """Test cache manager initialization."""
        cache_manager = ModelCacheManager(tmp_path)
        assert cache_manager.cache_dir == tmp_path
        assert cache_manager.metadata_file == tmp_path / "cache_metadata.json"
        assert cache_manager.metadata == {}
    
    def test_is_cached_returns_false_for_missing_model(self, tmp_path):
        """Test is_cached returns False for non-existent model."""
        cache_manager = ModelCacheManager(tmp_path)
        assert not cache_manager.is_cached("non-existent-model")
    
    def test_is_cached_returns_true_for_cached_model(self, tmp_path):
        """Test is_cached returns True for cached model."""
        cache_manager = ModelCacheManager(tmp_path)
        
        # Create fake cached model
        model_path = tmp_path / "test-model"
        model_path.mkdir()
        (model_path / "config.json").write_text("{}")
        (model_path / "tokenizer.json").write_text("{}")
        # Add model weights file
        (model_path / "model.safetensors").write_text("fake weights")
        
        # Manually set cache path
        cache_manager._get_model_cache_path = lambda x: model_path
        
        assert cache_manager.is_cached("test-model")
    
    def test_get_model_path_with_local_path(self, tmp_path):
        """Test get_model_path with local directory."""
        cache_manager = ModelCacheManager(tmp_path)
        
        # Create local model directory
        local_model = tmp_path / "local_model"
        local_model.mkdir()
        
        result = cache_manager.get_model_path(str(local_model))
        assert result == local_model
    
    def test_clear_cache_specific_model(self, tmp_path):
        """Test clearing cache for specific model."""
        cache_manager = ModelCacheManager(tmp_path)
        
        # Create fake cached models
        model1_path = tmp_path / "model1"
        model2_path = tmp_path / "model2"
        model1_path.mkdir()
        model2_path.mkdir()
        
        cache_manager._get_model_cache_path = lambda x: tmp_path / x
        cache_manager.metadata = {"model1": {}, "model2": {}}
        
        cache_manager.clear_cache("model1")
        
        assert not model1_path.exists()
        assert model2_path.exists()
        assert "model1" not in cache_manager.metadata
        assert "model2" in cache_manager.metadata


class DummyGroup:
    """Mock distributed group for testing."""
    
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def test_shard_and_load_initializes_group(monkeypatch, tmp_path):
    """Test shard_and_load initializes distributed group."""
    # Create cache manager
    cache_manager = ModelCacheManager(tmp_path)
    
    # Mock distributed init
    dummy_group = DummyGroup()
    monkeypatch.setattr(dim.mx.distributed, "init", lambda: dummy_group)
    monkeypatch.setattr(dim.mx.distributed, "all_sum", lambda arr, stream=None: arr)
    monkeypatch.setattr(dim.mx, "array", lambda x: x)
    monkeypatch.setattr(dim.mx, "cpu", None)
    monkeypatch.setattr(dim.mx, "eval", lambda *args, **kwargs: None)
    
    # Mock model loading
    class DummyModel:
        def parameters(self):
            return []
    
    monkeypatch.setattr(dim, "load_model", lambda path, lazy=False, strict=True: (DummyModel(), None))
    monkeypatch.setattr(dim, "load_tokenizer", lambda path: Mock())
    
    # Mock cache manager methods
    monkeypatch.setattr(cache_manager, "download_model_files", lambda *args, **kwargs: tmp_path)
    
    model, tokenizer, group = dim.shard_and_load("dummy-model", cache_manager)
    
    assert group is dummy_group
    assert group.rank() == 0
    assert group.size() == 1


def test_shard_and_load_handles_distributed_error(monkeypatch, tmp_path):
    """Test shard_and_load handles distributed setup errors."""
    cache_manager = ModelCacheManager(tmp_path)
    
    # Mock distributed init to raise error
    monkeypatch.setattr(dim.mx.distributed, "init", lambda: (_ for _ in ()).throw(Exception("MPI error")))
    
    with pytest.raises(DistributedSetupError) as exc_info:
        dim.shard_and_load("dummy-model", cache_manager)
    
    assert "Failed to initialize distributed group" in str(exc_info.value)


def test_shard_and_load_handles_model_load_error(monkeypatch, tmp_path):
    """Test shard_and_load handles model loading errors."""
    cache_manager = ModelCacheManager(tmp_path)
    
    # Mock successful distributed init
    dummy_group = DummyGroup()
    monkeypatch.setattr(dim.mx.distributed, "init", lambda: dummy_group)
    
    # Mock cache manager to raise error
    monkeypatch.setattr(
        cache_manager, 
        "download_model_files", 
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Download failed"))
    )
    
    with pytest.raises(ModelLoadError) as exc_info:
        dim.shard_and_load("dummy-model", cache_manager)
    
    assert "Failed to load model" in str(exc_info.value)


def test_generate_response_formats_chat_prompt(monkeypatch):
    """Test generate_response formats chat prompts correctly."""
    # Mock model and tokenizer
    model = Mock()
    tokenizer = Mock()
    tokenizer.chat_template = "template"
    tokenizer.apply_chat_template = Mock(return_value="formatted_prompt")
    
    # Mock stream_generate
    response_mock = Mock()
    response_mock.text = "Generated text"
    response_mock.prompt_tokens = 10
    response_mock.generation_tokens = 20
    response_mock.prompt_tps = 100.0
    response_mock.generation_tps = 50.0
    response_mock.peak_memory = 1.5
    
    monkeypatch.setattr(dim, "stream_generate", lambda *args, **kwargs: [response_mock])
    
    stats = dim.generate_response(
        model, tokenizer, "Test prompt", max_tokens=50, rank=0
    )
    
    # Verify chat template was applied
    tokenizer.apply_chat_template.assert_called_once()
    
    # Verify stats returned
    assert stats["generation_tokens"] == 20
    assert stats["prompt_tokens"] == 10
    assert stats["peak_memory"] == 1.5


def test_main_list_cached_models(monkeypatch, capsys, tmp_path):
    """Test main function with --list-cached option."""
    # Mock command line args
    test_args = ["script", "--list-cached", "dummy prompt"]
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Mock cache manager
    with patch("distributed_inference_mlx.ModelCacheManager") as mock_cache_class:
        mock_cache = Mock()
        mock_cache.list_cached_models.return_value = ["model1", "model2"]
        mock_cache.get_cache_size.return_value = {"model1": 1.5, "model2": 2.3}
        mock_cache_class.return_value = mock_cache
        
        # Run main (should exit after listing)
        try:
            dim.main()
        except SystemExit:
            pass
        
        # Check output
        captured = capsys.readouterr()
        assert "Cached models:" in captured.out
        assert "model1 (1.50 GB)" in captured.out
        assert "model2 (2.30 GB)" in captured.out


def test_main_clear_cache(monkeypatch, tmp_path):
    """Test main function with --clear-cache option."""
    # Mock command line args
    test_args = ["script", "--clear-cache", "dummy prompt"]
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Mock everything needed for main
    with patch("distributed_inference_mlx.ModelCacheManager") as mock_cache_class:
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        
        # Mock shard_and_load
        dummy_group = DummyGroup()
        with patch("distributed_inference_mlx.shard_and_load") as mock_shard:
            mock_shard.return_value = (Mock(), Mock(), dummy_group)
            
            # Mock generate_response
            with patch("distributed_inference_mlx.generate_response") as mock_gen:
                mock_gen.return_value = {"generation_time": 1.0}
                
                # Run main
                try:
                    dim.main()
                except SystemExit:
                    pass
                
                # Verify clear_cache was called
                mock_cache.clear_cache.assert_called_once()

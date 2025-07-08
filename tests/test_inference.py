import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import distributed_inference_mlx as dim


def test_download_returns_path(monkeypatch, tmp_path):
    def fake_snapshot(repo, allow_patterns):
        p = tmp_path / "model"
        p.mkdir()
        return str(p)

    monkeypatch.setattr(dim, "snapshot_download", fake_snapshot)
    result = dim.download("repo", ["*.json"])
    assert isinstance(result, Path)
    assert result.exists()


class DummyGroup:
    def __init__(self):
        self._rank = 0
        self._size = 1

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def test_shard_and_load_initializes_group(monkeypatch, tmp_path):
    # fake download returns tmp_path
    monkeypatch.setattr(dim, "download", lambda repo, allow_patterns: tmp_path)

    dummy_group = DummyGroup()
    monkeypatch.setattr(dim.mx.distributed, "init", lambda: dummy_group)
    monkeypatch.setattr(dim.mx.distributed, "all_sum", lambda arr, stream=None: arr)
    monkeypatch.setattr(dim.mx, "array", lambda x: x)
    monkeypatch.setattr(dim.mx, "cpu", None)
    monkeypatch.setattr(dim.mx, "eval", lambda *args, **kwargs: None)

    class DummyModel:
        def parameters(self):
            return [1]

    monkeypatch.setattr(dim, "load_model", lambda path, lazy=False, strict=True: (DummyModel(), None))
    monkeypatch.setattr(dim, "load_tokenizer", lambda path: object())

    model, tokenizer, group = dim.shard_and_load("dummy")
    assert group is dummy_group
    assert group.rank() == 0
    assert group.size() == 1

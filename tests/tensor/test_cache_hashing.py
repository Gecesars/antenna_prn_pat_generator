from __future__ import annotations

from core.image_tensor.pipeline.cache import PipelineCache


def test_cache_key_stability():
    c = PipelineCache(root="cache/image_tensor_test")
    sig = {"name": "p", "ops": [{"op": "gamma", "gamma": 1.1}]}
    k1 = c.key_for("abc", sig, "1")
    k2 = c.key_for("abc", sig, "1")
    k3 = c.key_for("abc", {"name": "p", "ops": [{"op": "gamma", "gamma": 1.2}]}, "1")
    assert k1 == k2
    assert k1 != k3

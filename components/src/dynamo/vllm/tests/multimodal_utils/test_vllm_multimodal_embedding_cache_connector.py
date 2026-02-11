# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DynamoMultimodalEmbeddingCacheConnector in ec_both mode."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole

from dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector import (
    DynamoMultimodalEmbeddingCacheConnector,
    MMMeta,
    MultimodalEmbeddingCacheConnectorMetadata,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
]

DEVICE = "cuda"


def _make_vllm_config(capacity_gb: float = 4.0) -> MagicMock:
    """Build a minimal mock VllmConfig with ec_transfer_config for ec_both."""
    ec_transfer_config = MagicMock()
    ec_transfer_config.ec_connector_extra_config = {
        "multimodal_embedding_cache_capacity_gb": capacity_gb,
    }

    vllm_config = MagicMock()
    vllm_config.ec_transfer_config = ec_transfer_config
    return vllm_config


def _make_request(mm_hashes: list[str], num_tokens: list[int]) -> MagicMock:
    """Build a minimal mock Request with mm_features."""
    request = MagicMock()
    features = []
    for h in mm_hashes:
        f = MagicMock()
        f.identifier = h
        features.append(f)
    request.mm_features = features

    def _get_num_encoder_embeds(index: int) -> int:
        return num_tokens[index]

    request.get_num_encoder_embeds = _get_num_encoder_embeds
    return request


class TestDynamoMultimodalEmbeddingCacheConnectorEcBoth:
    """Test DynamoMultimodalEmbeddingCacheConnector in ec_both mode."""

    def _make_connector(
        self, role: ECConnectorRole, capacity_gb: float = 4.0
    ) -> DynamoMultimodalEmbeddingCacheConnector:
        vllm_config = _make_vllm_config(capacity_gb)
        return DynamoMultimodalEmbeddingCacheConnector(vllm_config, role=role)

    # -- Scheduler-side --

    def test_has_caches_always_false(self) -> None:
        conn = self._make_connector(ECConnectorRole.SCHEDULER)
        request = _make_request(["h1", "h2"], [10, 20])
        assert conn.has_caches(request) == [False, False]

    def test_update_state_and_build_meta(self) -> None:
        conn = self._make_connector(ECConnectorRole.SCHEDULER)
        request = _make_request(["h1", "h2"], [10, 20])

        conn.update_state_after_alloc(request, 0)
        conn.update_state_after_alloc(request, 1)

        meta = conn.build_connector_meta(MagicMock())
        assert isinstance(meta, MultimodalEmbeddingCacheConnectorMetadata)
        assert len(meta.mm_datas) == 2
        assert meta.mm_datas[0] == MMMeta(mm_hash="h1", num_token=10)
        assert meta.mm_datas[1] == MMMeta(mm_hash="h2", num_token=20)

    def test_build_meta_resets_state(self) -> None:
        conn = self._make_connector(ECConnectorRole.SCHEDULER)
        request = _make_request(["h1"], [10])
        conn.update_state_after_alloc(request, 0)

        conn.build_connector_meta(MagicMock())
        meta = conn.build_connector_meta(MagicMock())
        assert len(meta.mm_datas) == 0

    # -- Worker-side: GPU <-> CPU cache movement --

    def test_save_moves_gpu_tensor_to_cpu_cache(self) -> None:
        """save_caches should copy a GPU tensor to the CPU-side LRU cache."""
        conn = self._make_connector(ECConnectorRole.WORKER, capacity_gb=0.01)

        gpu_tensor = torch.randn(5, 64, device=DEVICE)
        encoder_cache: dict[str, torch.Tensor] = {"h1": gpu_tensor}

        conn.save_caches(encoder_cache, "h1")
        assert conn._cache_manager.stats["entries"] == 1

        # The cached tensor should live on CPU
        cached = conn._cache_manager.get("h1")
        assert cached is not None
        assert cached.tensor.device.type == "cpu"
        assert torch.equal(cached.tensor, gpu_tensor.cpu())

    def test_load_promotes_cpu_cache_to_gpu(self) -> None:
        """start_load_caches should copy a cached CPU tensor back to GPU."""
        conn = self._make_connector(ECConnectorRole.WORKER, capacity_gb=0.01)

        # Seed the cache: save a GPU tensor, then clear encoder_cache
        original = torch.randn(5, 64, device=DEVICE)
        encoder_cache: dict[str, torch.Tensor] = {"h1": original}
        conn.save_caches(encoder_cache, "h1")
        encoder_cache.clear()

        # Request a load
        meta = MultimodalEmbeddingCacheConnectorMetadata()
        meta.add_mm_data(MMMeta(mm_hash="h1", num_token=5))
        conn.bind_connector_metadata(meta)

        conn.start_load_caches(encoder_cache)

        # The tensor should now be back on GPU with matching values
        assert "h1" in encoder_cache
        loaded = encoder_cache["h1"]
        assert loaded.device.type == "cuda"
        torch.cuda.synchronize()
        assert torch.equal(loaded, original)

    def test_load_skips_already_present(self) -> None:
        """start_load_caches should not overwrite a tensor already in encoder_cache."""
        conn = self._make_connector(ECConnectorRole.WORKER, capacity_gb=0.01)

        # Seed the cache
        gpu_tensor = torch.randn(5, 64, device=DEVICE)
        encoder_cache: dict[str, torch.Tensor] = {"h1": gpu_tensor}
        conn.save_caches(encoder_cache, "h1")

        # encoder_cache still has "h1" — load should be a no-op
        meta = MultimodalEmbeddingCacheConnectorMetadata()
        meta.add_mm_data(MMMeta(mm_hash="h1", num_token=5))
        conn.bind_connector_metadata(meta)

        conn.start_load_caches(encoder_cache)
        # Should be the exact same object, not a new copy
        assert encoder_cache["h1"] is gpu_tensor

    def test_load_cache_miss_is_noop(self) -> None:
        """start_load_caches with a hash not in cache should leave encoder_cache unchanged."""
        conn = self._make_connector(ECConnectorRole.WORKER, capacity_gb=0.01)
        encoder_cache: dict[str, torch.Tensor] = {}

        meta = MultimodalEmbeddingCacheConnectorMetadata()
        meta.add_mm_data(MMMeta(mm_hash="missing", num_token=10))
        conn.bind_connector_metadata(meta)

        conn.start_load_caches(encoder_cache)
        assert "missing" not in encoder_cache

    def test_save_missing_hash_is_noop(self) -> None:
        """save_caches with a hash not in encoder_cache should not crash or cache anything."""
        conn = self._make_connector(ECConnectorRole.WORKER, capacity_gb=0.01)
        encoder_cache: dict[str, torch.Tensor] = {}
        conn.save_caches(encoder_cache, "not_there")
        assert conn._cache_manager.stats["entries"] == 0

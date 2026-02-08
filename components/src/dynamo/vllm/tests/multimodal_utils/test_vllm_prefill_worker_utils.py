# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fetch_embeddings_via_local_cache in prefill_worker_utils."""

from unittest.mock import AsyncMock, patch

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_utils import prefill_worker_utils as mod
from dynamo.vllm.multimodal_utils.protocol import MultiModalGroup, MultiModalInput

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class TestFetchEmbeddingsViaLocalCache:
    @pytest.mark.asyncio
    async def test_all_cached(self):
        """All URLs cached -> no encode worker call."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(1, 10, dtype=torch.float16)
        grid = [[1, 2, 3]]
        url = "http://img1.png"
        key = mod.get_embedding_hash(url)
        cache.set(key, CachedEmbedding(tensor=tensor, image_grid_thw=grid))

        with patch.object(
            mod,
            "fetch_embeddings_via_stream",
            new_callable=AsyncMock,
        ) as mock_fetch:
            groups = await mod.fetch_embeddings_via_local_cache(
                cache,
                AsyncMock(),
                [url],
                "req-1",
                torch.float16,
                "cpu",
                None,
            )

        mock_fetch.assert_not_awaited()
        assert len(groups) == 1
        assert torch.equal(groups[0].cached_embedding, tensor)
        assert groups[0].image_grid_thw == grid

    @pytest.mark.asyncio
    async def test_all_uncached(self):
        """All URLs uncached -> full encode worker call, results cached."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        url = "http://img1.png"
        tensor = torch.randn(1, 10, dtype=torch.float16)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=[[1, 2, 3]],
        )

        with (
            patch.object(
                mod,
                "fetch_embeddings_via_stream",
                new_callable=AsyncMock,
                return_value=[fake_group],
            ) as mock_fetch,
            patch.object(
                mod,
                "load_embeddings",
                new_callable=AsyncMock,
                return_value=tensor,
            ),
        ):
            groups = await mod.fetch_embeddings_via_local_cache(
                cache,
                AsyncMock(),
                [url],
                "req-1",
                torch.float16,
                "cpu",
                None,
            )

        mock_fetch.assert_awaited_once()
        assert len(groups) == 1
        assert torch.equal(groups[0].cached_embedding, tensor)

        # Verify result was cached
        key = mod.get_embedding_hash(url)
        cached = cache.get(key)
        assert cached is not None
        assert torch.equal(cached.tensor, tensor)
        assert cached.image_grid_thw == [[1, 2, 3]]

    @pytest.mark.asyncio
    async def test_mixed_cache(self):
        """Mixed cache hits/misses -> only misses sent to encode workers."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)

        url_cached = "http://cached.png"
        url_miss = "http://miss.png"
        cached_tensor = torch.randn(1, 10, dtype=torch.float16)
        miss_tensor = torch.randn(1, 10, dtype=torch.float16)

        # Pre-populate cache for first URL
        key = mod.get_embedding_hash(url_cached)
        cache.set(key, CachedEmbedding(tensor=cached_tensor, image_grid_thw=None))

        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=[[4, 5, 6]],
        )

        with (
            patch.object(
                mod,
                "fetch_embeddings_via_stream",
                new_callable=AsyncMock,
                return_value=[fake_group],
            ) as mock_fetch,
            patch.object(
                mod,
                "load_embeddings",
                new_callable=AsyncMock,
                return_value=miss_tensor,
            ),
        ):
            groups = await mod.fetch_embeddings_via_local_cache(
                cache,
                AsyncMock(),
                [url_cached, url_miss],
                "req-1",
                torch.float16,
                "cpu",
                None,
            )

        # Only the miss URL should have been sent
        mock_fetch.assert_awaited_once()
        call_args = mock_fetch.call_args
        assert call_args[0][1] == [url_miss]

        assert len(groups) == 2
        assert torch.equal(groups[0].cached_embedding, cached_tensor)
        assert torch.equal(groups[1].cached_embedding, miss_tensor)

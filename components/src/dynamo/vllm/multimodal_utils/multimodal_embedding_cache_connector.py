# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


@dataclass
class MMMeta:
    mm_hash: str
    num_token: int


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self) -> None:
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta) -> None:
        self.mm_datas.append(mm_data)


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """EC connector for ec_both mode with in-memory CPU embedding cache.

    Keeps a CPU-side LRU cache so that repeated multimodal inputs skip
    encoding.  On every forward pass the worker calls save_caches (caches
    the GPU tensor to CPU) and start_load_caches (promotes a CPU hit back
    to the GPU encoder_cache dict so vLLM skips encoding).
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError(
                "ec_transfer_config must be set for DynamoMultimodalEmbeddingCacheConnector"
            )

        if "multimodal_embedding_cache_capacity_gb" not in (
            transfer_config.ec_connector_extra_config or {}
        ):
            raise ValueError(
                "multimodal_embedding_cache_capacity_gb must be set in "
                "ec_connector_extra_config for DynamoMultimodalEmbeddingCacheConnector"
            )
        capacity_gb: float = transfer_config.ec_connector_extra_config[
            "multimodal_embedding_cache_capacity_gb"
        ]
        capacity_bytes: int = int(capacity_gb * 1024**3)
        self._cache_manager = MultimodalEmbeddingCacheManager(capacity_bytes)

        # Scheduler-side: tracks mm_hashes that need loading this step
        self._mm_datas_need_loads: dict[str, int] = {}

        # Scheduler-side: collects ALL hashes seen in has_caches() this step,
        # so build_connector_meta() can pass them to the worker. The worker's
        # start_load_caches() will check its own CPU cache for each hash and
        # promote hits to encoder_cache (so vLLM skips encoding).
        # This avoids cross-process state sync: the scheduler doesn't need to
        # know what the worker's CPU cache contains.
        self._hashes_this_step: set[str] = set()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def has_caches(self, request: "Request") -> list[bool]:
        """Always return False — the scheduler schedules local encoding.

        We record the hashes here so build_connector_meta() can pass them
        to the worker. The worker's start_load_caches() will check its own
        CPU cache and promote any hits into encoder_cache before encoding
        runs, so vLLM's encoder will skip re-encoding cached items.
        """
        for f in request.mm_features:
            self._hashes_this_step.add(f.identifier)
        return [False] * len(request.mm_features)

    def has_cache_item(self, identifier: str) -> bool:
        """Always return False — same strategy as has_caches.

        Used by newer vLLM versions. Record the hash for metadata.
        """
        self._hashes_this_step.add(identifier)
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash: str = request.mm_features[index].identifier
        num_encoder_token: int = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        meta = MultimodalEmbeddingCacheConnectorMetadata()

        # Include hashes from update_state_after_alloc (external loads)
        for mm_hash, num_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta(mm_hash=mm_hash, num_token=num_token))

        # Include ALL hashes seen in has_caches() this step.
        # The worker will check its CPU cache for each and promote hits.
        for mm_hash in self._hashes_this_step:
            if mm_hash not in self._mm_datas_need_loads:
                meta.add_mm_data(MMMeta(mm_hash=mm_hash, num_token=0))

        logger.debug(
            "build_connector_meta: from_update_state=%d, from_has_caches=%d, "
            "total_meta_mm_datas=%d",
            len(self._mm_datas_need_loads),
            len(self._hashes_this_step),
            len(meta.mm_datas),
        )

        self._mm_datas_need_loads.clear()
        self._hashes_this_step.clear()
        return meta

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        cpu_cache_stats = self._cache_manager.stats
        logger.debug(
            "start_load_caches called: encoder_cache=%d, metadata_mm_datas=%d, "
            "cpu_cache_entries=%d, cpu_cache_hits=%d, cpu_cache_misses=%d, "
            "cache_mgr_id=%s",
            len(encoder_cache),
            len(metadata.mm_datas),
            cpu_cache_stats["entries"],
            cpu_cache_stats["hits"],
            cpu_cache_stats["misses"],
            id(self._cache_manager),
        )

        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                logger.debug(
                    "GPU encoder_cache hit: mm_hash=%s already on GPU, skipping",
                    mm_data.mm_hash,
                )
                continue
            cached = self._cache_manager.get(mm_data.mm_hash)
            if cached is not None:
                # non_blocking=True enqueues the H2D copy on the CUDA stream
                # without blocking the CPU thread.  CUDA stream ordering
                # guarantees the copy completes before the encoder forward
                # pass that consumes this tensor.
                encoder_cache[mm_data.mm_hash] = cached.tensor.to(
                    "cuda", non_blocking=True
                )
                logger.debug(
                    "Cache hit: loaded mm_hash=%s from CPU cache to GPU",
                    mm_data.mm_hash,
                )
            else:
                logger.debug(
                    "CPU cache miss: mm_hash=%s not in CPU cache (entries=%d)",
                    mm_data.mm_hash,
                    cpu_cache_stats["entries"],
                )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        cpu_cache_stats = self._cache_manager.stats
        logger.debug(
            "save_caches called: mm_hash=%s, encoder_cache=%d, "
            "cpu_cache_entries=%d, cache_mgr_id=%s",
            mm_hash,
            len(encoder_cache),
            cpu_cache_stats["entries"],
            id(self._cache_manager),
        )
        if mm_hash not in encoder_cache:
            logger.warning(
                "save_caches called but mm_hash=%s not in encoder_cache", mm_hash
            )
            return
        if self._cache_manager.contains(mm_hash):
            logger.debug(
                "Skipping redundant save: mm_hash=%s already in CPU cache", mm_hash
            )
            return
        cpu_tensor = encoder_cache[mm_hash].cpu()
        self._cache_manager.set(
            mm_hash, CachedEmbedding(tensor=cpu_tensor, image_grid_thw=None)
        )
        logger.debug(
            "Saved mm_hash=%s to CPU cache (now %d entries)",
            mm_hash,
            self._cache_manager.stats["entries"],
        )

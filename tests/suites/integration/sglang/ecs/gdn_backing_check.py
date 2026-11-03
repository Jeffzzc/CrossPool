"""Real pinned GDN kernels against production sparse storage and a small native oracle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.multiprocessing
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, KvBufferDesc, MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish
from tests.suites.integration.sglang.ecs.gdn_preflight import (
    MODEL,
    compare_state,
    hybrid_pool,
    make_args,
    make_backend,
    make_batch,
)

from xpool import ecs
from xpool.config import init_global_config
from xpool.integrations.sglang.ecs.gdn import ElasticMambaPool, GdnBackingPlan, GdnVmmBacking, prepared_gdn_scope
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking


def check_rank(rank: int, tp_size: int, output: Path) -> None:
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    arguments = make_args(init_global_config().model_path_of(MODEL), tp_size)
    publish(arguments, role="scheduler")
    model = ModelConfig.from_server_args(arguments)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank,
        distributed_init_method=(output / "rendezvous").resolve().as_uri(),
        local_rank=rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=tp_size)
    backing: GdnVmmBacking | None = None
    full: KvVmmBacking | None = None
    try:
        params = mambaish_config(model).mamba2_cache_params
        assert isinstance(params, Mamba2CacheParams)
        layouts = GdnVmmBacking.describe(device=device, size=127, cache_params=params, layer_ids=params.layers)
        full_layout = KvVmmBacking.describe(
            device=device,
            storage_dtype=torch.bfloat16,
            token_page_size=1,
            buffers_per_layer=2,
            descriptors=tuple(
                KvBufferDesc(f"kv-{index}", (4097, 4 // tp_size, 256), row_bytes=2048 // tp_size, tokens_per_row=1)
                for index in range(16)
            ),
        )
        quota = ecs.MemoryQuota(
            sum(layout.profile.bytes_for(layout.profile.bundle_capacity) for layout in layouts)
            + full_layout.profile.bytes_for(full_layout.profile.bundle_capacity)
        )
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        backing = GdnVmmBacking(layouts, quota)
        with (
            prepared_gdn_scope(backing),
            patch.object(HybridReqToTokenPool, "mamba_pool_cls", ElasticMambaPool),
            patch.object(MambaPool, "__init__", side_effect=AssertionError("native complete state allocation")),
        ):
            pool = hybrid_pool(params, str(device), 127)
        vmm = pool.mamba_pool
        assert isinstance(vmm, ElasticMambaPool)
        allocator_delta = torch.cuda.memory_allocated() - before
        assert allocator_delta < backing.profile.floor_bytes
        assert backing.backed_bytes == backing.profile.floor_bytes
        full = KvVmmBacking(full_layout, quota)
        # This four-slot oracle is test-only and never allocated in production.
        native = hybrid_pool(params, str(device), 4)
        backend, oracle = (
            make_backend(pool, model, arguments, str(device)),
            make_backend(native, model, arguments, str(device)),
        )
        slots = torch.tensor([1, 2], dtype=torch.int64, device=device)

        def prepare(target: tuple[int, ...], reference: tuple[int, ...]) -> None:
            ids = torch.tensor(target, dtype=torch.int64, device=device)
            assert backing is not None and backing.try_map(backing.plan_slots(ids))
            assert backing.missing_bundle_counts(ids).tolist() == [0] * len(target)
            vmm.clear_slots(ids)
            native.mamba_pool.clear_slots(torch.tensor(reference, dtype=torch.int64, device=device))
            pool.req_index_to_mamba_index_mapping[:2] = ids[:2].to(torch.int32)
            native.req_index_to_mamba_index_mapping[:2] = torch.tensor(reference[:2], dtype=torch.int32, device=device)

        prepare((1, 2, 3), (1, 2, 3))
        hk = params.shape.num_k_heads_per_tp
        hv, dv, dk = params.shape.temporal
        dim = 2 * hk * dk + hv * dv
        generator = torch.Generator(device=device).manual_seed(1729)
        weight = torch.randn(dim, 4, generator=generator, device=device, dtype=torch.bfloat16) * 0.03
        a_log = torch.randn(hv, generator=generator, device=device, dtype=torch.float32)
        bias = torch.randn(hv, generator=generator, device=device, dtype=torch.float32)
        layers = [
            RadixLinearAttention(layer, hk, hk, hv, dk, dk, dv, conv_weights=weight, A_log=a_log, dt_bias=bias)
            for layer in params.layers
        ]
        for lengths, prefixes in (((17, 9), (0, 0)), ((3, 5), (17, 9))):
            batch = make_batch(str(device), lengths, prefixes)
            backend.init_forward_metadata(batch)
            oracle.init_forward_metadata(batch)
            mixed = torch.randn(sum(lengths), dim, generator=generator, device=device, dtype=torch.bfloat16)
            a = torch.randn(sum(lengths), hv, generator=generator, device=device, dtype=torch.bfloat16)
            b = torch.randn_like(a)
            for layer in layers:
                torch.testing.assert_close(
                    backend.forward_extend(layer, batch, mixed.clone(), a, b),
                    oracle.forward_extend(layer, batch, mixed.clone(), a, b),
                    rtol=0,
                    atol=0,
                )
        compare_state(vmm, native.mamba_pool, (1, 2), (1, 2))
        destination = torch.tensor([3], dtype=torch.int64, device=device)
        vmm.copy_from(slots[:1], destination)
        native.mamba_pool.copy_from(slots[:1], destination)
        compare_state(vmm, native.mamba_pool, (1, 2, 3), (1, 2, 3))
        vmm.clear_slots(destination)
        native.mamba_pool.clear_slots(destination)
        compare_state(vmm, native.mamba_pool, (1, 2, 3), (1, 2, 3))
        batch = make_batch(str(device), (1, 1), (20, 14))
        batch.forward_mode = ForwardMode.DECODE
        backend.init_cuda_graph_state(2, 2)
        backend.forward_metadata = backend._capture_metadata(2, batch.req_pool_indices, ForwardMode.DECODE, None)
        oracle.init_forward_metadata(batch)
        mixed = torch.randn(2, dim, generator=generator, device=device, dtype=torch.bfloat16)
        a = torch.randn(2, hv, generator=generator, device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)

        def decode(owner: GDNAttnBackend) -> torch.Tensor:
            return torch.stack([owner.forward_decode(layer, batch, mixed.clone(), a, b) for layer in layers])

        decode(backend)
        decode(oracle)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = decode(backend)
        pointer = backend.state_indices_list[1].data_ptr()
        state_pointers = tuple(plane.view.data_ptr() for plane in backing.planes)
        for target, reference, padding in (((1, 2), (1, 2), 0), ((3, 2), (3, 2), 1), ((96, 2), (1, 2), 0)):
            batch.req_pool_indices.copy_(torch.tensor([0, 1], dtype=torch.int64, device=device))
            prepare(target, reference)
            for repetition in range(3):
                backend._replay_metadata(2, batch.req_pool_indices, ForwardMode.DECODE, None, None, num_padding=padding)
                batch._original_batch_size = 2 - padding
                oracle.init_forward_metadata(batch)
                expected = decode(oracle)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                compare_state(vmm, native.mamba_pool, target, reference)
            assert pointer == backend.state_indices_list[1].data_ptr()
            assert backend.state_indices_list[1].tolist() == list(target[: 2 - padding]) + [-1] * padding
        batch._original_batch_size = None
        # Withdraw the distant logical owner and wait for the actual forward
        # stream before unmapping. No retirement refund precedes this event.
        pool.req_index_to_mamba_index_mapping[:2] = slots.to(torch.int32)
        backend._replay_metadata(2, batch.req_pool_indices, ForwardMode.DECODE, None, None, num_padding=0)
        quota.set_limit(quota.charged_bytes)
        assert not full.try_map((1,))
        charged_before_event = quota.charged_bytes
        retiring = []
        for plane in backing.planes:
            far = set(plane.bundles_for(torch.tensor([96], device=device)))
            pinned = set(plane.bundles_for(torch.tensor([0, 1, 2, 3], device=device)))
            retiring.append(tuple(sorted(far - pinned)))
        free = torch.arange(4, 128, dtype=torch.int64, device=device)
        backing.begin_retirement(GdnBackingPlan(tuple(retiring)), free_slots=free, stream=torch.cuda.current_stream())
        assert quota.charged_bytes == charged_before_event
        assert backing.retirement is not None
        backing.retirement.event.synchronize()
        released = backing.finish_retirement()
        assert released > 0 and quota.charged_bytes == charged_before_event - released
        assert full.try_map((1,))
        full_release = full.unmap((1,))
        assert full_release > 0
        batch.req_pool_indices.copy_(torch.tensor([0, 1], dtype=torch.int64, device=device))
        prepare((96, 2), (1, 2))
        backend._replay_metadata(2, batch.req_pool_indices, ForwardMode.DECODE, None, None, num_padding=0)
        oracle.init_forward_metadata(batch)
        expected = decode(oracle)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        compare_state(vmm, native.mamba_pool, (96, 2), (1, 2))
        assert state_pointers == tuple(plane.view.data_ptr() for plane in backing.planes)
        assert vmm.mamba_cache is not None
        assert torch.count_nonzero(vmm.mamba_cache.temporal[:, 0]) == 0
        (output / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "tp_size": tp_size,
                    "rank": rank,
                    "floor_bytes": backing.profile.floor_bytes,
                    "torch_allocator_delta": allocator_delta,
                    "output_error": 0,
                    "state_error": 0,
                    "layers_exercised": params.layers,
                    "same_graph_after_remap": True,
                    "released_gdn_bytes": released,
                    "reused_full_bytes": full_release,
                    "planes": [
                        {
                            "shape": plane.layout.shape,
                            "dtype": str(plane.layout.dtype),
                            "stride": plane.view.stride(),
                            "minimum_granularity": plane.layout.minimum_granularity,
                            "recommended_granularity": plane.layout.granularity,
                            "layer_padding": plane.layout.layer_stride_bytes
                            - plane.layout.shape[1] * plane.layout.row_bytes,
                            "bundle_bytes": plane.layout.profile.bundle_bytes,
                        }
                        for plane in backing.planes
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        del graph, result
    finally:
        torch.cuda.synchronize()
        if full is not None:
            full.close()
        if backing is not None:
            backing.close()
            assert backing.quota.charged_bytes == 0
        destroy_model_parallel()
        destroy_distributed_environment()


def run_backing_check(tp_size: int, output: Path) -> None:
    os.environ["SGLANG_PLUGINS"] = "__gdn_production_storage_no_plugins__"
    output.mkdir()
    if tp_size == 1:
        check_rank(0, tp_size, output)
    else:
        torch.multiprocessing.spawn(check_rank, args=(tp_size, output), nprocs=tp_size, join=True)

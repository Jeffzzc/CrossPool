"""Pinned-SGLang bootstrap and numerical evidence for the isolated GDN prototype."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.distributed
import torch.multiprocessing
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.utils import resolve_linear_attn_backends
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool, MHATokenToKVPool
from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule, publish
from sglang.srt.server_args import ServerArgs
from tests.suites.integration.sglang.ecs.gdn_vmm_prototype import PrototypeMambaPool

from xkit.network import TcpEndpointReservation, TcpPortSpace
from xpool.config import init_global_config
from xpool.model import ModelId
from xpool.utils.cuda import check_driver

MODEL = ModelId("Qwen/Qwen3.5-9B")


def make_args(path: Path, tp_size: int, decode: str = "disabled", prefill: str = "disabled") -> ServerArgs:
    # ServerArgs is a runtime-generated msgspec Struct; its constructor's
    # keyword signature is unavailable to the static checker.
    return cast(Any, ServerArgs)(
        model_path=str(path),
        json_model_override_args='{"language_model_only":true}',
        disable_overlap_schedule=True,
        mamba_radix_cache_strategy="no_buffer",
        attention_backend="triton",
        linear_attn_prefill_backend="triton",
        linear_attn_decode_backend="triton",
        page_size=1,
        tp_size=tp_size,
        dtype="bfloat16",
        cuda_graph_backend_decode=decode,
        cuda_graph_backend_prefill=prefill,
        cuda_graph_bs_decode=[1, 2],
        cuda_graph_bs_prefill=[16, 32],
        max_running_requests=2,
        max_total_tokens=1024,
        max_mamba_cache_size=128,
        context_length=1024,
        chunked_prefill_size=32,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=True,
    )


def configuration_evidence(path: Path, tp_size: int) -> dict[str, object]:
    bad = make_args(path, tp_size)
    bad.json_model_override_args = "{}"
    bad.language_model_only = True
    try:
        bad.resolve_once()
    except ValueError as error:
        flag_error = str(error)
        assert "--language-model-only does not support" in flag_error
    else:
        raise AssertionError("pinned language-model-only allowlist changed; reassess the launch path")
    modes = []
    for decode, prefill in (("disabled", "disabled"), ("full", "disabled"), ("full", "breakable")):
        arguments = make_args(path, tp_size, decode, prefill)
        publish(arguments, role="scheduler")
        config = ModelConfig.from_server_args(arguments)
        assert config.is_lm_only and not config.is_multimodal
        graph = get_exec().graph.cuda_graph_config
        assert graph.decode.backend == decode and graph.prefill.backend == prefill
        assert not get_exec().mamba.enable_mamba_extra_buffer
        modes.append(asdict(graph))
    return {
        "official_architecture": config.hf_config.architectures,
        "language_model_only_flag_error": flag_error,
        "config_override": {"language_model_only": True},
        "is_multimodal": config.is_multimodal,
        "resolved_graph_modes": modes,
    }


def hybrid_pool(cache_params: Mamba2CacheParams, device: str, slots: int) -> HybridReqToTokenPool:
    return HybridReqToTokenPool(
        size=4,
        mamba_size=slots,
        mamba_spec_state_size=4,
        max_context_len=64,
        device=device,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=cache_params.layers,
        enable_mamba_extra_buffer=False,
        enable_overlap_schedule=False,
    )


def make_backend(pool: HybridReqToTokenPool, model: ModelConfig, arguments: ServerArgs, device: str) -> GDNAttnBackend:
    # A real, deliberately uninitialized runner supplies only the concrete
    # backend constructor contract. No weights or fake kernel implementation.
    runner = object.__new__(ModelRunner)
    runner.device = device
    runner.req_to_token_pool = pool
    runner.token_to_kv_pool = None
    runner.server_args = arguments
    runner.model_config = model
    runner.is_draft_worker = False
    runner.linear_attn_backends = resolve_linear_attn_backends()
    return GDNAttnBackend(runner)


def make_batch(device: str, lengths: tuple[int, int], prefixes: tuple[int, int] = (0, 0)) -> ForwardBatch:
    count = sum(lengths)
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.zeros(count, dtype=torch.int64, device=device),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=device),
        seq_lens=torch.tensor([a + b for a, b in zip(lengths, prefixes, strict=True)], device=device),
        out_cache_loc=torch.arange(count, dtype=torch.int64, device=device),
        seq_lens_sum=count + sum(prefixes),
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        extend_prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device=device),
        extend_start_loc=torch.tensor([0, lengths[0]], dtype=torch.int32, device=device),
        extend_num_tokens=count,
        extend_seq_lens_cpu=list(lengths),
        extend_prefix_lens_cpu=list(prefixes),
    )


def compare_state(
    vmm: MambaPool, native: MambaPool, vmm_slots: tuple[int, ...], native_slots: tuple[int, ...]
) -> float:
    maximum = 0.0
    for actual, expected in zip(
        (*vmm.mamba_cache.conv, vmm.mamba_cache.temporal),
        (*native.mamba_cache.conv, native.mamba_cache.temporal),
        strict=True,
    ):
        left, right = actual[:, list(vmm_slots)], expected[:, list(native_slots)]
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        maximum = max(maximum, float((left.float() - right.float()).abs().max()))
    return maximum


def kernel_evidence(
    pool: HybridReqToTokenPool,
    native: HybridReqToTokenPool,
    model: ModelConfig,
    arguments: ServerArgs,
    cache_params: Mamba2CacheParams,
) -> dict[str, object]:
    vmm = pool.mamba_pool
    assert isinstance(vmm, PrototypeMambaPool)
    device = pool.device
    slots = torch.tensor([1, 2], dtype=torch.int64, device=device)
    vmm.map_slots((1, 2, 3))
    native.mamba_pool.clear_slots(slots)
    pool.req_index_to_mamba_index_mapping[:2] = slots.to(torch.int32)
    native.req_index_to_mamba_index_mapping[:2] = slots.to(torch.int32)
    backend = make_backend(pool, model, arguments, device)
    reference = make_backend(native, model, arguments, device)
    hk = cache_params.shape.num_k_heads_per_tp
    hv, dv, dk = cache_params.shape.temporal
    dim = (2 * hk * dk) + hv * dv
    generator = torch.Generator(device=device).manual_seed(1729)
    conv_weight = torch.randn(dim, 4, generator=generator, device=device, dtype=torch.bfloat16) * 0.03
    a_log = torch.randn(hv, generator=generator, device=device, dtype=torch.float32)
    dt_bias = torch.randn(hv, generator=generator, device=device, dtype=torch.float32)
    layers = [
        RadixLinearAttention(layer, hk, hk, hv, dk, dk, dv, conv_weights=conv_weight, A_log=a_log, dt_bias=dt_bias)
        for layer in cache_params.layers
    ]
    maximum_output_error = 0.0
    for lengths, prefixes in (((17, 9), (0, 0)), ((3, 5), (17, 9))):
        batch = make_batch(device, lengths, prefixes)
        backend.init_forward_metadata(batch)
        reference.init_forward_metadata(batch)
        mixed = torch.randn(sum(lengths), dim, generator=generator, device=device, dtype=torch.bfloat16)
        a = torch.randn(sum(lengths), hv, generator=generator, device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        for layer in layers:
            actual = backend.forward_extend(layer, batch, mixed.clone(), a, b)
            expected = reference.forward_extend(layer, batch, mixed.clone(), a, b)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            maximum_output_error = max(maximum_output_error, float((actual.float() - expected.float()).abs().max()))
    state_error = compare_state(vmm, native.mamba_pool, (1, 2), (1, 2))
    source, destination = slots[:1], torch.tensor([3], dtype=torch.int64, device=device)
    vmm.copy_from(source, destination)
    native.mamba_pool.copy_from(source, destination)
    compare_state(vmm, native.mamba_pool, (1, 2, 3), (1, 2, 3))
    vmm.clear_slots(destination)
    native.mamba_pool.clear_slots(destination)
    compare_state(vmm, native.mamba_pool, (1, 2, 3), (1, 2, 3))

    batch = make_batch(device, (1, 1), (20, 14))
    batch.forward_mode = ForwardMode.DECODE
    backend.init_cuda_graph_state(2, 2)
    backend.forward_metadata = backend._capture_metadata(2, batch.req_pool_indices, ForwardMode.DECODE, None)
    reference.init_forward_metadata(batch)
    mixed = torch.randn(2, dim, generator=generator, device=device, dtype=torch.bfloat16)
    a = torch.randn(2, hv, generator=generator, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)

    def decode_all(owner: GDNAttnBackend) -> torch.Tensor:
        return torch.stack([owner.forward_decode(layer, batch, mixed.clone(), a, b) for layer in layers])

    # Compile once before capture. Restore native state identically afterwards.
    decode_all(backend)
    decode_all(reference)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = decode_all(backend)
    indices_pointer = backend.state_indices_list[1].data_ptr()
    pointers = tuple(plane.view.data_ptr() for plane in vmm.planes)
    replays = []
    for target_slots, reference_slots, padding in (((1, 2), (1, 2), 0), ((3, 2), (3, 2), 1), ((96, 2), (1, 2), 0)):
        batch.req_pool_indices.copy_(torch.tensor([0, 1], dtype=torch.int64, device=device))
        vmm.map_slots(target_slots)
        native.mamba_pool.clear_slots(torch.tensor(reference_slots, dtype=torch.int64, device=device))
        pool.req_index_to_mamba_index_mapping[:2] = torch.tensor(target_slots, dtype=torch.int32, device=device)
        native.req_index_to_mamba_index_mapping[:2] = torch.tensor(reference_slots, dtype=torch.int32, device=device)
        for repetition in range(3):
            backend._replay_metadata(2, batch.req_pool_indices, ForwardMode.DECODE, None, None, num_padding=padding)
            batch._original_batch_size = 2 - padding
            reference.init_forward_metadata(batch)
            expected = decode_all(reference)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
            compare_state(vmm, native.mamba_pool, target_slots, reference_slots)
        assert backend.state_indices_list[1].data_ptr() == indices_pointer
        expected_indices = list(target_slots[: 2 - padding]) + [-1] * padding
        assert backend.state_indices_list[1].tolist() == expected_indices
        replays.append({"slots": target_slots, "padding": padding, "replays": repetition + 1})
    batch._original_batch_size = None

    # Only the distant unowned slot is retired; dummy and live slots retain backing.
    event = torch.cuda.Event()
    event.record()
    event.synchronize()
    before_retirement = vmm.backed_bytes
    released = 0
    for plane in vmm.planes:
        isolated = plane.bundles_for((96,)).difference(plane.bundles_for((0, 1, 2, 3)))
        assert isolated
        released += plane.unmap_bundles(isolated)
    assert vmm.backed_bytes == before_retirement - released
    after_retirement = vmm.backed_bytes
    vmm.map_slots((96, 2))
    native.mamba_pool.clear_slots(slots)
    backend._replay_metadata(2, batch.req_pool_indices, ForwardMode.DECODE, None, None, num_padding=0)
    reference.init_forward_metadata(batch)
    expected = decode_all(reference)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
    compare_state(vmm, native.mamba_pool, (96, 2), (1, 2))
    assert tuple(plane.view.data_ptr() for plane in vmm.planes) == pointers
    assert torch.count_nonzero(vmm.state.temporal[:, 0]) == 0
    assert torch.count_nonzero(vmm.state.conv[0][:, 0]) == 0
    del graph, graph_output
    return {
        "layers_exercised": cache_params.layers,
        "prefill_lengths": [[17, 9], [3, 5]],
        "maximum_output_error": maximum_output_error,
        "maximum_state_error": state_error,
        "clear_copy_exact": True,
        "decode_capture_replay": replays,
        "state_indices_pointer_stable": True,
        "tensor_pointers_stable": True,
        "released_and_remapped_bytes": released,
        "backed_bytes_before_retirement": before_retirement,
        "backed_bytes_after_retirement": after_retirement,
        "same_graph_replayed_after_remap": True,
    }


def run_rank(rank: int, tp_size: int, output: Path) -> None:
    """Run one genuine TP shard and persist each completed evidence boundary."""
    # The parent sets the plugin filter before spawning; a child must never
    # import CrossPool's integration, which would change the subject under test.
    assert os.environ["SGLANG_PLUGINS"] == "__gdn_preflight_no_plugins__"
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    path = init_global_config().model_path_of(MODEL)
    status, driver_version = driver.cuDriverGetVersion()
    check_driver(status, "cuDriverGetVersion(gdn preflight)")
    report: dict[str, object] = {
        "tp_size": tp_size,
        "rank": rank,
        "device": torch.cuda.get_device_name(rank),
        "sglang_version": importlib.metadata.version("sglang"),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_driver_version": driver_version,
        "checkpoint_config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        "checkpoint_index_sha256": hashlib.sha256((path / "model.safetensors.index.json").read_bytes()).hexdigest(),
    }
    report_path = output / f"rank-{rank}.json"
    try:
        report["configuration"] = configuration_evidence(path, tp_size)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        arguments = make_args(path, tp_size)
        publish(arguments, role="scheduler")
        model = ModelConfig.from_server_args(arguments)
        init_distributed_environment(
            world_size=tp_size,
            rank=rank,
            distributed_init_method=(output / "rendezvous").resolve().as_uri(),
            local_rank=rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            attention_data_parallel_size=1,
            attention_context_model_parallel_size=1,
            moe_data_model_parallel_size=1,
        )
        cache_params = mambaish_config(model).mamba2_cache_params
        assert isinstance(cache_params, Mamba2CacheParams)
        assert get_parallel().attn_tp_size == tp_size
        report["state_payload_bytes_per_slot"] = cache_params.mamba_cache_per_req
        report["storage"] = records = []
        for granularity in ("minimum", "recommended"):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            with (
                patch.object(
                    HybridReqToTokenPool, "mamba_pool_cls", partial(PrototypeMambaPool, granularity_kind=granularity)
                ),
                patch.object(MambaPool, "__init__", side_effect=AssertionError("native complete state allocation")),
            ):
                pool = hybrid_pool(cache_params, f"cuda:{rank}", 127)
            vmm = pool.mamba_pool
            assert isinstance(vmm, PrototypeMambaPool)
            construction_allocator_bytes = torch.cuda.memory_allocated() - before
            assert construction_allocator_bytes < 1 << 20
            record: dict[str, object] = {
                "granularity_kind": granularity,
                "granularities": vmm.granularities,
                "planes": [plane.describe() for plane in vmm.planes],
                "native_complete_constructor_called": False,
                "construction_torch_allocator_bytes": construction_allocator_bytes,
            }
            records.append(record)
            try:
                native = hybrid_pool(cache_params, f"cuda:{rank}", 3)
                record["kernels"] = kernel_evidence(pool, native, model, arguments, cache_params)
                full = MHATokenToKVPool(
                    size=1024,
                    page_size=1,
                    dtype=torch.bfloat16,
                    head_num=4 // tp_size,
                    head_dim=256,
                    layer_num=8,
                    device=f"cuda:{rank}",
                    enable_memory_saver=False,
                )
                allocator = TokenToKVPoolAllocator(
                    size=1024, dtype=torch.bfloat16, device=f"cuda:{rank}", kvcache=full, need_sort=False
                )
                cache = create_tree_cache(
                    TreeCacheBuildContext(
                        server_args=arguments,
                        params=CacheInitParams(False, pool, allocator, 1),
                        is_hybrid_swa=False,
                        is_hybrid_ssm=True,
                        enable_hierarchical_cache=False,
                        disable_radix_cache=False,
                        effective_chunked_prefill_size=32,
                        tp_worker=None,
                        model_config=model,
                        tp_size=tp_size,
                        tp_rank=rank,
                        tp_group=None,
                    )
                )
                assert isinstance(cache, UnifiedRadixCache)
                assert isinstance(cache.components[ComponentType.MAMBA], MambaComponent)
                record["default_cache"] = {
                    "type": type(cache).__name__,
                    "components": [x.name for x in cache.components],
                }
            finally:
                vmm.close()
                del pool
                gc.collect()
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if torch.distributed.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()


def model_batch(runner: ModelRunner, *, decode: bool, step: int = 0) -> ForwardBatch:
    assert runner.token_to_kv_pool_allocator is not None
    assert runner.req_to_token_pool is not None
    device = f"cuda:{runner.gpu_id}"
    batch = make_batch(device, (1, 1) if decode else (16, 16), (16 + step, 16 + step) if decode else (0, 0))
    batch.forward_mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
    batch.capture_hidden_mode = CaptureHiddenMode.NULL
    batch.input_ids = torch.tensor([100, 200] if decode else list(range(100, 132)), dtype=torch.int64, device=device)
    batch.positions = (
        torch.full((2,), 16 + step, dtype=torch.int64, device=device)
        if decode
        else torch.arange(16, dtype=torch.int64, device=device).repeat(2)
    )
    batch.seq_lens_cpu = batch.seq_lens.cpu()
    batch.orig_seq_lens = batch.seq_lens.clone()
    batch.global_num_token_non_padded_cpu = batch.input_ids.numel()
    batch.extend_logprob_start_lens_cpu = [0, 0]
    locations = runner.token_to_kv_pool_allocator.alloc(batch.input_ids.numel())
    assert locations is not None
    batch.out_cache_loc = locations
    if decode:
        runner.req_to_token_pool.req_to_token[:2, 16 + step] = locations
    else:
        runner.req_to_token_pool.req_to_token[:2, :16] = locations.view(2, 16)
    return batch


def run_model_rank(rank: int, tp_size: int, output: Path, port: int, mode: str) -> None:
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    path = init_global_config().model_path_of(MODEL)
    decode, prefill = {
        "eager": ("disabled", "disabled"),
        "decode-full": ("full", "disabled"),
        "combined": ("full", "breakable"),
    }[mode]
    arguments = make_args(path, tp_size, decode, prefill)
    publish(arguments, role="scheduler")

    model = ModelConfig.from_server_args(arguments)
    report: dict[str, object] = {"tp_size": tp_size, "rank": rank, "mode": mode}
    pool: PrototypeMambaPool | None = None
    native_pool_guard = patch.object(
        MambaPool, "__init__", side_effect=AssertionError("native complete state allocation")
    )
    try:
        runner = ModelRunner(
            model_config=model,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=rank,
            ps=ParallelState.trivial(
                tp_rank=rank, tp_size=tp_size, attn_tp_rank=rank, attn_tp_size=tp_size, gpu_id=rank
            ),
            nccl_port=port,
            server_args=arguments,
        )
        assert isinstance(runner.model, Qwen3_5ForConditionalGeneration)
        assert runner.model.language_model_only and runner.model.visual is None
        layers = runner.model.model.layers
        assert isinstance(layers, torch.nn.ModuleList)
        assert len(layers) == 32
        report["loading"] = {
            "class": type(runner.model).__name__,
            "visual_is_none": True,
            "decoder_layers": len(layers),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size() for parameter in runner.model.parameters()
            ),
        }
        (output / f"model-rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        native_pool_guard.start()
        with patch.object(
            HybridReqToTokenPool,
            "mamba_pool_cls",
            partial(PrototypeMambaPool, granularity_kind="recommended"),
        ):
            runner.alloc_memory_pool()
        assert isinstance(runner.req_to_token_pool, HybridReqToTokenPool)
        candidate = runner.req_to_token_pool.mamba_pool
        assert isinstance(candidate, PrototypeMambaPool)
        pool = candidate
        pool.map_slots((1, 2, 3))
        runner.init_attention_backends()
        runner.init_cuda_graphs()
        report["graph_capture_completed"] = True
        captured = {}
        if decode == "full":
            assert isinstance(runner.decode_cuda_graph_runner, DecodeCudaGraphRunner)
            assert isinstance(runner.decode_cuda_graph_runner.backend, FullCudaGraphBackend)
            backend = runner.decode_cuda_graph_runner.backend
            assert backend._graphs
            captured["decode"] = {"backend": type(backend).__name__, "shapes": len(backend._graphs)}
        if prefill == "breakable":
            assert isinstance(runner.prefill_cuda_graph_runner, PrefillCudaGraphRunner)
            assert isinstance(runner.prefill_cuda_graph_runner.backend, BreakableCudaGraphBackend)
            backend = runner.prefill_cuda_graph_runner.backend
            assert not backend._debug_eager and backend._graphs
            captured["prefill"] = {"backend": type(backend).__name__, "shapes": len(backend._graphs)}
        report["captured_graphs"] = captured
        runner.req_to_token_pool.req_index_to_mamba_index_mapping[:2] = torch.tensor(
            [1, 2], dtype=torch.int32, device=f"cuda:{rank}"
        )
        pool.clear_slots(torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=f"cuda:{rank}"))
        results = []
        for step in range(4):
            batch = model_batch(runner, decode=step > 0, step=max(0, step - 1))
            result = runner.forward(batch)
            torch.cuda.synchronize()
            assert isinstance(result.logits_output, LogitsProcessorOutput)
            logits = result.logits_output.next_token_logits
            assert logits is not None and bool(torch.isfinite(logits).all())
            expected_graph = prefill == "breakable" if step == 0 else decode == "full"
            assert result.can_run_graph == expected_graph
            torch.save(logits.detach().cpu(), output / f"logits-rank-{rank}-step-{step}.pt")
            results.append({"step": step, "can_run_graph": result.can_run_graph, "argmax": logits.argmax(-1).tolist()})
        report["forwards"] = results
        report["planes"] = [plane.describe() for plane in pool.planes]
        del result, logits, batch, runner
        gc.collect()
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        native_pool_guard.stop()
        if pool is not None:
            pool.close()
        (output / f"model-rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if torch.distributed.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()


def run_experiment(tp_size: int, output: Path, *, model_loading: bool = False) -> None:
    """Start fresh ranks with plugins isolated before any SGLang process bootstrap."""

    os.environ["SGLANG_PLUGINS"] = "__gdn_preflight_no_plugins__"
    output.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(run_rank, args=(tp_size, output), nprocs=tp_size, join=True)
    if model_loading:
        run_model_experiment(tp_size, output)


def run_model_experiment(tp_size: int, output: Path) -> None:
    """Run each official-model graph combination in an independent TP world."""

    os.environ["SGLANG_PLUGINS"] = "__gdn_preflight_no_plugins__"
    for mode in ("eager", "decode-full", "combined"):
        directory = output / mode
        directory.mkdir(parents=True, exist_ok=True)
        reservation = TcpEndpointReservation.reserve(
            "127.0.0.1", port_space=TcpPortSpace.local(), deadline=time.monotonic() + 30
        )
        try:
            port = reservation.port
            reservation.release_for_spawn()
            torch.multiprocessing.spawn(
                run_model_rank, args=(tp_size, directory, port, mode), nprocs=tp_size, join=True
            )
        finally:
            reservation.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-loading", action="store_true")
    arguments = parser.parse_args()
    run_experiment(arguments.tp_size, arguments.output, model_loading=arguments.model_loading)
    reports = [json.loads(path.read_text()) for path in sorted(arguments.output.glob("rank-*.json"))]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

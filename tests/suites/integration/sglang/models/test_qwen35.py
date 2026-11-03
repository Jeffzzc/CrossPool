from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from sglang.srt.models import qwen2_moe, qwen3_5
from tests.suites.integration.sglang.ecs.gdn_preflight import MODEL, make_args

import xtest
from xpool.config import init_global_config
from xpool.integrations.sglang.models.qwen3_5 import XpoolQwen35MLP, around_construct, around_load_weights
from xtest.harness.native.case import run_native_case


def construct_official_text_wrapper(workdir: Path) -> None:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.runtime_context import publish

    path = init_global_config().model_path_of(MODEL)
    arguments = make_args(path, 1)
    publish(arguments, role="scheduler")
    config = ModelConfig.from_server_args(arguments)
    torch.cuda.set_device(0)
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=(workdir / "rendezvous").as_uri(), local_rank=0, backend="nccl"
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    initialize_dp_attention(arguments, config)
    original = qwen3_5.Qwen2MoeMLP
    try:
        model = object.__new__(qwen3_5.Qwen3_5ForConditionalGeneration)
        with torch.device("meta"):
            around_construct(qwen3_5.Qwen3_5ForConditionalGeneration.__init__, model, config.hf_config)
        assert model.visual is None
        assert len(model.model.layers) == 32
        for layer in model.model.layers:
            assert isinstance(layer.mlp, XpoolQwen35MLP)
            assert not tuple(layer.mlp.parameters())
            assert not layer.layer_communicator.allow_reduce_scatter
        assert qwen3_5.Qwen2MoeMLP is qwen2_moe.Qwen2MoeMLP is original
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@xtest.requirements(device_count=1, requires_config=True, model_ids=(MODEL,))
def test_official_text_wrapper_constructs_only_parameter_free_ffn(tmp_path: Path) -> None:
    run_native_case(construct_official_text_wrapper, tmp_path, workdir=tmp_path / "child", timeout_seconds=120)


def test_wrapper_shim_owns_no_projection_weights_and_requires_main_prefix() -> None:
    shim = XpoolQwen35MLP(4096, 12288, "silu", prefix="model.language_model.layers.3.mlp")
    assert shim.layer_id == 3 and not tuple(shim.parameters())
    with pytest.raises(RuntimeError, match="main-wrapper"):
        XpoolQwen35MLP(4096, 12288, "silu", prefix="model.layers.3.mlp")


def test_constructor_scope_preserves_other_mlp_aliases_and_restores_on_failure() -> None:
    original = qwen3_5.Qwen2MoeMLP
    fusion = qwen3_5._maybe_enable_silu_fp4_quant_fusion
    model = object.__new__(qwen3_5.Qwen3_5ForConditionalGeneration)

    def construct(candidate: qwen3_5.Qwen3_5ForConditionalGeneration) -> None:
        assert qwen3_5.Qwen2MoeMLP is XpoolQwen35MLP
        assert qwen2_moe.Qwen2MoeMLP is original
        raise ValueError("construction failure")

    with pytest.raises(ValueError, match="construction failure"):
        around_construct(construct, model)
    assert qwen3_5.Qwen2MoeMLP is original
    assert qwen3_5._maybe_enable_silu_fp4_quant_fusion is fusion


def test_original_weight_stream_filters_only_main_ffn_and_mtp() -> None:
    names = (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "lm_head.weight",
    )
    model = object.__new__(qwen3_5.Qwen3_5ForConditionalGeneration)

    def load(
        candidate: qwen3_5.Qwen3_5ForConditionalGeneration, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        return {name for name, tensor in weights}

    assert around_load_weights(load, model, ((name, torch.empty(0)) for name in names)) == set(names[2:])

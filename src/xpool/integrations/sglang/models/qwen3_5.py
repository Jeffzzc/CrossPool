"""Official Qwen3.5-9B text-only FFN boundary and launch policy."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_exec, get_forward, get_memory, get_parallel, get_schedule

from xpool.integrations.sglang.adapter import SglangInstanceRankRuntime, SglangShimAdapter, model_runner_architectures
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.shim import FfnShimModule, ShimUnavailableError
from xpool.model import ModelId
from xpool.native.ffn import LayerKind

LAYER_PREFIX = re.compile(r"^model\.language_model\.layers\.(?P<layer_id>\d+)\.mlp$")
FFN_WEIGHT = re.compile(r"^model\.language_model\.layers\.\d+\.mlp(?:\.|$)")


class XpoolQwen35MLP(FfnShimModule, Qwen2MoeMLP):
    """Parameter-free Dense FFN with the pinned Qwen2MoeMLP call surface."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ) -> None:
        """Derive the main decoder layer without allocating local projections."""
        match = LAYER_PREFIX.fullmatch(prefix)
        if hidden_act != "silu" or quant_config is not None or match is None:
            raise ShimUnavailableError("xpool Qwen3.5 requires an unquantized SiLU main-wrapper FFN")
        FfnShimModule.__init__(
            self, layer_id=int(match.group("layer_id")), hidden_size=hidden_size, layer_kind=LayerKind.DENSE
        )


def around_construct(
    original_fn: Callable[..., None], model: qwen3_5.Qwen3_5ForConditionalGeneration, *args: object, **kwargs: object
) -> None:
    """Limit the MLP substitution to this model's synchronous construction.

    HookRegistry propagates class replacements to every imported alias. A
    module-local construction scope preserves other families' Qwen2MoeMLP.
    The admitted BF16 path has no local FP4 projection fusion to configure.
    """
    original_mlp = qwen3_5.Qwen2MoeMLP
    original_fusion = qwen3_5._maybe_enable_silu_fp4_quant_fusion
    setattr(qwen3_5, "Qwen2MoeMLP", XpoolQwen35MLP)
    setattr(qwen3_5, "_maybe_enable_silu_fp4_quant_fusion", lambda mlp: None)
    try:
        original_fn(model, *args, **kwargs)
    finally:
        setattr(qwen3_5, "Qwen2MoeMLP", original_mlp)
        setattr(qwen3_5, "_maybe_enable_silu_fp4_quant_fusion", original_fusion)
    layers = model.model.layers
    if not isinstance(layers, torch.nn.ModuleList):
        raise ShimUnavailableError("xpool Qwen3.5 requires original decoder layers")
    for layer in layers:
        if not isinstance(layer, qwen3_5.Qwen3_5LinearDecoderLayer | qwen3_5.Qwen3_5AttentionDecoderLayer):
            raise ShimUnavailableError("xpool Qwen3.5 encountered an unsupported decoder layer")
        layer.layer_communicator.allow_reduce_scatter = False


def around_load_weights(
    original_fn: Callable[..., set[str]],
    model: qwen3_5.Qwen3_5ForConditionalGeneration,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Filter original checkpoint keys lazily before upstream namespace mapping."""
    return original_fn(
        model,
        ((name, tensor) for name, tensor in weights if FFN_WEIGHT.match(name) is None and not name.startswith("mtp.")),
    )


def linear_decoder_forward(
    self: qwen3_5.Qwen3_5LinearDecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep upstream GDN/norm order while handing a complete row buffer to FFN."""
    batch = kwargs["forward_batch"]
    if not isinstance(batch, ForwardBatch):
        raise ShimUnavailableError("xpool Qwen3.5 linear decoder requires ForwardBatch")
    hidden_states, residual = self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
        hidden_states, residual, batch, captured_last_layer_outputs=kwargs.get("captured_last_layer_outputs")
    )
    if not batch.forward_mode.is_idle() and hidden_states.shape[0] > 0:
        hidden_states = self.linear_attn(hidden_states, batch)
    hidden_states, residual = self.layer_communicator.prepare_mlp(hidden_states, residual, batch)
    with get_forward().scoped(fuse_mlp_allreduce=False, mlp_reduce_scatter=False):
        hidden_states = self.mlp(hidden_states, forward_batch=batch)
    return self.layer_communicator.postprocess_layer(hidden_states, residual, batch)


def attention_decoder_forward(
    self: qwen3_5.Qwen3_5AttentionDecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    forward_batch: ForwardBatch,
    captured_last_layer_outputs: list[torch.Tensor] | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep gated Full Attention execution and supply batch metadata to the shim."""
    hidden_states, residual = self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
        hidden_states, residual, forward_batch, captured_last_layer_outputs=captured_last_layer_outputs
    )
    if not forward_batch.forward_mode.is_idle() and hidden_states.shape[0] > 0:
        hidden_states = self.self_attention(
            positions=positions, hidden_states=hidden_states, forward_batch=forward_batch
        )
    hidden_states, residual = self.layer_communicator.prepare_mlp(hidden_states, residual, forward_batch)
    with get_forward().scoped(fuse_mlp_allreduce=False, mlp_reduce_scatter=False):
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
    return self.layer_communicator.postprocess_layer(hidden_states, residual, forward_batch)


class Qwen35ShimAdapter(SglangShimAdapter):
    """Admit the official BF16 text-only, no_buffer TP1/TP2 execution boundary."""

    name = "qwen3_5"

    def hooks(self) -> tuple[SglangHook, ...]:
        """Install only model-owned constructor, loader and decoder seams."""
        return (
            SglangHook(
                "sglang.srt.models.qwen3_5.Qwen3_5ForConditionalGeneration.__init__", around_construct, HookType.AROUND
            ),
            SglangHook(
                "sglang.srt.models.qwen3_5.Qwen3_5ForConditionalGeneration.load_weights",
                around_load_weights,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.models.qwen3_5.Qwen3_5LinearDecoderLayer.forward", linear_decoder_forward, HookType.REPLACE
            ),
            SglangHook(
                "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.forward",
                attention_decoder_forward,
                HookType.REPLACE,
            ),
        )

    def matches(self, model_runner: ModelRunner) -> bool:
        """Match the original official wrapper architecture, retaining its identity."""
        return "Qwen3_5ForConditionalGeneration" in model_runner_architectures(model_runner)

    def validate_before_load(self, model_runner: ModelRunner) -> None:
        """Fail before state/weight construction for unqualified execution modes."""
        super().validate_before_load(model_runner)
        model = model_runner.model_config
        parallel, execution, memory, schedule = get_parallel(), get_exec(), get_memory(), get_schedule()
        mamba = execution.mamba
        if (
            not model.is_lm_only
            or model.is_multimodal
            or model.dtype != torch.bfloat16
            or parallel.attn_tp_size not in (1, 2)
            or parallel.dp_size != 1
            or not schedule.disable_overlap_schedule
            or schedule.page_size != 1
            or mamba.enable_mamba_extra_buffer
            or mamba.enable_linear_replayssm
            or mamba.enable_linear_replayssm_spec
            or mamba.enable_int8_mamba_checkpoint
            or memory.enable_session_radix_cache
            or mamba.linear_attn_prefill_backend not in (None, "triton")
            or mamba.linear_attn_decode_backend not in (None, "triton")
            or mamba.linear_attn_backend != "triton"
        ):
            raise ShimUnavailableError("xpool Qwen3.5 requires qualified BF16 text-only TP1/TP2 no_buffer execution")
        params = model.hf_text_config.mamba2_cache_params
        if params.dtype.conv != torch.bfloat16 or params.dtype.temporal != torch.float32:
            raise ShimUnavailableError("xpool Qwen3.5 requires BF16 Conv and FP32 Temporal state")

    def bind_runtime(self, model_runner: ModelRunner) -> None:
        """Keep the first supported checkpoint identity separate from family discovery."""
        super().bind_runtime(model_runner)
        if SglangInstanceRankRuntime.require(model_runner).binding.model_id != ModelId("Qwen/Qwen3.5-9B"):
            raise ShimUnavailableError("xpool Qwen3.5 currently admits only Qwen/Qwen3.5-9B")

    def validate_after_load(self, model_runner: ModelRunner) -> None:
        """Require every Dense shim and verify the complete-row collective boundary."""
        model = model_runner.model
        if not isinstance(model, qwen3_5.Qwen3_5ForConditionalGeneration) or model.visual is not None:
            raise ShimUnavailableError("xpool Qwen3.5 requires the original text-only wrapper")
        shims = self.require_ffn_shims(
            model,
            expected_layer_kinds=(LayerKind.DENSE,) * model.config.num_hidden_layers,
            allowed_shim_types=(XpoolQwen35MLP,),
        )
        self.require_full_mlp_boundaries(model, shims, allow_reduce_scatter=False)

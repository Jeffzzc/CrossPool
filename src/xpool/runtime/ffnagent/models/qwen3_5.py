"""Official Qwen3.5-9B text decoder FFN compilation."""

from __future__ import annotations

from xpool import ffn
from xpool.model import ModelId
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import architecture


class Qwen35Adapter(architecture.FfnModelAdapter):
    """Compile the admitted unquantized BF16 wrapper checkpoint namespace."""

    architecture_name = "Qwen3_5ForConditionalGeneration"
    checkpoint_decoder_prefix = "model.language_model.layers"

    @classmethod
    def compile(cls, *, model_id: ModelId, model_config: architecture.FfnSourceConfig) -> ffn.FfnModelSpec:
        """Preserve source keys while reusing ordinary gated Dense execution."""
        if model_id != ModelId("Qwen/Qwen3.5-9B") or model_config.get("model_type", str) != "qwen3_5":
            raise ValueError("Qwen3.5 admission requires the official Qwen/Qwen3.5-9B wrapper")
        text = architecture.FfnSourceConfig(model_config.get("text_config", dict))
        text.validate_family_profile(model_type="qwen3_5_text", dtype_field="dtype")
        if text.get("dtype", str) != "bfloat16":
            raise ValueError("Qwen3.5 text FFN requires bfloat16")
        for config in (model_config, text):
            if config.optional("quantization_config", dict, allow_none=True) is not None:
                raise ValueError("Qwen3.5 text FFN requires unquantized weights")
        layers = text.get("num_hidden_layers", int, ge=1)
        kinds = text.get("layer_types", list)
        if kinds != ["full_attention" if index % 4 == 3 else "linear_attention" for index in range(layers)]:
            raise ValueError("Qwen3.5 text FFN requires the qualified GDN/GDN/GDN/Full layer pattern")
        if text.optional("mlp_only_layers", list) not in (None, []):
            raise ValueError("Qwen3.5 text FFN requires an MLP in every main decoder layer")
        hidden = text.get("hidden_size", int, ge=1)
        intermediate = text.get("intermediate_size", int, ge=1)
        return ffn.FfnModelSpec(
            model_id=model_id,
            hidden_size=hidden,
            activation=ffn.SiluActivation(),
            layers=tuple(
                ffn.DenseFfnSpec(
                    kind=LayerKind.DENSE,
                    layer_id=index,
                    intermediate_size=intermediate,
                    projections=text.gated_projections(f"{cls.checkpoint_decoder_prefix}.{index}.mlp"),
                )
                for index in range(layers)
            ),
        )

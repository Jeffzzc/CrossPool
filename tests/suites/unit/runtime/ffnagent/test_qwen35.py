from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpool.ffn import DenseFfnSpec
from xpool.model import ModelId
from xpool.runtime.ffnagent import architecture

MODEL = ModelId("Qwen/Qwen3.5-9B")


def checkpoint(path: Path, *, dtype: str = "bfloat16", extra: tuple[str, ...] = ()) -> None:
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "dtype": dtype,
            "hidden_act": "silu",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        },
    }
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    shard = "model-00001-of-00001.safetensors"
    (path / shard).touch()
    keys = [
        f"model.language_model.layers.{layer}.mlp.{projection}_proj.weight"
        for layer in range(4)
        for projection in ("gate", "up", "down")
    ]
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys((*keys, *extra), shard)})
    )


def test_original_wrapper_namespace_compiles_all_dense_layers(tmp_path: Path) -> None:
    checkpoint(tmp_path / "model", extra=("mtp.layers.0.mlp.gate_proj.weight", "model.visual.weight"))
    spec = architecture.load(model_id=MODEL, model_path=tmp_path / "model")
    assert spec.hidden_size == 16 and len(spec.layers) == 4
    for layer in spec.layers:
        assert isinstance(layer, DenseFfnSpec)
        assert layer.tensor_keys() == tuple(
            f"model.language_model.layers.{layer.layer_id}.mlp.{projection}_proj.weight"
            for projection in ("gate", "up", "down")
        )


@pytest.mark.parametrize(
    "extra",
    (
        "model.language_model.layers.0.mlp.bias",
        "model.language_model.layers.4.mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ),
)
def test_wrapper_rejects_unowned_or_mixed_ffn_weights(tmp_path: Path, extra: str) -> None:
    checkpoint(tmp_path / "model", extra=(extra,))
    with pytest.raises(RuntimeError, match=r"FFN|namespaces"):
        architecture.load(model_id=MODEL, model_path=tmp_path / "model")


def test_wrapper_rejects_unqualified_payload_dtype(tmp_path: Path) -> None:
    checkpoint(tmp_path / "model", dtype="float16")
    with pytest.raises(RuntimeError, match="bfloat16"):
        architecture.load(model_id=MODEL, model_path=tmp_path / "model")

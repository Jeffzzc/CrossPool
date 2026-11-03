from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tests.suites.integration.sglang.ecs.gdn_preflight import MODEL, run_experiment, run_model_experiment

import xtest
from xkit import ResourceRequirements
from xtest.harness.native.case import run_native_case
from xtest.harness.sglang.serving.alignment import PREFILL_DISTRIBUTION_KL_LIMIT

pytestmark = pytest.mark.timeout(2400)


def requirements(*, tp_size: int) -> ResourceRequirements:
    return ResourceRequirements(tp_size, True, (MODEL,))


@pytest.mark.parametrize("tp_size", (1, 2))
@xtest.requirements(requirements)
def test_layer_major_gdn_vmm_state_and_graph_replay(tp_size: int, tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    run_native_case(run_experiment, tp_size, output, workdir=tmp_path / "child", timeout_seconds=900)
    reports = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(tp_size)]
    for report in reports:
        assert "error" not in report
        assert report["state_payload_bytes_per_slot"] == 51_511_296 // tp_size
        assert len(report["storage"]) == 2
        for storage in report["storage"]:
            assert storage["default_cache"] == {"type": "UnifiedRadixCache", "components": ["FULL", "MAMBA"]}
            assert not storage["native_complete_constructor_called"]
            assert all(plane["per_layer_contiguous"] for plane in storage["planes"])
            assert storage["kernels"]["released_and_remapped_bytes"] > 0


@pytest.mark.parametrize("tp_size", (1, 2))
@xtest.requirements(requirements)
def test_official_text_weights_and_gdn_vmm_phase_runners(tp_size: int, tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    run_native_case(run_model_experiment, tp_size, output, workdir=tmp_path / "child", timeout_seconds=1800)
    alignment = []
    for rank in range(tp_size):
        for mode in ("eager", "decode-full", "combined"):
            report = json.loads((output / mode / f"model-rank-{rank}.json").read_text())
            assert "error" not in report
            assert report["graph_capture_completed"]
            for step in range(4):
                reference = torch.load(output / "eager" / f"logits-rank-{rank}-step-{step}.pt", weights_only=True)
                actual = torch.load(output / mode / f"logits-rank-{rank}-step-{step}.pt", weights_only=True)
                assert reference.shape == actual.shape == (2, 248320)
                kl = torch.nn.functional.kl_div(
                    actual.float().log_softmax(-1),
                    reference.float().log_softmax(-1),
                    reduction="batchmean",
                    log_target=True,
                ).item()
                assert kl <= PREFILL_DISTRIBUTION_KL_LIMIT
                if mode == "decode-full":
                    assert torch.equal(actual.argmax(-1), reference.argmax(-1))
                alignment.append({"rank": rank, "mode": mode, "step": step, "forward_kl": kl})
    (output / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.suites.integration.sglang.ecs.gdn_backing_check import run_backing_check
from tests.suites.integration.sglang.ecs.gdn_preflight import MODEL

import xtest
from xkit import ResourceRequirements
from xtest.harness.native.case import run_native_case


def requirements(*, tp_size: int) -> ResourceRequirements:
    return ResourceRequirements(tp_size, True, (MODEL,))


@pytest.mark.parametrize("tp_size", (1, 2))
@xtest.requirements(requirements)
def test_production_gdn_backing_native_kernels_and_graph_remap(tp_size: int, tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    run_native_case(run_backing_check, tp_size, output, workdir=tmp_path / "child", timeout_seconds=600)
    for rank in range(tp_size):
        result = json.loads((output / f"rank-{rank}.json").read_text())
        assert result["output_error"] == result["state_error"] == 0
        assert result["same_graph_after_remap"]
        assert result["released_gdn_bytes"] >= result["reused_full_bytes"] > 0

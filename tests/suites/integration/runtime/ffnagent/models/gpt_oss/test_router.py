from __future__ import annotations

import os
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
import torch

import xtest
from xkit.child import PythonChildProcess
from xpool.runtime.ffnagent import operators, weights
from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter


def run_workspace_free_router(connection: Connection, seed: int) -> None:
    """Exercise the FfnAgent cuBLAS policy before the child's first CUDA use."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    generator = torch.Generator().manual_seed(seed)
    # Quarter-integer operands make the independent FP32 dot products exact.
    hidden = torch.randint(-4, 5, (512, 2880), generator=generator).bfloat16() * 0.25
    weight = torch.randint(-4, 5, (32, 2880), generator=generator).bfloat16() * 0.25
    bias = torch.randn(32, generator=generator).bfloat16()
    products = hidden.float() @ weight.float().t()
    router = weights.MoeRouterWeights(weight=weight.cuda(), correction_bias=None, bias=bias.cuda())
    hidden = hidden.cuda()

    for rows in (1, 32, 512):
        expected_logits = (
            (products[:rows] + bias.float()).bfloat16() if rows == 1 else products[:rows].bfloat16().add(bias)
        ).cuda()
        workspace = torch.empty(
            GptOssAdapter.router_workspace_bytes(
                payload_dtype=torch.bfloat16,
                payload_row_capacity=rows,
                hidden_size=2880,
                routed_expert_count=32,
                routed_topk=4,
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        ids = torch.empty((rows, 4), dtype=torch.int32, device="cuda")
        probabilities = torch.empty((rows, 4), dtype=torch.float32, device="cuda")
        expected_ids = torch.empty_like(ids)
        expected_probabilities = torch.empty_like(probabilities)
        operators.compute_softmax_topk(
            logits=expected_logits,
            routed_ids=expected_ids,
            routed_weights=expected_probabilities,
            renormalize=True,
        )

        def launch() -> None:
            GptOssAdapter.compute_routed_topk(
                hidden_states=hidden[:rows],
                router_weights=router,
                workspace=workspace,
                routed_ids=ids,
                routed_weights=probabilities,
                renormalize=True,
            )

        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(probabilities, expected_probabilities, rtol=1e-6, atol=1e-7)
    connection.send(True)


@pytest.mark.timeout(180)
@xtest.requirements(device_count=1)
def test_gpt_oss_router_preserves_bias_rounding_without_cublas_workspace(tmp_path: Path) -> None:
    """Match the BF16 Router boundary under real workspace-free graph replay."""

    child = PythonChildProcess("workspace-free-router", run_workspace_free_router, 83, log_path=tmp_path / "child.log")
    try:
        child.start()
        assert child.receive(bool, timeout_seconds=120)
        child.wait(timeout_seconds=30)
    finally:
        if child.process.is_alive():
            PythonChildProcess.terminate_all((child,))
        child.close()

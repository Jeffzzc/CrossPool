"""Explicit qualification; serving coverage is required before declaring support."""

from __future__ import annotations

from pathlib import Path

from tests import TEST_CATALOG_PATH

import xtest
from xkit.deployment import resolve_deployment_path
from xpool.model import ModelId
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang import numerical
from xtest.harness.sglang.catalog import E2eFfnInputMatrix, E2eFfnNumericalCase

pytest_plugins = ("xtest.harness.support.config",)
MODEL = ModelId("Qwen/Qwen3.5-9B")
NUMERICAL_CASES: tuple[E2eFfnNumericalCase, ...] = (
    E2eFfnNumericalCase(
        description="Official BF16 text decoder FFN against isolated original SGLang.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn2-lanes1"),
        model_id=MODEL,
        language_model_only=True,
        layer_ids=(0, 31),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1, 32, 4096)),
        estimated_duration_seconds=600,
        timeout_seconds=1800,
    ),
)


@xtest.parameterize("case", tuple(numerical.case_parameter(case) for case in NUMERICAL_CASES))
@xtest.requirements(numerical.requirements_of)
def test_ffn_numerical(
    case: E2eFfnNumericalCase, e2e_base_config: ResolvedConfig, tmp_path: Path, task_artifact_dir: Path | None
) -> None:
    numerical.run_numerical_case(case, e2e_base_config, tmp_path, task_artifact_dir)

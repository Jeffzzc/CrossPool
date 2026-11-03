# Qwen3.5-9B implementation evidence

Baseline: `a3467caa261fd601b76492bf20d45f226cec0fce`, SGLang 0.5.20,
Transformers 5.12.1. Development branch: `feat/qwen35-9b-gdn-ecs`.
All new experiments restrict physical devices to 0 through 4.

## Phase 1: official text FFN boundary

The source prototypes `gdn_vmm_prototype.py`, `gdn_preflight.py` and
`test_gdn_preflight.py` are present under `tests/suites/integration/sglang/ecs/`.
Their source, rather than only the recorded JSON, remains the preflight basis.

The official wrapper compiles its nested text configuration and original
`model.language_model.layers` checkpoint namespace into existing gated Dense
FFN execution. Other families retain their default namespace. Upstream
checkpoint mapping handles the retained attention/GDN/embedding/head weights.
The wrapper constructor substitutes only its own temporary MLP binding and
restores it even on failure. Decoder handoffs use complete rows with MLP
collective fusion and reduce-scatter disabled.

The launch policy admits BF16 text-only TP1/TP2, DP1, no speculative execution,
no_buffer, disabled overlap and page size 1. Before the GDN ECS factory is
installed, the reservation hook explicitly rejects Mamba models before the
native static state pool can be constructed. This phase enables FFN testing,
not production hybrid serving. The support table remains unsupported.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 UV_ENV_FILE=.env uv run --no-sync xtest run \
  tests/suites/unit/runtime/ffnagent/test_qwen35.py \
  tests/suites/unit/runtime/ffnagent/test_architecture.py \
  tests/suites/unit/runtime/ffnagent/test_checkpoint.py \
  tests/suites/integration/sglang/models \
  tests/suites/integration/sglang/ecs/test_hooks.py --strict-requirements
CUDA_VISIBLE_DEVICES=0,1,2,3,4 UV_ENV_FILE=.env uv run --no-sync xtest run \
  --suite Qwen/Qwen3.5-9B --strict-requirements
```

Focused strict run `20261103-085006-3109572-493267870044098`: 86 passed,
0 skipped, 0 failed (26 unit, 59 CPU integration and 1 device integration).
Focused Ruff and type checks passed for the changed runtime and harness code.

Model numerical run `20261103-084811-3104770-493152932615409`: 1 passed,
0 skipped, 0 failed; original SGLang Reference versus actual CrossPool FFN TP2,
layers 0/31 and row counts 1/32/4096. All six comparisons passed existing
thresholds; maximum normalized RMSE 0.004671, maximum scaled error 0.023985,
no outliers. Both 4096-row comparisons were exact. The original official
checkpoint was loaded with the JSON `language_model_only` override.

Actual official wrapper construction run
`20261103-084833-3105706-493174754962456`: 4 passed, 0 skipped; all 32 MLPs
parameter-free, visual module absent, communicator boundaries retained, other
families' MLP bindings preserved, constructor-failure restoration verified.

Uncovered at this phase: attention TP2 forward collective correctness,
production GDN backing, request/cache state ownership, HTTP serving,
elasticity and complete model qualification. Earlier preflight model/Graph
results validate the prototype, not this production adapter.

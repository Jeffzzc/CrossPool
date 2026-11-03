# Qwen3.5-9B GDN storage preflight

## Boundary and verdict

Layer-major Conv/Temporal VMM views with dense per-layer slot rows work with the
pinned SGLang GDN implementation on the tested A100 devices. The experiment
reuses SGLang's GDN backend, kernels, clear/copy operations, request pool and
phase runners. It changes no production Scheduler, model adapter, daemon ECS
protocol or GDN mathematics.

This is storage and execution compatibility evidence. It does not qualify
CrossPool model support, concurrent quota admission, prefix COW transactions,
cache-driven reclamation or a complete serving endpoint.

The public `--language-model-only` assumption in the initial plan was incorrect
for this dependency version. The existing model-config override is the tested
text-only loading route; see the launch section below.

## Reproduction and evidence

The baseline branch is `a3467ca`, using SGLang 0.5.20, PyTorch 2.13.0+cu130 and
the official local `Qwen/Qwen3.5-9B` checkpoint resolved through `xpool.config`.
The Device is NVIDIA A100-SXM4-40GB. Per-rank JSON records include the CUDA
driver/runtime versions and checkpoint config/index SHA256 values.

The standalone experiment is split into
[storage](../../../tests/suites/integration/sglang/ecs/gdn_vmm_prototype.py) and
[bootstrap, kernels and phase runners](../../../tests/suites/integration/sglang/ecs/gdn_preflight.py).
[Integration tests](../../../tests/suites/integration/sglang/ecs/test_gdn_preflight.py)
declare checkpoint, configuration and TP1/TP2 Device requirements and run the
experiment in fresh supervised processes.

```bash
UV_ENV_FILE=.env uv run --no-sync xtest run \
  tests/suites/integration/sglang/ecs/test_gdn_preflight.py \
  --strict-requirements
```

For an explicitly selected idle Device set, the standalone command is:

```bash
UV_ENV_FILE=.env uv run --no-sync python -m \
  tests.suites.integration.sglang.ecs.gdn_preflight \
  --tp-size 2 --output /tmp/qwen35-gdn-preflight --model-loading
```

The CLI does not acquire an xtest Device lease; use the integration command for
shared test execution. It writes per-rank JSON, per-mode model JSON and logits.
The integration test additionally computes the retained logit alignment.
The compact retained record is [preflight-results.json](preflight-results.json);
the canonical run identity and source digests are recorded there.
The final focused strict run is `20261102-223801-2586703-456542941386787`:
four tests passed, none failed or skipped. Ruff lint/format, ty and document
link/whitespace checks cover the added experiment. This is not a complete
repository-suite or CrossPool model-support verdict.

## Official checkpoint and launch modes

The original `config.json` declares `Qwen3_5ForConditionalGeneration` and nested
`qwen3_5_text`. Its identity and checkpoint keys remain unchanged.

Calling `ServerArgs.resolve_once()` with `language_model_only=True` rejects
this architecture in `arg_groups/model_hook.py:handle_language_model_only()`.
The public flag allowlist contains MuseGlimmer/Cosmos architectures, not Qwen3.5.
Do not bypass that check by patching the allowlist.

The tested existing route is:

```text
--json-model-override-args '{"language_model_only":true}'
--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule
--page-size 1 --attention-backend triton
--linear-attn-prefill-backend triton --linear-attn-decode-backend triton
```

Resolution produces `is_lm_only=True`, `is_multimodal=False`. The real model
loader reads the official shards, creates `Qwen3_5ForConditionalGeneration`,
retains all 32 text decoder layers and sets `visual=None`. Text parameter bytes
are 17,907,608,064 at TP1 and 8,954,077,440 per rank at TP2. The upstream loader
reports absent visual parameters while skipping the checkpoint's vision keys;
that diagnostic is not a missing text-weight verdict.

All three requested combinations resolve and execute with `no_buffer` and
disabled overlap:

| Mode | Prefill execution | Decode execution |
| --- | --- | --- |
| Eager | Eager | Eager |
| Decode Full | Eager | Full CUDA Graph |
| Combined | Breakable CUDA Graph | Full CUDA Graph |

The model experiment captures decode batch sizes 1/2 and Breakable prefill
token buckets 16/32, then executes two requests with 16 input tokens each and
three fixed-input Decode steps. It checks the concrete graph backend, nonempty
capture inventory, Breakable debug-eager being disabled, and each forward's
actual `can_run_graph` result. Eager-to-graph full-distribution forward KL is
zero in the measured runs; Decode Full argmax IDs also match Eager. These are
short fixed-input model forwards, not autoregressive HTTP qualification.

Default cache selection is tested through `mem_cache.registry.create_tree_cache`
with no custom backend: it constructs `UnifiedRadixCache` with `FULL` and
`MAMBA`, whose MAMBA owner is the actual `MambaComponent`. No special cache class
is selected to make the prototype pass. This tests factory composition, not
request-level cache insertion, locking or eviction transactions.

## State geometry and physical granularity

The storage experiment reserves 127 usable slots plus dummy slot 0. Dimensions
come from the official config's upstream `mamba2_cache_params` after real TP
group initialization. Both TP2 ranks run the same checks on their local shards.

| TP | Plane | Complete tensor shape | Dtype | Element strides |
| --- | --- | --- | --- | --- |
| 1 | Conv | `[24,128,8192,3]` | BF16 | `[4194304,24576,3,1]` |
| 1 | Temporal | `[24,128,32,128,128]` | FP32 | `[67633152,524288,16384,128,1]` |
| 2 | Conv | `[24,128,4096,3]` | BF16 | `[3145728,12288,3,1]` |
| 2 | Temporal | `[24,128,16,128,128]` | FP32 | `[34078720,262144,16384,128,1]` |

Each full tensor is deliberately non-contiguous, while every selected layer's
`[slots,...]` view is contiguous. Only the outer layer stride has padding. The
experiment adds one granularity quantum beyond the aligned layer extent to
exercise that distinction even when the extent is already aligned. Production
need not retain that extra test padding.

The official slot payload is 51,511,296 bytes at TP1 and 25,755,648 bytes per rank
at TP2. That is 49.125/24.5625 MiB before physical rounding. Conv and Temporal
keep their distinct dtypes; neither is coerced to the model payload dtype.

For the exact device-local, non-exportable allocation properties used here:

| Quantity | TP1 | TP2 per rank |
| --- | --- | --- |
| CUDA minimum mapping granularity | 2 MiB | 2 MiB |
| CUDA recommended mapping granularity | 2 MiB | 2 MiB |
| Smallest one-layer physical allocation/map | 2 MiB | 2 MiB |
| Coordinated one-plane bundle, all 24 layers | 48 MiB | 48 MiB |
| Conv dummy-slot floor | 48 MiB | 48 MiB |
| Temporal dummy-slot floor | 48 MiB | 48 MiB |
| Joint dummy-slot floor | 96 MiB | 96 MiB |

Both granularity policies are queried and exercised, although their values
coincide on this hardware. These are measured values, not portable constants.
At TP2 a Temporal quantum covers two slots; a slot is therefore not a physical
allocation unit. Conv slots can cross quantum boundaries and cover several
intervals. Future byte pricing must use exact interval unions.

An initial mapping experiment attempted one compound physical handle with a
nonzero allocation offset for each layer's subrange. `cuMemMap` returned
`CUDA_ERROR_NOT_SUPPORTED`. The working layout instead owns one physical
handle per layer interval and always uses offset zero. The logical bundle
still coordinates those 24 handles and charges their summed bytes. Partial
construction and release need per-handle bookkeeping.

## Construction and upstream operations

`PrototypeMambaPool` inherits the upstream pool's state access, clear and copy
methods, but initializes its state using VMM views directly. It never calls
`MambaPool.__init__`. During prototype construction that original constructor
is patched to fail if called; the real-model experiment keeps that guard active
through capture and forward as well.

The storage experiment's Torch allocator increase during hybrid-pool
construction is only 3,072 bytes for request/index/scalar storage. The separate
physical VMM floor is 96 MiB. No native full 128-slot state pool is created and
then copied or retained. A separate four-slot original pool exists only as the
explicit numerical oracle; it is not the original full reservation and never
backs prototype execution. The full-model phase-runner experiment has no
parallel native state-pool oracle.

Only mapped slots are initialized. The whole virtual reservation is never
passed to `zero_()`. The prototype retains normal slot IDs and upstream hybrid
identity translation. It adds no compaction or virtual-to-physical table.

GDN Prefill receives contiguous per-layer Conv and Temporal views, so the
upstream non-contiguous-state gather/scatter branch is unnecessary. Decode
uses the same layer views and upstream packed recurrent kernel. Outer padding
disables the pool-wide fused Conv clear/copy predicate; the upstream indexed
fallback remains exact and exercises the padded layer stride.

## Numerical, indices and remapping checks

For each granularity policy and every GDN layer, the experiment compares:

- variable-length cold Prefill with lengths 17/9;
- continuation Prefill with lengths 3/5 and nonzero prefixes 17/9;
- all-layer Conv/Temporal contents against original pool storage;
- upstream slot copy and clear, including unchanged neighboring slots;
- captured Decode against original eager execution.

Output and state comparisons use exact equality, with maximum observed absolute
error zero. This compares storage substitution under the same upstream math;
it is not an independent proof of GDN mathematics.

The real upstream `_capture_metadata()` and `_replay_metadata()` populate the
static int32 `state_indices_list`. The same captured graph executes three times
for each of slots 1/2, a changed destination with one padded request, and newly
mapped distant slot 96. The test checks pointer stability and the `-1` padded
state sentinel, and verifies dummy slot 0 remains unchanged.

Replay metadata mutates padded request indices in place. Each new logical
batch must restore its real request indices before replay preparation; the
prototype explicitly does so. Reusing the poisoned padded tail as a new live
batch is invalid upstream usage.

After a recorded CUDA Event completes, only distant-slot bundles disjoint from
dummy/live slots are unmapped. Exactly 96 MiB is released per rank, remapped at
the same VA, initialized, and used by the same still-live captured graph.
Outputs and state again match the original pool. This demonstrates real sparse
physical release and reuse, rather than a static state pool with changing
logical counters. The prototype explicitly controls quiescence; it does not
implement production lifetime discovery or quota arbitration.

## Consequences for the implementation plan

Keep the layer-major layout and per-layer contiguous views. Implement per-layer
zero-offset allocation handles, independent Conv/Temporal interval inventories,
and aggregate byte accounting. Preserve upstream indexed clear/copy and graph
indices instead of introducing new kernels.

Use the verified JSON model override for the pinned text-only entry. Runtime
adapter checks must inspect the resolved model capability, not insist on the
rejected raw CLI flag. Keep the distinction between cache composition and
validated cache lifecycle.

The remaining implementation requires production-safe allocation failure
handling, shared quota charging, admission for active/checkpoint slots, COW
source locks, rollback receipts, matched TP decisions and event-ordered
retirement. Extra buffers, overlap, sessions, beam/speculative execution,
multimodal input and long-context serving remain outside this preflight.

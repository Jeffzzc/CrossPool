# Qwen3.5-9B 文本适配与 GDN Elastic Context State

## Goal

在现有 SGLang/CUDA 部署边界内，适配官方 `Qwen/Qwen3.5-9B` checkpoint
的文本生成，让其 Dense FFN 在 FfnAgent 执行，Full Attention 与 Gated
DeltaNet（GDN）仍在 Instance Rank 执行；Full KV 与 GDN retained state
共同使用现有 ECS 字节配额，支持物理 backing 增长与安全回收。

本方案是待评审的目标设计，不表示已经实现或通过资格验证。用户已确认首版
先解决文本，图片输入与 ViT 留作后续兼容测试。使用官方 checkpoint 和
SGLang 的 text-only config override，不修改原始 `config.json`，不制作文本
权重副本。顶层仍是 `Qwen3_5ForConditionalGeneration`；本文讨论的
`Qwen3_5ForCausalLM` 是其文本计算骨干，不能代替 checkpoint 身份。

首个受测组合为 BF16 模型/FFN payload、FP32 GDN temporal state、BF16
conv state、Triton linear-attention backend、token `page_size=1`、attention
TP1/TP2、DP1、非 speculative、无 Mamba extra buffer，且关闭 overlap
schedule。Eager、Decode Full、Decode Full + Prefill Breakable 均须验收。
这些限制属于该适配的首次兼容边界，不改变其他已支持模型的运行模式。
选定 `mamba_radix_cache_strategy=no_buffer` 后，page size 1 与关闭 overlap
也是 pinned SGLang 自身的兼容约束。Full Attention 同样显式选择 Triton。

For the pinned SGLang 0.5.20, the public `--language-model-only` flag rejects
`Qwen3_5ForConditionalGeneration` in `handle_language_model_only()` before
loading. The existing `--json-model-override-args '{"language_model_only":true}'`
path preserves the official architecture and supplies the model-level flag.
Use that path for this target; a constructor field alone does not prove that
the corresponding public CLI flag is accepted.

The authorized target is the five-stage production text-only adaptation below,
building on [Preflight evidence](preflight.md). Experiments use physical Devices
0 through 4. Production model support remains unqualified until all required
numerical, lifecycle, elasticity and regression evidence is complete.

## Baseline

基线是本次调研时的仓库 `a3467ca`，依赖由 [pyproject.toml](../../../pyproject.toml)
与锁文件固定为 SGLang 0.5.20、Transformers 5.12.1。已通过
`init_global_config()` 和 `XpoolConfig.model_path_of()` 解析本地 9B 模型，
检查原始配置、权重索引与所有被引用 shard 的存在性。
Executable loading, storage and Graph compatibility evidence is recorded in
[Preflight evidence](preflight.md), with its qualification limits stated there.
外部依据、源码入口与计算公式见 [上游调研](research.md)。

| 已核实项目 | 当前事实及适配含义 |
| --- | --- |
| 模型结构 | hidden size 4096、32 层、FFN intermediate size 12288；每四层为 GDN/GDN/GDN/Full，Full 层号为 3、7、11、15、19、23、27、31。 |
| Full KV | 4 KV heads、head dim 256；池只能覆盖 8 个 Full 层，不能按 32 层计价。 |
| GDN | 24 层；16 K heads、32 V heads、K/V dim 128、conv kernel 4。TP1 一个完整 state slot 的有效载荷为 49.125 MiB，不含 VMM rounding、dummy slot 和临时 workspace。 |
| 权重 | 主 FFN 在 `model.language_model.layers.<i>.mlp.{gate,up,down}_proj.weight`；另有 `mtp.layers.*` 和 vision 权重。 |
| FFN 编译 | `FfnSourceConfig` 与 `Qwen3Adapter` 面向顶层字段及 `model.layers.*`；现有 checkpoint coverage 同样硬编码该 namespace，无法直接复用配置解析。Dense 算法与执行器可以复用。 |
| SGLang 接入 | Qwen3.5 两种 decoder 都调用 dense `Qwen2MoeMLP`，且 dense 分支不传 `ForwardBatch`；现有 shim 要求该参数。当前没有 Qwen3.5 adapter。 |
| ECS 控制面 | 已是 rank-local `MemoryQuota`、aggregate floor/reservation、绝对字节 quota command；不要求 daemon 理解 KV/GDN 池。 |
| ECS 本地机制 | `SglangKvComponent` 只绑定 Full/SWA token allocator；`KvVmmBacking` 要求统一 shape/dtype；需求向量、eviction 和 journal 都只覆盖 token page。 |
| 启动缺口 | `compute_reservation_budget()` 替换 `_profile_available_bytes()`，未执行其中的 `_handle_max_mamba_cache()`。不能假设 upstream 已自动定容 Mamba pool。 |
| 生命周期缺口 | `HybridReqToTokenPool` 独立持有 Mamba slot allocator、state 与 request-to-state 映射；COW、chunk snapshot、完成时 donation 不走 token allocator。当前 journal 不记录这些状态。 |

当前架构依据为 [ECS](../../designs/elastic-context-state.md)、
[FFN Execution](../../designs/ffn-execution.md)、
[Control Plane](../../designs/control-plane.md) 和
[Qualification](../../designs/qualification.md)。本任务扩展 ECS 的具体 SGLang
存储与 cache 合约，不更换已完成的字节配额控制面，不重写 GDN kernel。

## Proposed Changes

### 1. 原始模型身份与 FFN namespace

新增 `runtime/ffnagent/models/qwen3_5.py`，为官方
`Qwen3_5ForConditionalGeneration` 编译主文本 decoder 的 32 个 Dense
`DenseFfnSpec`。从严格验证的 `text_config` 读取 geometry、SiLU、dtype，
投影键始终保留 checkpoint 原名 `model.language_model.layers.*`。
不为 9B 硬编码全部尺寸；9B 的具体值由模型资格验证固定。

The first implementation registers only the official wrapper and Model ID.
Native `Qwen3_5ForCausalLM` checkpoints remain outside admission. The adapter
declares `checkpoint_decoder_prefix`, defaulting to `model.layers` for existing
families and using `model.language_model.layers` for the official wrapper.
This source namespace stays separate from `FfnModelSpec` execution mathematics.

`validate_ffn_coverage(spec, key_view, *, decoder_prefix,
allow_trailing_ffn_layers)` 在该 namespace 内逐层检查缺失、额外
projection、层号越界；发现另一个已知 decoder namespace 中的 FFN 键也拒绝，
避免混合两个语言骨干。保留现有 GLM trailing-layer 政策。
忽略 checkpoint 中的 MTP 权重不允许启动 MTP/speculative 执行。

新增 SGLang `Qwen3_5ShimAdapter`，精确识别官方 wrapper 的
`language_model_only=True` 路径。
SGLang's HookRegistry propagates class replacements to every module alias,
including other models' `Qwen2MoeMLP` bindings. Instead, wrap the Qwen3.5 wrapper
constructor and temporarily replace only `qwen3_5.Qwen2MoeMLP` during its
synchronous construction, restoring that binding in `finally`. The shim keeps
the original MLP constructor signature, accepts only the official wrapper's
`model.language_model.layers.<id>.mlp` prefix and owns no local FFN parameters.
Other source architectures and checkpoint identities remain outside admission.

在两个 Qwen3.5 decoder 的 forward 接缝传入 `forward_batch`，保留原始
norm、residual、GDN/Full attention、output gate 和 communicator 顺序。
首版关闭 MLP all-reduce fusion，并将该 adapter 的 communicator 限定为
`ScatterMode.FULL`、`allow_reduce_scatter=False`，使 FFN 输出交接沿用已支持
的完整 row 边界。不得使用通用 forward context 猜测缺失的 batch。

官方 wrapper 的 loader 在 rename 之前惰性过滤精确主 FFN subtree 与
`mtp.`；保留上游对 attention/GDN/embedding/head 的 checkpoint 映射。
post-load 验证全部 32 个
shim、layer ordinal、无本地 FFN 参数以及两种 decoder 的完整 MLP 边界。
vision 跳过由上游 `language_model_only` 完成；媒体请求沿用该模式的明确拒绝。

### 2. GDN state 的稳定存储与 slot 分配

保留上游 `HybridLinearKVPool`、`HybridReqToTokenPool` 和 Python
`UnifiedRadixCache`。前者的 Full child 使用现有 elastic MHA pool；后者通过
`mamba_pool_cls` / `_init_mamba_pool()` 接缝安装 `ElasticMambaPool` 和
`ElasticMambaSlotAllocator`。保持 hybrid 类型，不能用普通 `ReqToTokenPool`
替代；沿用 `mamba_map` 和 request-to-state 映射。

新增专用 `GdnVmmBacking`，不用只支持同 dtype/同 shape 的
`KvVmmBacking` 冒充 recurrent storage。它持有 conv planes 和 temporal
plane、每个 plane 的稳定 VA 与稀疏物理 inventory；所有 plane 与 Full KV
共用一个 `ecs.MemoryQuota`。数据仍为上游逻辑布局：

```text
conv[0]: [24, reserved_slots + 1, 8192 / TP, 3]       BF16
temporal:[24, reserved_slots + 1, 32 / TP, 128, 128]  FP32
```

shape/dtype 来自上游已解析 `mamba2_cache_params`，上式仅是 9B 实例。
每层 `[slots, ...]` 连续，只在最外层 layer stride 加 VMM granularity
padding；不改变单层 slot stride。这样 GDN Prefill 不需要因非连续 state
而进行整池 gather/scatter。池级 fused clear/copy 若因 outer stride 失去
contiguous 条件，保留上游正确的逐张量 fallback。

每个 plane 独立定义 bundle：同一字节区间跨其全部 GDN 层，成本为
`mapping_granularity * layer_count`。conv 与 temporal 的 bundle inventory
独立，slot 可横跨多个 bundle；slot 可交付的条件是它在所有 plane 中涉及
的 bundle 均已映射。不要为了对齐两种 row size，强制用其 LCM 个 slot
作为最小增长单位。后者会把较小 conv row 的 rounding 放大成巨额 state
批次。物理 bundle 与逻辑 slot 的转换必须计算区间覆盖和去重。

Back each layer interval with its own CUDA allocation handle and map at
allocation offset zero. A bundle is the coordinated inventory across those
handles, not one handle submapped at nonzero offsets: that latter operation
returns `CUDA_ERROR_NOT_SUPPORTED` on the preflight driver. Partial construction
must retain each handle and each successful mapping until checked cleanup.

每个 plane 的 `PoolProfile` 记录真实 bundle 成本、reservation 与覆盖
dummy slot 0 的 floor。向 `ElasticContextStateProfile.pools` 展平这些
profile，顺序为 Full、conv planes、temporal。只映射 dummy floor 即可
构造 state Tensor；所有池级初始化只访问已映射区间，禁止对完整虚拟
reservation 调用 `zero_()`。复用上游的 fresh-slot deferred clear；dummy
slot 在 capture 前清零并永久保持 backing。

`ElasticMambaSlotAllocator(MambaSlotAllocator)` 保留上游 slot API 与
slot 0 约定，增加 resident-free / unavailable / retiring 的分区，以及
group allocation 中尚未交付 slot 的真实所有权。优先使用 resident holes，
再按缺失 bundle 成本选择可映射 slot；同一 slot 的多 plane 必须联合计价。
只有全部 backing 成功才发布 slot。普通 OOM clean rollback 可留下已经
成功映射且仍收费的空闲 backing；设备异常或释放不确定沿用 Generation
failure，不能伪装成容量不足。

GDN slot id 始终是该稳定 tensor 的行号，沿用普通 hybrid pool 的 identity
translation，不引入 state compaction、第二份 request 表或 v2p 映射。
upstream unified-memory allocator 与 ECS 均有独立内存所有权，首版继续拒绝
`enable_unified_memory`，不将二者嵌套。

### 3. 明确 token page 与 state slot 的需求单位

`SglangKvComponent` 保留 KV 专属实现，增加
`SglangRecurrentComponent(component=ComponentType.MAMBA, allocator=...)`。
`SglangElasticContextState.components` 改为这两种 concrete binding 的 union，
逻辑顺序为 Full、可选 SWA、可选 MAMBA；本任务实际为 Full + MAMBA。
每个 binding 显式提供单位 `TOKEN_PAGE` / `RECURRENT_SLOT`，不得从数组长度
或 `is_hybrid_swa` 猜测。该逻辑顺序与上文多个物理 plane profile 的顺序分离。

内部单位相关 API 成组重命名并迁移全部 consumers：

| 当前接口 | 目标接口 / 含义 |
| --- | --- |
| `SglangMemoryRequirement.forecast_pages / allocation_pages` | `forecast_units / allocation_units: tuple[int, ...]`，每个位置按 binding 单位解释。 |
| `SglangPrefillProposal.preparation_pages` | `preparation_units`，包含本次实际 state 与 snapshot 需求。 |
| `logical_free_pages() / resident_free_pages()` | `logical_free_units() / resident_free_units()`。 |
| `allocation_pages(batch, token_per_req=...)` | `allocation_units(batch, token_per_req=...)`，KV 按 token 增量，MAMBA 按实际新增 slot。 |
| `plan_allocation / allocation_bytes / prepare_allocation` | 保留职责，参数改为 unit vector，返回 KV 或 recurrent concrete plan。 |
| reconciler 的 `admission_free_pages / deferred_eviction_pages` | 改为 `*_units`；Full/SWA token budget 的换算仅在相应分支执行。 |
| `eviction_page_targets(EvictParams)` | `eviction_unit_targets`，分别处理 `num_tokens`、`swa_num_tokens`、`mamba_num`。 |

所有向量长度等于逻辑 binding 数；非负，forecast 覆盖 immediate acquisition。
TP minimum 同步逻辑可用量；backing inventory、slot id 与 free-list 仍 rank-local。
精确 byte plan 汇总每个 backing 的 missing bundles 的并集，不按
`slots * 49.125 MiB` 代替 physical rounding。

### 4. 启动定容与共享 quota

在 upstream factory 之前建立不可变 `SglangHybridReservation`，字段包括
`full_token_capacity: int`、`recurrent_slot_capacity: int`、
`request_row_capacity: int`，以及已解析的 Full/GDN layouts。预描述必须与
factory 的实际 `HybridLinearKVPool`、Full child 和 Mamba state 描述相符。
主 Full layout 用 upstream `full_attention_layer_ids`，不用全部 decoder 数。

继续以 `compute_reservation_budget()` 的确定性字节预算描述最大 VA，但为
hybrid 显式执行一次上游 Mamba sizing 的非 speculative 分支，传入正确的
GiB/byte 单位，处理显式 slot cap、ratio 和 request cap。将结果存入上述
reservation，补齐目前 replacement 跳过的 `max_mamba_cache_size` 推导。
抽取后续搜索需要的纯 sizing 计算，不重复调用会修改 runtime context、
再次做 DP 缩放的 `_handle_max_mamba_cache()`。

原有 `mamba_full_memory_ratio` 只参与初始虚拟容量和 reference limits 推导，
不能成为 live quota 的固定 Full/GDN 份额。对 resolved slot/token caps
计算精确 VMM reservation 和 summed floor。floor 小于最低可执行布局或
显式配置无法容纳合法请求时明确启动失败。

factory 在一个 prepared-backing scope 下构造全部池，共用同一 ledger。
不先创建完整静态 `MambaPool` 再复制到 VMM；所有额外受限状态模式在分配
之前拒绝。registration 继续只传 aggregate floor/reservation。

Graph capture 之后，daemon 的 immutable service ceiling 同时约束 Full 和
GDN。扩展 `reference_bytes()` 与 `reference_config()`：精确计入 Full
backing、state dummy/rounding，以及能支持并发请求的 state 数。保留上游
request-count ratio 的保守边界，并校验至少一个 active slot 加后处理
snapshot credit 能执行。最终服务并发不高于 reserved request rows、capture
范围及上游约束；必要时只降低服务 limit，保持已分配 request table 和
captured tensors 的大小/地址。逻辑 token 上限与 concurrency 在这里一次
冻结，不能随着每次 quota command 重设 allocator 或 capture geometry。

GDN 临时 kernel workspace、activation 和 graph metadata 仍属于 Attention
Runtime Headroom，不得计入 retained state；任何实际新增的 retained plane
必须计入 ECS。现有 post-capture free-memory observation、headroom margin、
daemon donor policy 和 native control-channel ABI 都不需要增加 GDN 字段。

### 5. COW、缓存插入和分配的共同提交边界

只托管 state tensor 而不修改 admission 会在 prefix hit、chunk continuation
或 cache insertion 时触发上游断言。首版关闭 extra buffer 和 overlap，
让每请求只有一个 active slot，另对每个 Prefill 请求预备一个用于
`cache_unfinished_req` / chunk stash 的 snapshot credit。
这个 credit 是已分配且已 backing 的 slot，归真实 request 的临时执行
receipt 持有；实际插入后转交 cache，未使用或 duplicate 时归还。
它不是新的 daemon 配额或永久池份额。

将申请分成以下三个明确边界；低层 `alloc()`、`free()`、copy/clear hook
不得自行进入 collective：

1. **候选准备。** 在 `Req.init_next_round_input(tree_cache)` 改变字段前记录
   request checkpoint。匹配阶段调用上游纯匹配能力，禁用自动 COW 分配；
   把 source node/有效 state 边界作为该 attempt 的临时 match receipt。
   仍由上游决定最长合法 GDN checkpoint，不能直接采用更长的 Full KV hit。
   进入 `ElasticPrefillAdder` 的共同候选边界后，临时锁住确切 source，
   联合评估 Full page、新 active/COW destination 和 snapshot credit。
   预先批量 `alloc_group_begin(len(waiting_queue))` 不允许为整个 waiting
   queue 映射 state；elastic group 只分发当前候选已准备的 slot。
2. **实际 Extend 分配。** 共同准备 backing 后，通过原始 hybrid request
   pool 分配 request rows、缺失 active slots 和 request-to-state 映射；
   用已保留 source 设置 deferred COW，cold request 设置 deferred clear。
   snapshot credit 同样在此取得并写入 receipt。Full KV、state 与 request
   metadata 进入现有一次 `READY / CAPACITY_MISS / FAILED` TP MAX verdict。
   对不在 `alloc_for_extend()` 内的候选 clean miss，也在外层 Prefill wrapper
   统一 vote/rollback 后退出，不能只在某个 rank 提前返回。
3. **Forward 与缓存发布。** READY 后由上游收集 deferred clear/COW 并在
   forward stream 执行，再运行 GDN/Full attention。后处理的
   `MambaComponent._alloc_mamba_slot()` 消耗本请求的 snapshot credit，
   不临时借钱、不触发局部 eviction 或 TP collective。保留上游 insert、
   duplicate cleanup、finished donation 和锁规则。每个 Prefill 最多一次
   unfinished cache 插入；若 pinned 路径出现第二次申请，按接缝失配失败，
   不能在 forward 后将其解释为可回滚的 capacity miss。

正常 Decode 继续使用已有 active slot，不按每 token 新增 state；完成时
直接 donation active slot，不再申请 snapshot。retract/abort 释放请求
state，resume 走新的匹配和 active state 恢复路径；首版拒绝 beam 和持久
session，这两者会引入额外 state 分叉与跨轮次所有权。

扩展 `SglangRequestCheckpoint`，记录 `mamba_pool_idx`、
`mamba_cow_src_index`、`mamba_needs_clear`、`mamba_last_track_seqlen`
及本任务改变的有效匹配边界；journal 增加实际 slot acquisitions、
request-to-state mapping 旧值和 snapshot-credit receipts。候选 watch 必须
早于初始化匹配，而不是等到 `add_one_req()` 才 capture。
回滚先恢复映射/metadata，再按 receipts 归还 slot、row 和锁；避免
`HybridReqToTokenPool.free(req)` 隐式重复释放。已经提交的 cache eviction、
request retraction 与窗口释放保持提交，成功映射的 backing 继续收费。

Deferred copy/clear 尚未执行时允许 clean rollback；设备复制或 forward
开始后不再回滚 recurrence。COW source 必须由真实 tree lock/receipt 保护到
copy 完成，snapshot destination 归 receipt/cache 持有到相应使用完成。
首版无 overlap 让后处理与下一次规划有清晰顺序；之后恢复 overlap/extra
buffer 必须另行扩展 in-flight receipts 与状态写回安全证明。

### 6. MAMBA cache eviction、物理回收和 Graph

扩展 `SglangCacheEviction` 为 Full/SWA/MAMBA 的确切 component effect，
identity 继续使用含 namespace/salt 的 root-to-node Radix key，不广播 slot
id。借用 MAMBA 原策略选择候选；每个 TP rank 解析同一 node、验证 state
存在、锁引用和 cascade effect 后共同提交。MAMBA-only eviction 若会带走
Full 内容，须升级成显式 Full effect 并重新校验，不能隐式改变作用范围。
Full eviction 的真实 cascade 可能归还 state；只按实际 free receipts
更新 slot owner，不用 `mamba_num` 推测物理已释放。

活跃 state、COW source、snapshot credit、cache snapshot 与 group 里未交付
slot 都不是可回收 slot。只有 plane bundle 覆盖的全部 slot 已不归任何
逻辑 owner，才 withdraw 相交 free slot 并开始 retirement。任何 slot 所需
plane 处于 retirement 时，该 slot 不能重新分配。与 Full 共用现有执行
stream CUDA Event：确认其先前 clear、copy、GDN writes、cache copy 及
Graph replay 已完成，才 unmap；最后释放成功才退款。即使 cache eviction
没有凑成一个完整空 bundle，也必须保留准确的未释放收费。

quota shrink 保留原有 chunk-owner 保护与 donor completion 合约；active
state 可以延迟 shrink 完成，不能为了给 peer 退款而丢弃 recurrence。状态
不足时用上游普通 retraction/recompute 路径，与下一轮 grant 需求一起协调。
独立 rank 的物理 Event 完成不改变 collective 次序。

GDN graph metadata 的 `state_indices_list` 沿用上游按 batch size 预分配并
原地 `copy_()` 的逻辑。VA、shape、stride、dummy slot 在全部 quota 调整中
固定；replay 前只填入已 backing 的 live indices。capture/warmup 中若引用
非 dummy slot，也须登记并 pin 对应临时 ownership，完成 capture 后安全
释放，实际 backing 计入 startup publication。禁止用大 reservation
Tensor 的整体清零来完成 warmup。

## Interface Changes

除上文 namespace 与 unit API 外，新增以下 SGLang 本地具体接口；签名是
本方案的目标声明，当前源码尚不存在：

```python
GdnVmmBacking.describe(*, device, size, cache_params, layer_ids) -> tuple[GdnPlaneLayout, ...]
GdnVmmBacking.plan_slots(slot_ids: torch.Tensor) -> GdnBackingPlan
GdnVmmBacking.try_map(plan: GdnBackingPlan) -> bool
GdnVmmBacking.empty_bundles(free_slot_ids: torch.Tensor) -> tuple[tuple[int, ...], ...]

ElasticMambaSlotAllocator.plan_slots(count: int) -> SglangSlotPlan | None
ElasticMambaSlotAllocator.begin_retirement(bundle_ids_by_plane) -> None
ElasticMambaSlotAllocator.finish_retirement() -> None
```

plan 均为无分配副作用的 immutable 值。`SglangSlotPlan` 包含选中 slot IDs 和
各 plane 的 missing bundle IDs；backing plan 只拥有几何/ID，不拥有 logical
slot。`try_map` 受共享 ledger 限制；普通容量不足返回 False；不确定设备
失败抛出原异常并保留 resource owners。slot publication 和 retirement
只能在 Scheduler 线程发生，construction 属于原 startup owner。

`ElasticMambaPool` 需要保持上游 `State`、`mamba2_layer_cache`、
`clear_slots`、`copy_from` 的可调用合约及 identity translation。若上游实现
依赖 constructor 设置的字段，显式完成所需初始化，不先调用完整静态池
构造函数来获得这些字段。增加 `validate_seams()` 对新增 hook 签名与所选
state 模式的启动检查，后续依赖升级须重新资格验证。pool stats 与 MAMBA
invariant checker 要区分逻辑未分配、resident free 和 unavailable；逻辑
空闲不能直接当成可交付 slot，也不能把 unavailable 误报成泄漏。

现有 `SglangLaunchModel` 只携带 Model ID 与 graph mode，尚不能传递本任务
所需模式。扩展为带默认值的 typed `server_options: SglangServerOptions`，
包含 `language_model_only: bool=False`、`disable_overlap_schedule: bool=False`、
`page_size: int | None=None`、`attention_backend: Literal["triton"] | None=None`、
`linear_attn_prefill_backend / linear_attn_decode_backend:
Literal["triton"] | None=None` 与
`mamba_radix_cache_strategy: Literal["no_buffer"] | None=None`。
选定策略的条件组合在准备阶段校验，普通模型保持默认行为。
`E2eServingCase` 增加 `server_options: Mapping[ModelId, SglangServerOptions]`
（默认空映射），键必须是 case 的 Model ID；launch preparation 原样传递，
`xkit` 的 `server_command()` 唯一负责投影成以下 engine flags：

```text
--json-model-override-args '{"language_model_only":true}'
--disable-overlap-schedule --page-size 1
--attention-backend triton --mamba-radix-cache-strategy no_buffer
--linear-attn-prefill-backend triton --linear-attn-decode-backend triton
```

不使用按 Model ID 隐式注入参数的逻辑，也不增加任意 CLI 字符串透传。
独立 FFN reference 的 `FfnReferenceJob` 增加
`language_model_only: bool=False`，9B numerical case 显式声明该值。
Project this semantic option through `ServerArgs.json_model_override_args`,
not the rejected `ServerArgs.language_model_only` flag. Preserve other explicit
JSON overrides and reject a conflicting value rather than silently replacing it.
`reference/ffn_child.py` 增加官方 wrapper 的 concrete
class 验证与 `model.layers` 提取，保持 plugin isolation，不以 CrossPool
source adapter 的输出充当独立参考。原生 SGLang 全模型 serving 对比复用
同一个 engine-options 值并关闭 CrossPool plugin。

不增加服务 HTTP API、Fabric FFN enum、native ECS ABI、GDN daemon RPC、
新 serving engine 或新的配置文件路径解析方式。

## Implementation

依赖顺序按完整机制划分；每一层完成行为验证再连接上层，避免出现能 offload
FFN 却静态持有整个 GDN pool 的伪弹性完成条件。

```text
src/xpool/runtime/ffnagent/
  architecture.py / checkpoint.py     修改：家族 namespace 的严格覆盖
  models/qwen3_5.py                   新增：wrapper/causal source 编译
src/xpool/integrations/sglang/
  models/qwen3_5.py                   新增：shim、decoder 和 loader 接缝
  ecs/recurrent.py                   新增：混合 dtype state VMM、pool/slot 机制
  ecs/state.py                       修改：Full + MAMBA 组合、联合定容/需求
  ecs/allocation.py / admission.py    修改：COW、snapshot credit、rollback
  ecs/radix.py / capacity.py          修改：MAMBA eviction 与 retirement
  hooks/ecs.py / server_args.py       修改：factory、mode gate、共同边界
src/xpool-dev/xtest/harness/sglang/
  catalog.py / numerical.py          修改：模型 mode 声明与传递
  reference/ 与 serving/             修改：原始 wrapper reference 与文本启动
src/xpool-dev/xkit/serving/sglang/
  launch.py / server.py              修改：typed engine options 与 CLI 投影
tests/suites/
  unit/runtime/ffnagent/             增加 FFN 配置/coverage 行为
  integration/sglang/models/         新增 qwen3_5 adapter 行为
  integration/sglang/ecs/            增加 state、transaction、回收行为
  models/Qwen/Qwen3.5-9B/            新增真实 checkpoint 资格验证
configs/deployments/                 增加该模型的可移植 TP1/TP2/peer scene
```

1. **状态存储与定容。** 先完成 recurrent planes、slot allocator、Hybrid
   factory、混合单位 binding 和共同 byte pricing。将 Full/SWA 所有
   consumers 同步迁移，保留已有回归。验证 dummy、mixed dtype、slot 跨
   bundle、部分失败、延迟释放和 shape/stride，再接启动 publication。
2. **调度与生命周期。** 实现纯 match、候选准备、共同分配 journal、
   snapshot credit、MAMBA eviction 与 Event retirement，覆盖所有 state
   allocation sites。关闭未受测的 GDN modes，完成 TP1/TP2 的非对称
   clean miss / failure / rollback，保留 idle quota progress。
3. **模型接入与整体验证。** 加入 FFN source namespace、model shim、
   decoder/loader hooks 和真实 checkpoint scenes，完成独立参考、三类
   graph 与 sharing pressure 验收。实现接受后用 `write-design` 更新 ECS、
   FFN、qualification 所属文档；plan 清理由用户确认，不自动删除。

## Validation

遵循 [测试架构](../../../tests/README.md)。以下是实施后的必需证据；本次
文档调研不运行 Device tests，也不宣称模型已 supported。

### 纯逻辑与组件

- FFN wrapper/text config、dtype、层模式和原始投影键编译；缺失/多余/
  越界键、混合 namespace、量化和 MTP 执行拒绝；旧模型 namespace 行为
  保持。独立参考从 pinned SGLang 原始 dense MLP 构造，覆盖 GDN 后和
  Full 后的 FFN 层，例如 0、3、31。
- 真实 CUDA recurrent backing：FP32 temporal/BF16 conv、TP1/TP2 shape、
  每层 contiguous、跨 plane 的精确定价、slot 跨 bundle、sparse holes、
  allocation group、partial map rollback、dummy floor 与稳定 pointer。
- 冷启动 clear、prefix COW、source lock、长 Full hit/短 GDN checkpoint、
  chunk stash、duplicate insert、finish donation、abort/retract/resume
  及 slot 重用。比较实际 state 内容和输出，不能只检查 free-list 长度。
- TP 非对称 inventory 和本地不足：在 pure match、active slot、snapshot
  credit、request rows、Full allocation 失败后，各 rank 保持同一逻辑
  状态与 collective 次序；rollback 不泄漏、不 double-free。
- 同 quota 内 Full/GDN 互相增长、MAMBA-only/Full cascade eviction、
  active state 阻挡 shrink、snapshot credit 存活时不得 unmap、CUDA Event
  前不退款、完成后 physical release 和 command completion 一致。

### 真实模型与 installed serving

模型 suite 为 `tests/suites/models/Qwen/Qwen3.5-9B/`，权重只经有效
CrossPool config 解析，不下载。相同官方 checkpoint、dtype、文本模式与
backend 下建立三个证据层：

1. 独立原生 SGLang 文本参考与 CrossPool Eager 的首 Prefill logits 和
   greedy token 行为，覆盖 cold、prefix reuse/branch、multi-chunk 和
   retract/resume，验证 decoder 接缝与 recurrence。使用已有 numerical
   policy；必要的完整模型 forward KL 使用现有 serving-logit 比较机制，
   不因 GDN 另设宽松阈值。确定性参考差异须保留原始 logits/tokens 排查。
2. 现有模型资格验证：独立 FFN reference 的 rows 1/32/4096；Eager 对
   Decode Full 的 token IDs；Eager 对 combined mode 的 first-prefill
   full-distribution KL，同时 Graph Observer 证明 Breakable 真正执行。
   TP1/TP2 各覆盖真实 GDN state indices、padding 与多次 graph replay。
3. GDN 与现有小 Qwen 共置同一 attention Device，主动造成 quota grow/
   shrink：验证前缀命中减少、可恢复的稳定命中、两模型并发完成、存活
   request 连续 decode 与无收缩参考一致，以及 slot 重用无旧请求污染。
   组件证据证明 map/unmap 与 Event，E2E 不依赖日志推断协议完成。

用可移植 deployment 声明 attention TP1/TP2、FFN geometry 和 SLO；精简
resource matrix，复用已验证的 Fabric cardinality 与 lifecycle harness。
CPU/Device 资源声明、进程清理和 attention MPS 边界遵循现有约定。

最后执行 canonical build/stubs、受影响资格验证与冻结源码上的完整 strict
suite；命令包括 `uv run xtest run --suite Qwen/Qwen3.5-9B
--strict-requirements`，并满足 [最终验收](../../designs/qualification.md#acceptance-and-invalidation)。
不把启动成功、FFN 单测或一次短文本请求当作 GDN 弹性通过。

`docs/supported-models.md` 保留官方完整架构名称。文本验收只支持标注清楚
`text-only / language_model_only config override` 的已验证范围；不得把尚未测试的图片/
ViT 能力隐含在无条件的 `Supported` 标记中。后续图片兼容测试沿用上游
vision 路径，并检查 media Prefill、M-RoPE、文本/图片 prefix、workspace
headroom 和 graph；只有那一阶段完成后才扩大声明。

## Out of Scope

- 图片/视频输入与 ViT 的首版实现或验收；保留后续兼容测试工作。
- Qwen3.5 MoE、量化权重、其他参数规模的支持声明。
- MTP/speculative、ReplaySSM、INT8 recurrent checkpoints、extra/ping-pong
  buffers、overlap、beam、持久 session、DP attention、CP/PP、PD/offload。
- 重写 GDN 数学/kernel、state compaction、live quota 固定份额、新的
  daemon 策略或 native 传输协议。
- 性能门槛。记录吞吐、TTFT/TBT、state backing 与 cold/COW 成本作为
  report-only 结果；性能优化在正确性证据后进行。

## Open Questions

首版输入范围已确定为文本。上述 mode、layout、所有权和验收是本版提出的
具体取舍，等待方案评审；不存在需要猜测本地模型身份或额外权重路径的问题。
The selected per-layer strides, sparse VMM mapping and GDN/phase-runner Graph
behavior have executable preflight evidence. Production post-capture concurrency
sizing, joint admission, cache ownership and quota reconciliation still require
the implementation and qualification described above.

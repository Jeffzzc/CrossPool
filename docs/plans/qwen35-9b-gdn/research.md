# Qwen3.5-9B 与 GDN 上游调研

本文件记录可复核的上游事实和对适配的约束；目标设计由本目录
`README.md` 决定。调查以仓库锁定的 SGLang 0.5.20、Transformers 5.12.1
为执行基线，已检查本地安装包中的对应源码。在线源码链接固定到 SGLang
`v0.5.20`，避免把 `main` 后续变化误认为当前依赖的能力。

## 模型身份和计算结构

官方 `Qwen/Qwen3.5-9B` checkpoint 的顶层架构实际是
`Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`；文本结构在
`text_config`，其 `model_type=qwen3_5_text`。因此“支持
`Qwen3_5ForCausalLM`”必须同时规定如何从原始官方 checkpoint 进入文本执行
路径，不能只注册一个官方配置里没有出现的 architecture 字符串。
[官方配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)

9B 文本结构如下，均应读取配置而非硬编码成整个模型族的常量：

| 字段 | 9B 配置值 |
| --- | --- |
| 隐藏维度 / 层数 / FFN 中间维度 | 4096 / 32 / 12288 |
| 层模式 | 每 4 层为 GDN、GDN、GDN、full attention |
| full attention | 16 个 Q heads，4 个 KV heads，head dimension 256 |
| GDN | 16 个 K heads，32 个 V heads，K/V head dimension 均为 128 |
| 卷积窗口参数 | `linear_conv_kernel_dim=4` |
| 状态精度 | `mamba_ssm_dtype=float32` |
| FFN | dense SiLU gated MLP |

因此 0-based full attention 层号为 `3,7,11,15,19,23,27,31`，其余 24 层
为 GDN。配置还包含 attention output gate、partial rotary factor 0.25、
M-RoPE 参数、非共享 embedding/head 和 MTP 配置，均不能因“dense FFN 相同”
而忽略。[官方配置](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json)

上游同名类存在不同层级：`models/qwen3_5.py:Qwen3_5ForCausalLM` 是文本
backbone，`models/qwen3_5_text.py:Qwen3_5ForCausalLM` 才是带 LM head 和
logits processor 的可注册文本模型。后者显式处理 M-RoPE positions、权重
前缀剥离及 LM head 加载。不能把 backbone 当作完整生成模型替换。
[文本模型入口](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/qwen3_5_text.py)

原始 conditional-generation 入口也支持 `language_model_only`，该标志由模型
配置传给上游多模态基类，跳过 vision tower；它与把顶层配置手工改名为
causal architecture 是不同操作。适配实现应选择单一受测的文本入口并保留
官方 config/checkpoint，不生成私有 checkpoint 变体。
[配置解析](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/configs/model_config.py)、
[多模态基类](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/qwen3_vl.py)

The executable preflight corrects the earlier CLI inference: SGLang 0.5.20's
`handle_language_model_only()` rejects the Qwen3.5 wrapper through the public
`--language-model-only` flag. The existing JSON config override
`{"language_model_only":true}` reaches the model-level text-only path while
preserving the official checkpoint. `--language-only` remains the distinct
encoder-disaggregation mode. The backbone construction prefix is still
`model.language_model`, whereas the loader normalizes parameter dictionary keys.
[Argument validation](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/arg_groups/model_hook.py),
[configuration](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/configs/model_config.py).
See [executable evidence](preflight.md) for the exact validation boundary.

## FFN 可以复用，attention 需要完整保留

上游 GDN 与 full-attention decoder 都在 `qwen3_5_text` 分支构造
`Qwen2MoeMLP`，其中使用合并的 `gate_up_proj`、SiLU-and-multiply 和
`down_proj`。这支持复用 CrossPool 已有 dense gated FFN 执行器的方向；
资格验证仍须覆盖权重名、TP 切片、bias 和量化 schema，不能仅凭类名下结论。
[decoder 构造及权重映射](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/qwen3_5.py)、
[MLP 实现](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/qwen2_moe.py)

GDN 的投影、门控、卷积和 recurrence 均留在 attention participant。上游已有
`RadixLinearAttention` 到 `GDNKernelDispatcher` 的执行路径，包含 Triton 与
平台相关的其他 kernel；本任务无需重写 GDN 数学或 CUDA kernel。不同 kernel
对 SSM dtype 和连续性有约束，应固定首个受测组合。
[GDN backend](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/attention/linear/gdn_backend.py)

## 两类状态的布局、索引与生命周期

GDN 不为每个 token 追加历史 K/V，而是更新每请求的卷积状态和 recurrent
matrix。`Qwen3NextConfig.mamba2_cache_params` 为 GDN 构造 TP 后的状态尺寸；
conv 顺序是分别 TP 分片的 `[Q,K,V]` 子块，而不是对全拼接矩阵任意等分。
层号映射也必须区分 24 个 GDN 层和 8 个 full-attention 层。
[GDN 配置与形状推导](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/configs/qwen3_next.py)

设 attention TP degree 为 `T`，且相关 head 数可整除 `T`。按默认 BF16
conv / FP32 temporal 推导，9B 非 speculative 模式的持久状态为：

```text
conv:     [24, slots+1, 8192/T, 3]        bfloat16
temporal: [24, slots+1, 32/T, 128, 128]   float32
```

这里 `+1` 是保留 slot。单个完整请求状态在 TP1 为
`24 * (8192*3*2 + 32*128*128*4) = 51,511,296 bytes = 49.125 MiB`。
该数值是由上游形状公式计算得到，不包含 allocator、graph、prefix snapshots
和 scratch。不能用模型 BF16 dtype 推断 recurrent state 也是 BF16。
[状态形状与 dtype 规则](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/configs/mamba_utils.py)

`HybridReqToTokenPool` 在普通 request-to-token 映射之外持有
`mamba_allocator`、`req_index_to_mamba_index_mapping`、`mamba_map` 和
`mamba_pool`。GDN backend 从 `req_to_token_pool.mamba2_layer_cache(layer_id)`
取得状态。普通池的 slot id 就是 physical id；统一内存池才增加 virtual 到
physical translation。扩展池必须保持 `HybridReqToTokenPool` 类型约束。
构造接缝为 `mamba_pool_cls` 与 `_init_mamba_pool()`，适合接入托管 backing。
[池定义](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/mem_cache/memory_pool.py)

`MambaPool.State` 至少包含 `conv: list[Tensor]` 和 `temporal: Tensor`；
ReplaySSM/speculative 模式还有 ring、cursor 或中间状态。首版若未管理这些
附加状态，应在启用前拒绝该组合，而不是只迁移两个主张量。
[状态对象](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/mem_cache/memory_pool.py)

## Prefix cache、COW 与调度

上游 `MambaComponent` 已有 prefix 状态语义：tree node 保存某个 token
边界的状态 snapshot；命中后为请求分配 writable slot，并记录 COW source。
Full KV 命中边界可能超过最近的可复用 GDN snapshot，需要从有效状态边界
补算。不能把 GDN snapshot 当作 full token prefix 的同长度数组。
该组件断言 request pool 是 hybrid 类型；禁用 extra buffer 时要求
`page_size=1`。
[MAMBA tree component](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/mem_cache/unified_cache/components/mamba.py)

配置校验 `validate_mamba_no_buffer()` 同时要求关闭 overlap schedule、
page size 为 1（或尚未解析），并排除 `trtllm_mha` attention backend。
因此首版选择 `no_buffer` 时，关闭 overlap 是上游约束，不只是本地简化。
[Mamba 参数校验](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/arg_groups/mamba_hook.py)

clear/COW 在 `ModelRunner._maybe_execute_deferred_mamba_cow_and_clear()`
里于 extend forward 之前、forward stream 上执行；输入字段为
`ForwardBatch.mamba_clear_indices`、`mamba_cow_src_indices` 和
`mamba_cow_dst_indices`。它对需要的 id 做 physical translation，执行复制后
清空字段。弹性 admission dry-run 必须记录和回滚这些副作用，不能提前复制
或把字段统一清成 `None`。
[deferred clear/COW](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/model_executor/model_runner.py)

非 speculative 情况下，请求有 1 个 active slot；extra buffer 还会占用
tracking slots，overlap 模式通常使用两个 ping-pong tracking slots，另有
prefix snapshot 占用。admission 不能以 batch size 等同完整状态 slot 需求。
[request slot 管理](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/mem_cache/memory_pool.py)

上游预算由 `KVCacheConfigurator._handle_max_mamba_cache()` 处理，输入包括
`max_mamba_cache_size`、`mamba_full_memory_ratio`、最大请求数及并行切分。
显式 slot cap 和按 ratio 推导是不同路径；必须防止在 CrossPool arena 预算
之外又扣一遍或额外分配静态 Mamba pool。
[预算计算](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/mem_cache/kv_cache_configurator.py)

## CUDA graph 与内存弹性约束

`HybridLinearAttnBackend` 同时初始化 full 和 linear 两个子 backend 的
metadata。linear backend 为各个 capture batch size 保留静态
`state_indices_list`，replay 用 `copy_()` 原地写入 physical indices。
因此 graph 兼容要求状态 tensor 地址稳定，replay 前更新映射，padding slot
不得指向真实请求。仅保持 full KV 的地址稳定不够。
[hybrid backend 与 graph metadata](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py)

上游 unified-memory 模式已有 `UnifiedMambaPool`：同一 byte buffer 中的混合
dtype strided views，配合独立的 full-token 和 mamba-slot id 空间。它证明
混合状态可由一个 byte arena 承载，但不能直接替换 CrossPool ECS 的原生
frame 所有权与事务边界。
[统一 backing](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/mem_cache/unified_memory_pool.py)

`UnifiedMambaTokenToKVPoolAllocator` 把状态视为 `page_size=1` 的整请求 slot，
full 仍按 token page 分配；联合 admission 需要按 byte 计算二者消耗，slot
成本向 token 等价值换算时向上取整。应借鉴其独立单位与共同预算，避免把
MAMBA 塞进现有只表达 token acquisition 的组件。
[联合分配器](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/mem_cache/allocator/unified_mamba.py)

对独立 VMM backing 的布局建议是保留每层 `[slots,...]` 的 dense slot stride，
仅在最外层 layer stride 加映射粒度所需 padding。原因是 GDN extend 检查
每层 conv/temporal view 的 `is_contiguous()`；slot-strided envelope 需要额外
gather/contiguous/scatter 以避开 CUDA prefill kernel 的连续写回假设。
outer layer padding 不破坏单层连续性。池级 clear/COW 的索引赋值可以处理
这种 padding；全张量不连续时上游自动禁用 fused conv slot copy，退回正确的
逐张量实现。此建议仍需用受测 Triton prefill/decode 和 slot copy 验证。
[prefill 连续性处理](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/attention/linear/gdn_backend.py)、
[slot clear/copy](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/mem_cache/memory_pool.py)

## 对目标设计的直接约束

以下为基于上述事实的设计推论，不是上游已经提供给 CrossPool 的能力：

1. 首版定义为官方 9B checkpoint 的 text-only、dense BF16、非 speculative
   路径；复用 dense FFN，保留上游 attention/GDN 数学实现。
2. ECS 需要同时表达 full token pages 与 GDN state slots，以及 conv/temporal
   不同 dtype 的稳定 tensor views；frame packing 和分配单位显式区分。
3. prefix match、COW、clear、slot 分配、tracking 和完成释放须纳入同一个
   admission/rollback contract，先做到不会污染或泄漏状态，再验证回收能力。
4. 内存收缩必须识别 active/tracking/COW-source/cache snapshot 的存活状态；
   回收 snapshot 必须同时更新 prefix tree，active state 不可静默丢弃。
5. 资格验证至少区分 upstream 本地基线、CrossPool FFN offload 基线及
   ECS grow/shrink 后的连续 decode，覆盖 chunked prefill、prefix 分叉、请求
   slot 重用和 graph replay。上游能加载模型不等于 CrossPool 已合格。

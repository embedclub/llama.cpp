# `include/llama.h` 功能分析报告

项目：`llama.cpp`（ggml-org） | 文件：`include/llama.h` | 总行数：1589 行 | 报告日期：2026-11-12

## 目录

1. 文件概述
2. 整体架构与设计原则
3. 文件头部：宏、导入与符号可见性
4. 核心类型与枚举定义
5. 参数结构体
6. API 功能模块逐一分析
7. 废弃（Deprecated）与向后兼容机制
8. 典型使用生命周期
9. 总结与观察

## 1. 文件概述

`include/llama.h` 是 llama.cpp 项目的**唯一公共 C API 头文件**，是对外暴露的全部库接口。它为 LLaMA 系列以及其他大语言模型（支持 GGUF 格式的编码器、解码器、编码-解码器、递归型如 Mamba/RWKV、混合型如 Jamba、扩散型如 LLaDA/Dream 等模型）提供：

- 模型加载/保存/量化（GGUF 格式）
- 推理上下文的创建与控制
- batch 编码与解码（token 推理）
- 词表（vocab）操作与分词/反分词
- 聊天模板应用
- 采样器链（sampling chain）
- KV/记忆（memory）管理与序列状态
- 状态/会话（session）保存与恢复
- LoRA 适配器、控制向量（control vector）
- 性能统计、日志、训练（opt）等辅助功能

> **关键特征：**纯 C 接口（`extern "C"`），不透明指针（opaque pointer）封装实现细节，所有公共函数以 `LLAMA_API` 宏标注导出属性，兼容 Windows DLL 与 GCC `visibility` 两种导出机制。

## 2. 整体架构与设计原则

| 设计原则 | 体现 |
| --- | --- |
| 不透明类型（Opaque Types） | 仅前置声明 `struct llama_model`、`struct llama_context`、`struct llama_vocab`、`struct llama_sampler`、`struct llama_adapter_lora`，外部只能通过 API 函数操作，隐藏实现（PIMPL 风格）。见 61–66 行。 |
| 依赖 GGML | 头文件引入 `ggml.h`、`ggml-cpu.h`、`ggml-backend.h`、`ggml-opt.h`、`gguf.h`，复用 GGML 的张量、后端调度、线程池、优化器与 GGUF 容器设施（4–8 行）。 |
| 参数结构体 + 默认值函数 | 配置通过值传递的结构体（`llama_model_params` / `llama_context_params` / `llama_model_quantize_params`）完成，并提供 `llama_*_default_params()` 获取默认值，保持 ABI 稳定式演进。 |
| 回调注入 | 进度回调、中止回调、eval 回调、tensor 数据回调、日志回调、参数筛选回调等，允许宿主程序介入库内部流程。 |
| 命名分区 | 文件按注释分节组织：C interface / Memory / State / Decoding / Vocab / Tokenization / Chat templates / Sampling / Model split / Performance / Training 等。 |

## 3. 文件头部：宏、导入与符号可见性 （1–53 行）

- **Include guard**：`LLAMA_H` 防止重复包含。
- **`LLAMA_API` 导出宏**（15–27 行）：编译共享库时，Windows 下按 `LLAMA_BUILD` 区分 `dllexport/dllimport`；GCC/Clang 使用 `__attribute__((visibility("default")))`；静态库时为空。
- **`DEPRECATED(func, hint)` 宏**（29–35 行）：跨编译器标注废弃函数，并给出迁移提示（GCC 用 `__attribute__((deprecated))`，MSVC 用 `__declspec(deprecated)`）。
- **常量宏**（37–49 行）：

  - `LLAMA_DEFAULT_SEED` = 0xFFFFFFFF（随机种子）
  - `LLAMA_TOKEN_NULL` = -1（无效 token）
  - 历史文件魔数 `ggla/ggsn/ggsq`；会话文件 magic `ggsn` + 版本 9；单序列状态文件 magic `ggsq` + 版本 2。

## 4. 核心类型与枚举定义 （61–244 行）

### 4.1 基础不透明类型与别名

```cpp
struct llama_vocab; struct llama_model; struct llama_context; struct llama_sampler;
typedef struct llama_memory_i * llama_memory_t;   // 记忆（KV/递归缓存）接口句柄
typedef int32_t llama_pos;      // token 位置
typedef int32_t llama_token;    // token id
typedef int32_t llama_seq_id;   // 序列 id
```

### 4.2 主要枚举

| 枚举 | 作用 | 要点 |
| --- | --- | --- |
| `llama_vocab_type` | 词表/分词器类型 | 支持 SPM、BPE、WPM(BERT)、UGM(T5)、RWKV、PLaMo-2 共 7 种，反映多模型家族支持。 |
| `llama_rope_type` | 旋转位置编码类型 | NONE / NORM / NEOX / MROPE / IMROPE / VISION，部分值直接映射 GGML 的 RoPE 常量。 |
| `llama_token_type` / `llama_token_attr` | token 类别与位掩码属性 | attr 支持 NORMALIZED、LSTRIP、RSTRIP、SINGLE_WORD 等细粒度标志；旧 type 枚举标注待移除。 |
| `llama_ftype` | 模型量化文件类型 | 从 F32/F16 到 Q2_K~Q8_0、IQ 系列非对称量化、BF16、TQ1_0/TQ2_0、MXFP4_MOE、NVFP4、Q1_0 共 40 余种，含已删除项的注释。 |
| `llama_rope_scaling_type` | RoPE 外推/缩放策略 | NONE / LINEAR / YaRN / LongRoPE，均可用模型内置值（UNSPECIFIED）。 |
| `llama_pooling_type` | embedding 池化方式 | NONE / MEAN / CLS / LAST / RANK（rerank 模型把分类头接入计算图）。 |
| `llama_attention_type` | 注意力模式 | CAUSAL / NON_CAUSAL（embedding 场景）。 |
| `llama_flash_attn_type` | Flash Attention 开关 | AUTO / DISABLED / ENABLED，另有 `llama_flash_attn_type_name()` 返回名称。 |
| `llama_split_mode` | 多 GPU 切分策略 | NONE（单卡）/ LAYER（按层）/ ROW（行级张量并行）/ TENSOR。 |
| `llama_model_kv_override_type` | 模型元数据覆盖类型 | INT / FLOAT / BOOL / STR，配合 tagged union `llama_model_kv_override`（key 最长 128 字节）。 |
| `llama_model_meta_key` | 采样相关元数据键 | 枚举模型内嵌的推荐采样参数（temperature、top_k、mirostat 等）。 |
| `llama_params_fit_status` | `llama_params_fit()` 的结果 | SUCCESS / FAILURE / ERROR。 |

### 4.3 关键数据结构

- **`llama_token_data` / `llama_token_data_array`**（201–215 行）：采样候选 token 及其 logit/probability；数组带 `selected`（数组下标而非 token id）与 `sorted` 标志——调用方不得假设数据已排序。
- **`llama_batch`**（235–244 行）：**推理输入的核心结构**，SoA 布局：

  | 字段 | 说明 |
  | --- | --- |
  | `token` | token id 数组（与 `embd` 二选一） |
  | `embd` | 直接输入嵌入向量（n_embd 维 float） |
  | `pos` | 位置数组，NULL 时自动跟踪 |
  | `n_seq_id` / `seq_id` | 每个 token 所属的多个序列（NULL 默认 seq 0） |
  | `logits` | int8 标志，非 0 才输出该 token 的 logits/embeddings（注释提示未来改名为 output） |

- **`llama_progress_callback`**（217 行）：模型加载进度回调，返回 false 可中止加载。
- **`llama_logit_bias`**、**`llama_chat_message`**（role + content）、**`llama_model_imatrix_data`**（量化重要性矩阵）、**`llama_model_tensor_override` / `llama_model_tensor_buft_override`**（按名称模式覆盖张量类型/缓冲区类型）。

## 5. 参数结构体

### 5.1 `llama_model_params` （286–322 行）

控制模型加载行为：

- `devices`：NULL 结尾的设备列表（NULL = 使用所有设备）。
- `tensor_buft_overrides`：按模式为张量指定 buffer type。
- `n_gpu_layers`：GPU 卸载层数，负值 = 全部。
- `split_mode` / `main_gpu` / `tensor_split`：多 GPU 分配。
- `progress_callback(+user_data)`：加载进度/中止。
- `kv_overrides`：覆盖 GGUF 元数据键值。
- 布尔组（注释说明集中放置以避免对齐问题）：`vocab_only`（只载词表）、`use_mmap`、`use_direct_io`、`use_mlock`、`check_tensors`、`use_extra_bufts`（权重重打包）、`no_host`、`no_alloc`（仅载元数据，模拟分配）。

### 5.2 `llama_context_params` （331–383 行）

控制推理上下文，是 API 中最复杂的结构：

- **尺寸**：`n_ctx`（0 = 从模型读取）、`n_batch`（逻辑批）、`n_ubatch`（物理批）、`n_seq_max`（最大并行序列数）、`n_threads`/`n_threads_batch`。
- **注意力/位置编码**：rope_scaling_type、pooling_type、attention_type、flash_attn_type、rope_freq_base/scale、YaRN 参数组（ext_factor、attn_factor、beta_fast、beta_slow、orig_ctx）、`defrag_thold`（已废弃的 KV 碎片整理阈值）。
- **回调**：`cb_eval`（后端调度 eval 回调）、`abort_callback`（目前仅 CPU 侧生效）。
- **[EXPERIMENTAL] KV 缓存量化**：`type_k`/`type_v`（GGML 数据类型）。
- **布尔组**：`embeddings`、`offload_kqv`、`no_perf`、`op_offload`、`swa_full`（全尺寸 SWA 缓存，多序列下设 false 可能有性能问题）、`kv_unified`（跨序列共享统一 KV 缓冲）。
- **[EXPERIMENTAL] 后端采样**：`samplers`/`n_samplers`——把采样链嵌入计算图（需为 `llama_sampler_chain_init` 构造的链，调用方须保持 sampler 存活）。

### 5.3 `llama_model_quantize_params` （397–412 行）

量化参数：线程数、目标 ftype、输出张量/嵌入张量类型、是否允许重量化、是否量化 output.weight、only_copy、pure（全部用默认类型）、keep_split（保持分片数）、dry_run（只算体积）、imatrix（重要性矩阵）、kv_overrides、tt_overrides、`prune_layers`（裁剪指定层）。

### 5.4 其他

- `llama_sampler_seq_config`（seq_id → sampler 绑定，用于按序列后端采样）。
- `llama_sampler_chain_params`：仅 `no_perf`。

## 6. API 功能模块逐一分析

### 6.1 初始化与线程池 （432–456 行）

- `llama_backend_init()` / `llama_backend_free()`：程序级一次性初始化/清理（free 目前仅 MPI 使用）。
- `llama_numa_init(numa)`：可选 NUMA 优化。
- `llama_attach_threadpool()` / `llama_detach_threadpool()`：显式传入 ggml 线程池（生成线程池 + 批处理线程池），否则 ggml 自动创建。
- 默认参数获取：`llama_model_default_params`、`llama_context_default_params`、`llama_sampler_chain_default_params`、`llama_model_quantize_default_params`。

### 6.2 模型加载 / 保存 / 构建 （458–500 行）

- `llama_model_init_from_user()`：从 GGUF 元数据 + 用户提供的张量数据回调在内存中构建模型（张量默认 F32，可通过同名异类型张量覆盖）。
- `llama_model_load_from_file()`：标准加载；分片文件需遵循 `<name>-%05d-of-%05d.gguf` 命名。
- `llama_model_load_from_file_ptr()`：从已打开的 `FILE*` 加载。
- `llama_model_load_from_splits()`：自定义命名分片，路径须按顺序传入。
- `llama_model_save_to_file()` / `llama_model_free()`。
- 废弃别名：`llama_load_model_from_file` → `llama_model_load_from_file`；`llama_free_model` → `llama_model_free`。

### 6.3 上下文创建与参数自适应 （502–533 行）

- `llama_init_from_model(model, cparams)`：由模型创建推理上下文（废弃名 `llama_new_context_with_model`）；`llama_free()` 释放。
- `llama_params_fit()`：**自动适配**——在设备显存受限时调整 mparams/cparams（假设系统内存无限），可设每设备内存余量 `margins`、最小上下文 `n_ctx_min`、日志级别；非线程安全（修改全局 logger 状态）；只修改与默认值一致的参数，上下文仅当原值为 0 时调整。

### 6.4 环境能力查询 （535–544 行）

- `llama_time_us()`：高精度时间。
- `llama_max_devices()`、`llama_max_parallel_sequences()`、`llama_max_tensor_buft_overrides()`：返回数组容量上限（用于分配 tensor_split / overrides 缓冲）。
- `llama_supports_mmap/mlock/gpu_offload/rpc()`：运行时能力探测。

### 6.5 上下文/模型/词表属性 （546–650 行）

- 上下文实际参数查询（推荐创建后回查，实际值可能与请求值不同）：`llama_n_ctx`、`llama_n_ctx_seq`、`llama_n_batch`、`llama_n_ubatch`、`llama_n_seq_max`。
- `llama_get_model()`、`llama_get_memory()`（获取记忆句柄）、`llama_pooling_type()`。
- 模型几何信息（新命名）：`llama_model_n_ctx_train`、`llama_model_n_embd`、`n_embd_inp`、`n_embd_out`、`n_layer`、`n_head`、`n_head_kv`、`n_swa`、`llama_model_rope_freq_scale_train`、`llama_model_rope_type`。旧短名 `llama_n_embd/n_layer/n_head/n_ctx_train/n_vocab` 均已废弃。
- 分类模型：`llama_model_n_cls_out()`、`llama_model_cls_label()`。
- 词表：`llama_model_get_vocab()`、`llama_vocab_type()`、`llama_vocab_n_tokens()`。
- GGUF 元数据访问（统一约定：返回字符串长度，失败返回 -1 并清空缓冲；需多留 1 字节给 NUL；不支持 GGUF 数组值）：`llama_model_meta_val_str(_by_index)`、`llama_model_meta_count`、`llama_model_meta_key(_by_index/_str)`、`llama_model_desc()`、`llama_model_size()`、`llama_model_n_params()`、`llama_model_chat_template()`。
- 模型形态判定：`llama_model_has_encoder/has_decoder`、`llama_model_decoder_start_token`、`llama_model_is_recurrent`（Mamba/RWKV）、`llama_model_is_hybrid`（Jamba/Granite）、`llama_model_is_diffusion`（LLaDA/Dream）。
- `llama_model_quantize()`：文件级量化，成功返回 0。

### 6.6 适配器（LoRA / 控制向量）（651–708 行）

- `llama_adapter_lora_init(model, path)`：加载 LoRA；适配器生命周期依附于模型。
- 适配器 GGUF 元数据访问：与模型同约定（val_str / count / key_by_index / val_str_by_index）。
- `llama_adapter_lora_free()`：手动释放（否则随模型释放）。
- aLoRA（调用式 LoRA）：`llama_adapter_get_alora_n_invocation_tokens()` / `llama_adapter_get_alora_invocation_tokens()`。
- `llama_set_adapters_lora(ctx, adapters, n, scales)`：在上下文上启用/切换一组 LoRA（相同则不变更）。
- `llama_set_adapter_cvec()`：施加控制向量（按层范围 il_start/il_end，数据按 layer 1 起 n_embd×n_layers 排布）；传 NULL 清除。

### 6.7 记忆（Memory / KV Cache）管理 （710–782 行）

通过 `llama_memory_t` 句柄操作统一抽象的 KV/递归缓存：

- `llama_memory_clear(mem, data)`：清空（data=true 连数据缓冲一起清）。
- **序列算术**（区间约定统一：`p0<0` 表示从头，`p1<0` 表示到末尾）： `llama_memory_seq_rm`（删除 [p0,p1) 内的 token；整段序列删除永不失败）、`seq_cp`（复制到另一序列）、`seq_keep`（只保留某序列）、`seq_add`（位置整体平移 delta）、`seq_div`（位置整除 d）。
- `llama_memory_seq_pos_min/pos_max()`：序列位置上下界（SWA 缓存下 pos_min 通常非 0；[pos_min,pos_max] 区间保证完整；空序列返回 -1）。
- `llama_memory_can_shift()`：缓存是否支持位置平移。

### 6.8 状态 / 会话保存恢复 （784–907 行）

- 整上下文状态：`llama_state_get_size / get_data / set_data`（内存字节流；get_size 仅在保存时使用，恢复时可能偏小）。
- 会话文件：`llama_state_load_file / save_file`（含 token 序列校验，magic `ggsn` v9）。
- 单序列状态：`llama_state_seq_get_size / get_data / set_data / save_file / load_file`，以及带 `llama_state_seq_flags` 的 _ext 版本。
- 兼容宏：`LLAMA_STATE_SEQ_FLAGS_SWA_ONLY` 与 `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` 同值（=1），用于仅处理部分状态（SWA KV / Mamba 递归缓存）。
- 全部旧名（`llama_get_state_size`、`llama_copy_state_data`、`llama_set_state_data`、`llama_load_session_file`、`llama_save_session_file`）已标废弃。

### 6.9 Batch 与解码 （909–1029 行）

- **batch 工具**：`llama_batch_get_one()`（单序列便捷批，seq 固定 0，注释不推荐使用）、`llama_batch_init(n_tokens, embd, n_seq_max)`（堆分配，成员全部不初始化）、`llama_batch_free()`。
- **`llama_encode(ctx, batch)`**：处理 batch，**不使用 KV cache**；编码-解码模型中走编码器，编码器输出缓存供解码器交叉注意力使用。0 成功，<0 失败且状态回滚。
- **`llama_decode(ctx, batch)`**：核心推理入口，要求上下文有 memory；返回码语义丰富：

  | 返回值 | 含义 |
  | --- | --- |
  | 0 | 成功 |
  | 1 | 找不到 KV 槽位（减小 batch 或增大上下文） |
  | 2 | 被中止（已处理的 ubatch 保留在 memory 中） |
  | -1 | batch 无效 |
  | <-1 | 致命错误（已处理 ubatch 保留，需用 seq_pos_min/max 查询状态） |

- **运行时控制**：`llama_set_n_threads()` / `llama_n_threads(_batch)()`、`llama_set_embeddings()`、`llama_set_causal_attn()`、`llama_set_warmup()`（预热：激活全部张量以载入/缓存权重）、`llama_set_abort_callback()`、`llama_synchronize()`（等待计算完成，一般无需手动调用）。
- **输出生成**： `llama_get_logits()`（全量输出，行=有输出标志的 token，列=n_vocab；计划废弃）、 `llama_get_logits_ith()`（第 i 个输出，负索引从尾部倒数，无效 id 返回 NULL）、 `llama_get_embeddings()`（仅 pooling=NONE 或生成模型）、 `llama_get_embeddings_ith()`、 `llama_get_embeddings_seq()`（按序列池化结果；RANK 模式返回 n_cls_out 个排序分）。

### 6.10 后端采样结果查询 [EXPERIMENTAL] （1031–1055 行）

当上下文以 `llama_context_params.samplers` 初始化（采样嵌入计算图）时使用：`llama_get_sampled_token_ith()`、`llama_get_sampled_probs_ith(+count)`、`llama_get_sampled_logits_ith(+count)`、`llama_get_sampled_candidates_ith(+count)`（candidates 用于把 prob/logit 下标映射回 token id）。

### 6.11 词表查询 （1057–1116 行）

- 单 token：`llama_vocab_get_text / get_score / get_attr`、`llama_vocab_is_eog`（EOG：EOS/EOT 等）、`llama_vocab_is_control`。
- 特殊 token：`bos / eos / eot / sep / nl / pad / mask`。
- 自动添加控制：`llama_vocab_get_add_bos / add_eos / add_sep`。
- FIM（fill-in-middle 代码补全）token：`fim_pre / fim_suf / fim_mid / fim_pad / fim_rep / fim_sep`。
- 旧 `llama_token_*` 系列全部废弃（含 `llama_vocab_cls`——CLS 等价于 BOS）。

### 6.12 分词 / 反分词 （1118–1167 行）

- **`llama_tokenize()`**：text→tokens，线程安全。参数 `add_special`（按模型配置加 BOS/EOS）、`parse_special`（是否解析特殊/控制 token，不会自动加前导空格）。成功返回 token 数（≤ n_tokens_max）；失败返回负数=所需 token 数；溢出 int32 上限时返回 INT32_MIN。
- **`llama_token_to_piece()`**：单 token→文本片段（不写 NUL；`lstrip` 可跳过前导空格；`special` 控制是否渲染特殊 token）。
- **`llama_detokenize()`**：tokens→text，`remove_special` / `unparse_special` 与 tokenize 对称；失败返回负数=所需字节数。

### 6.13 聊天模板 （1169–1192 行）

- **`llama_chat_apply_template()`**：将消息数组格式化为 prompt（仿 HF apply_chat_template，但**非 Jinja 解析器**，仅支持内置模板集合）；`add_ass` 决定是否追加 assistant 起始 token；返回完整字节数（超出缓冲需重新分配再调用；建议分配 2×消息总字符）。
- `llama_chat_builtin_templates()`：列出内置模板名。

### 6.14 采样 API （1194–1491 行）

**面向对象式插件设计**：`llama_sampler` = 接口表 `llama_sampler_i`（name/accept/apply/reset/clone/free，均除 apply 外可为 NULL）+ 上下文指针，用户可完全自定义采样器。

- **接口镜像函数**：`llama_sampler_init(iface, ctx)`、`llama_sampler_name / accept / apply / reset / clone / free`。注意：已加入链的 sampler 不得单独 free（链持有所有权）。
- **采样链**：`llama_sampler_chain_init / chain_add / chain_get / chain_n / chain_remove`（remove 后所有权归还调用方；chain_get 传 -1 返回链本身，可用于类型检测）。
- **内置采样器**（均以 `llama_sampler_init_*` 工厂函数创建）：

  | 采样器 | 说明 |
  | --- | --- |
  | `greedy` | 贪心（argmax） |
  | `dist(seed)` | 按分布随机采样，`LLAMA_DEFAULT_SEED` = 随机种子 |
  | `top_k / top_p / min_p / typical / top_n_sigma` | 截断类（含对应论文引用） |
  | `temp / temp_ext` | 温度缩放 / 动态温度（熵自适应） |
  | `xtc` | XTC 排除式采样 |
  | `mirostat / mirostat_v2` | 困惑度自适应采样 v1/v2 |
  | `grammar / grammar_lazy_patterns` | GBNF 语法约束（懒触发版支持触发词/模式/触发 token）；旧 lazy 版已废弃 |
  | `penalties` | 重复/频率/存在惩罚 |
  | `dry` | DRY 重复惩罚 |
  | `adaptive_p` | 自适应概率目标采样（EMA 机制，选择型采样器，须置于链尾） |
  | `logit_bias` | logit 偏置 |
  | `infill` | infilling 专用（置于 top_k+top_p 之后，优先 EOG、合并同前缀概率） |

- **便捷入口**：`llama_sampler_sample(smpl, ctx, idx)`——从第 idx 个输出取 logits→应用采样链→accept→返回 token；`llama_sampler_get_seed()` 返回有效种子。
- **[EXPERIMENTAL] 后端采样接口**：`llama_sampler_i` 中的 `backend_init / backend_apply / backend_accept / backend_set_input`，将采样逻辑编译进 ggml 计算图（GPU 上执行）；另有 `llama_set_sampler(ctx, seq_id, smpl)` 按序列挂载采样器，及 `llama_sampler_data`（logits/probs/sampled/candidates 张量组）。
- 头文件内附完整**示例用法注释**（链构建 + 解码循环 + free，1197–1227 行）。

### 6.15 模型分片路径工具 （1493–1505 行）

- `llama_split_path()`：由前缀 + 序号/总数生成分片文件名。
- `llama_split_prefix()`：仅当序号/总数匹配时从分片路径提取前缀。

### 6.16 日志 （1507–1514 行）

- `llama_print_system_info()`：打印编译/系统信息。
- `llama_log_set / llama_log_get`：全局日志回调（未设置则输出 stderr；**非线程安全**）。

### 6.17 性能统计 （1516–1550 行）

- `llama_perf_context_data`：加载/prompt 处理/生成耗时（ms）及 token 计数、图重用次数；`llama_perf_context / _print / _reset`。
- `llama_perf_sampler_data`：采样耗时与次数；`llama_perf_sampler / _print / _reset`（仅对链式采样器有效）。
- `llama_memory_breakdown_print()`：按设备打印内存占用分解。
- 注释提示：第三方应用建议自行测量，这些接口主要服务于官方示例/工具。

### 6.18 训练（fine-tuning）（1552–1583 行）

- `llama_opt_param_filter`：筛选可训练张量的回调；`llama_opt_param_filter_all()` 全选实现。
- `llama_opt_params`：训练上下文尺寸、参数筛选回调、优化器参数回调（复用 ggml-opt）、优化器类型。
- `llama_opt_init()` + `llama_opt_epoch()`：基于 ggml-opt 的训练循环（数据集、训练/评估结果、epoch 回调）。

## 7. 废弃（Deprecated）与向后兼容机制

该头文件体现了一次大规模 API 重命名（"llama_verb → llama_object_verb" 风格），旧名通过 `DEPRECATED()` 宏保留编译期警告与迁移提示：

| 旧 API | 新 API | 类别 |
| --- | --- | --- |
| `llama_load_model_from_file` | `llama_model_load_from_file` | 模型加载 |
| `llama_free_model` | `llama_model_free` | 模型释放 |
| `llama_new_context_with_model` | `llama_init_from_model` | 上下文创建 |
| `llama_n_ctx_train / n_embd / n_layer / n_head / n_vocab` | `llama_model_* / llama_vocab_n_tokens` | 属性查询 |
| `llama_token_get_text/score/attr、llama_token_bos/eos/...` | `llama_vocab_*` | 词表 |
| `llama_get_state_size / copy_state_data / set_state_data / load/save_session_file` | `llama_state_*` | 状态 |
| `llama_sampler_init_grammar_lazy` | `llama_sampler_init_grammar_lazy_patterns` | 采样 |

此外还有显式标注 **[EXPERIMENTAL]** 的接口（KV 量化、后端采样、`llama_params_fit` 相关缓冲约定等），提示可能变化；`defrag_thold` 已标 [DEPRECATED]。

## 8. 典型使用生命周期

```cpp
llama_backend_init();                          // 1. 初始化后端
struct llama_model_params mp = llama_model_default_params();
struct llama_model * model = llama_model_load_from_file("model.gguf", mp);   // 2. 加载模型
struct llama_context_params cp = llama_context_default_params();
struct llama_context * ctx = llama_init_from_model(model, cp);              // 3. 创建上下文

const llama_vocab * vocab = llama_model_get_vocab(model);                   // 4. 词表与分词
llama_tokenize(vocab, prompt, len, tokens, n_max, true, true);

struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);                // 5. 组装 batch
/* 填充 token / pos / logits 标志 */
llama_decode(ctx, batch);                                                   // 6. 推理

struct llama_sampler * chain = llama_sampler_chain_init(sparams);           // 7. 采样链
llama_sampler_chain_add(chain, llama_sampler_init_top_k(50));
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
llama_token id = llama_sampler_sample(chain, ctx, -1);                      // 8. 采样

llama_get_logits_ith(ctx, -1);  /* ... 反分词、输出 ... */

llama_sampler_free(chain);                                                  // 9. 释放
llama_batch_free(batch);
llama_free(ctx);
llama_model_free(model);
llama_backend_free();
```

## 9. 总结与观察

| 维度 | 观察 |
| --- | --- |
| 接口规模 | 约 220+ 个导出函数、20+ 枚举/结构体，覆盖从模型文件管理到训练的全链路。 |
| 抽象层次 | 完全建立在 GGML 之上（张量、后端、调度、线程池、优化器、GGUF），本身只做"模型级"编排。 |
| 内存/状态模型 | 以 `llama_memory_t` 统一抽象 KV/递归/SWA 缓存；状态可按上下文或单序列粒度导出/恢复，支持会话文件持久化。 |
| 多序列与并行 | `llama_batch` 原生支持一 token 属多序列、最多 `llama_max_parallel_sequences()` 路并行；多 GPU 支持 layer/row/tensor 级切分。 |
| 演进趋势 | ① 采样正从主机端 `llama_token_data_array` 迁移到**后端（计算图内）采样**；② 旧 API 正被对象化新名替代；③ 对 encoder-decoder、递归、混合、扩散模型的统一支持日益明显（`llama_encode/decode` 分离即是标志）。 |
| 稳定性提示 | API 分四级：稳定（默认）、[EXPERIMENTAL]（可变）、DEPRECATED（保留兼容）、移除（注释保留的 ftype）。 |
| 易用性设计 | 默认参数工厂函数、能力探测函数、`llama_params_fit` 自动显存适配、头文件内采样示例注释，显著降低接入门槛。 |

本报告基于 `include/llama.h`（1589 行）源码静态分析生成；行号引用对应源文件当前版本。

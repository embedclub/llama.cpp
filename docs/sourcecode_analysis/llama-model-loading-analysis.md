# llama.cpp 模型加载流程分析报告

## 1. 概述

本文以 `tools/cli/cli.cpp` 为入口，对 llama.cpp 的模型加载（Model Loading）过程进行完整追踪与分析。llama.cpp 是一个用 C/C++ 实现的 LLM 推理引擎，支持 GGUF 格式的模型文件。模型加载涉及参数解析、后端初始化、GGUF 文件解析、元数据读取、超参数加载、词表加载、张量（权重）分配与加载等核心步骤。

---

## 2. 整体调用链

```
main() [tools/cli/cli.cpp:346]
  │
  ├─ common_params_parse()          [common/common.cpp]       // 解析 CLI 参数
  ├─ llama_backend_init()           [src/llama.cpp]           // 初始化后端
  ├─ llama_numa_init()              [src/llama.cpp]           // NUMA 初始化
  │
  └─ ctx_cli.ctx_server.load_model(params)  [tools/server/server-context.cpp:3074]
       │
       └─ server_context_impl::load_model(params)  [tools/server/server-context.cpp:745]
            │
            ├─ common_init_from_params(params)   [common/common.cpp:1268]
            │    │
            │    └─ common_init_result::common_init_result(params) [common/common.cpp:1143]
            │         │
            │         ├─ llama_model_load_from_file(path, mparams)  [src/llama.cpp:1163]
            │         │    │
            │         │    └─ llama_model_load_from_file_impl(...)  [src/llama.cpp:931]
            │         │         │
            │         │         ├─ 设备枚举与选择（GPU/RPC/CPU）
            │         │         │
            │         │         └─ llama_model_load(metadata, ..., model, params) [src/llama.cpp:875]
            │         │              │
            │         │              ├─ llama_model_loader 构造函数 [llama-model-loader.cpp:510]
            │         │              │    ├─ gguf_init_from_file() — 解析 GGUF 文件
            │         │              │    ├─ 加载 split 文件（多分片模型）
            │         │              │    ├─ 构建 weights_map（张量名 → 权重索引）
            │         │              │    └─ 打印元数据信息
            │         │              │
            │         │              ├─ model.load_arch(ml)   [llama-model.cpp:683]
            │         │              ├─ model.load_hparams(ml) [llama-model.cpp:693]
            │         │              ├─ model.load_vocab(ml)  [llama-model.cpp:2952]
            │         │              ├─ model.load_stats(ml)  [llama-model.cpp:678]
            │         │              └─ model.load_tensors(ml) [llama-model.cpp:2958]
            │         │                   │
            │         │                   ├─ 构建 CPU/GPU buffer type 列表
            │         │                   ├─ 计算 layer 到设备的拆分策略
            │         │                   ├─ 按架构类型创建各权重张量（create_tensor）
            │         │                   └─ load_all_data() [llama-model-loader.cpp:1399]
            │         │                        ├─ mmap 方式：文件映射 + GPU 分配
            │         │                        └─ 非 mmap 方式：读取 + 异步/同步上传 GPU
            │         │
            │         └─ llama_init_from_model(model, cparams) [llama-context.cpp:2926]
            │              │
            │              └─ 创建 llama_context，初始化 KV cache、计算缓冲区
            │
            ├─ 加载 draft model（推测解码）
            ├─ 加载 mmproj（多模态投影模型）
            └─ 初始化 slot 状态（推理槽位）
```

---

## 3. 第一阶段：CLI 入口与参数准备

### 3.1 `tools/cli/cli.cpp::main()` (第 346 行)

CLI 的入口函数执行以下关键操作：

1. **参数解析**（第 353 行）：`common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)` 将命令行参数填充到 `common_params` 结构体，包括模型路径、上下文大小、GPU 层数、是否使用 mmap 等。

2. **后端初始化**（第 366 行）：`llama_backend_init()` 初始化 ggml 后端系统，注册所有可用的后端（CPU、CUDA、Vulkan、Metal、RPC 等）。

3. **模型加载**（第 392 行）：`ctx_cli.ctx_server.load_model(params)` 委托给 `server_context::load_model()`。

4. **推理循环**（第 470 行起）：交互式读取用户输入，构建 chat 消息，并通过 `generate_completion()` 提交推理任务。

### 3.2 `common_params` 与 `common_params_parse`

`common_params` 定义在 `common/common.h`，是一个大型结构体，包含所有可配置参数。`common_params_parse` 使用 `arg.h` 中的参数解析框架进行声明式解析，该方法支持自动生成帮助信息、参数校验和类型转换。

关键模型相关参数：
- `model.path` — GGUF 模型文件路径
- `model.n_gpu_layers` — 卸载到 GPU 的层数
- `model.split_mode` — 张量并行拆分模式
- `model.use_mmap` — 是否使用内存映射
- `model.use_mlock` — 是否锁定内存

---

## 4. 第二阶段：Server 上下文中的模型加载

### 4.1 `server_context::load_model()` (第 3074 行)

这是一个薄包装，委托给 `server_context_impl::load_model()`。

### 4.2 `server_context_impl::load_model()` (第 745 行)

核心实现：

```cpp
bool load_model(common_params & params) {
    params_base = params;
    llama_init = common_init_from_params(params_base);       // 核心加载入口
    model = llama_init->model();                              // 获取模型指针
    ctx   = llama_init->context();                            // 获取上下文指针
    vocab = llama_model_get_vocab(model);                     // 获取词表
    n_ctx = llama_n_ctx(ctx);                                 // 获取上下文长度
    ...
}
```

之后依次加载：
1. **Draft model**（第 768 行）：若启用了推测解码，通过 `llama_model_load_from_file()` 加载 draft 模型。
2. **Multimodal projector**（第 804 行）：若指定了 `--mmproj`，通过 `mtmd_init_from_file()` 加载多模态投影模型。
3. **Slot 初始化**（后续代码）：为并行推理创建 slot。

---

## 5. 第三阶段：`common_init_from_params` — 高层封装

### 5.1 `common_init_from_params()` (common/common.cpp:1268)

将 `common_params` 转换为 `llama_model_params` 和 `llama_context_params`，调用 `llama_model_load_from_file()` 加载模型，再调用 `llama_init_from_model()` 创建上下文。返回值 `common_init_result` 封装了 `llama_model*` 和 `llama_context*`。

### 5.2 `common_init_result` 构造函数 (common/common.cpp:1143)

依次：
1. 调用 `llama_model_load_from_file(path, mparams)` 加载模型
2. 调用 `llama_init_from_model(model, cparams)` 创建上下文
3. 应用各种参数覆写

---

## 6. 第四阶段：公共 API — `llama_model_load_from_file`

### 6.1 `llama_model_load_from_file()` (src/llama.cpp:1163)

```cpp
struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, nullptr, params);
}
```

简单的包装函数，给 `path_model` 参数，其他数据源（`metadata`, `file`）置空，委托给 `llama_model_load_from_file_impl`。

### 6.2 其他公共 API 变体

- `llama_model_load_from_splits()` — 从分片 GGUF 文件加载
- `llama_model_load_from_file_ptr()` — 从 `FILE*` 指针加载
- `llama_model_init_from_user()` — 从用户提供的 GGUF 元数据 + 回调加载（用于自定义数据源）
- `llama_load_model_from_file()` — 已弃用，同 `llama_model_load_from_file`

---

## 7. 第五阶段：`llama_model_load_from_file_impl` — 核心加载编排

位于 `src/llama.cpp:931`，是整个模型加载过程的**总调度器**。

### 7.1 输入源校验（第 940 行）

确保 `metadata`, `path_model`, `file` 三者中恰好一个被定义。

### 7.2 进度回调设置（第 962 行）

若用户未提供 `progress_callback`，使用默认的逐点打印回调（打印 `.` 表示进度）。

### 7.3 设备枚举与选择（第 982–1126 行）

这是设备发现的复杂逻辑：

1. **用户指定设备**（第 982 行）：若 `params.devices` 非空，使用用户指定的设备列表。
   - `LLAMA_SPLIT_MODE_TENSOR` 模式：创建一个 Meta 设备，内部聚合多个物理设备做张量并行。
   - 其他模式：直接使用设备列表。

2. **默认设备发现**（第 1007 行）：枚举所有后端设备，分类为：
   - `gpus` — 独立 GPU（CUDA、Vulkan、Metal 后端）
   - `igpus` — 集成 GPU
   - `rpc_servers` — RPC 远程设备（协议为 RPC 后端）
   
   添加顺序：RPC 服务器优先 → GPU → 集成 GPU（仅在无其他设备时）。

3. **单 GPU 模式**（第 1104 行）：`LLAMA_SPLIT_MODE_NONE` 时只保留 `main_gpu` 指定的设备。

### 7.4 调用 `llama_model_load()`（第 1128 行）

将控制权交给核心加载函数。

---

## 8. 第六阶段：`llama_model_load` — 加载编排核心

位于 `src/llama.cpp:875`，执行模型加载的六步流程。

### 8.1 `llama_model_loader` 构造函数 (llama-model-loader.cpp:510)

这是 GGUF 解析的核心类：

1. **KV Overrides 处理**（第 529 行）：应用用户指定的 `kv_overrides`，覆盖 GGUF 中的元数据值。

2. **GGUF 文件初始化**（第 537 行）：
   - 调用 `gguf_init_from_file()` 解析 GGUF 文件到 `gguf_context`
   - 从元数据读取 `general.architecture` 字段得到架构名称（如 `"llama"`, `"qwen2"`）
   - 将架构名称映射到 `llm_arch` 枚举（通过 `llm_arch_from_string()`）

3. **文件分片处理**（第 584 行）：
   - 读取 `split.count` 元数据判断是否为分片模型
   - 若 `n_split > 1`，自动查找或验证分片文件列表
   - 遍历每个分片的张量，统一构建 `weights_map`（张量名 → 文件索引 + 偏移量）

4. **权重映射构建**（第 574 行）：遍历 GGML 上下文中的每个张量，记录其所在的文件、偏移量、元数据到 `weights_map`。

5. **信息打印**（第 699 行起）：打印 GGUF 文件版本、张量类型、文件大小、元数据 KV 对等。

### 8.2 `model.load_arch(ml)` (llama-model.cpp:683)

从 `llama_model_loader` 读取 `general.architecture` KV 对，通过 `llm_arch_from_string()` 转换为 `llm_arch` 枚举值，并验证该架构是否受支持。

### 8.3 `model.load_hparams(ml)` (llama-model.cpp:693)

读取所有模型超参数，包括：
- `n_ctx_train` — 训练时的上下文长度
- `n_embd` — 嵌入维度
- `n_layer` — Transformer 层数
- `n_head`, `n_head_kv` — 注意力头数
- `n_ff` — FFN 中间维度
- `n_expert`, `n_expert_used` — MoE 专家数
- `rope_freq_base`, `rope_scaling` — RoPE 参数
- `use_swa` — 滑动窗口注意力
- `vocab_size`, `pooling_type` 等

每读取一个参数后，**可能会有架构特定的处理逻辑**，如 Llama 与 Qwen2 的 rope 参数推导路径不同。

### 8.4 `model.load_vocab(ml)` (llama-model.cpp:2952)

委托给 `llama_vocab::load()`（`src/llama-vocab.cpp:3531`），处理：
1. **Tokenizer 类型检测**（BPE、SentencePiece、WordPiece、MGM、Unigram 等）
2. **Token 列表读取**：token 字符串、分数、类型
3. **特殊 Token 识别**：BOS、EOS、SEP、PAD、CLS、MASK 等
4. **Token 合并规则**（BPE 模型的 `merges`）
5. **添加 BOS token 标志**

### 8.5 `model.load_stats(ml)` (llama-model.cpp:678)

统计 tensor 数量和字节数。

### 8.6 `model.load_tensors(ml)` (llama-model.cpp:2958)

权重张量的核心加载过程：

1. **Buffer Type 列表构建**（第 2965 行）：为每个设备构建 CPU 和 GPU 的 buffer type 列表，使用 `buft_list_t` 结构表示不同类型的候选 buffer（F32、F16、BF16、Q8_0 等）。

2. **Layer 拆分计算**（第 2999 行）：根据 GPU 空闲内存或用户指定的 `tensor_split` 比例，确定每层张量分配到哪个设备。

3. **张量创建循环**（第 3015 行起）：针对每种架构类型（switch on `model.arch`），调用 `ml.create_tensor()` 创建每个权重张量：
   - Token 嵌入（`token_embd`）
   - 输出层：`output_norm`, `output`
   - 每层：`attn_norm`, `wq`, `wk`, `wv`, `wo`, `ffn_norm`, `ffn_gate`, `ffn_down`, `ffn_up`
   - MoE：`ffn_gate_inp`, `ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps`
   - RoPE：`rope_freqs`

   支持超过 100 种模型架构，每种架构的 tensor 布局在 `load_tensors()` 中有独立的 case 分支。

4. **create_tensor() 方法**（llama-model-loader.cpp:1045）：
   - 为每个 buffer type 创建独立的 `ggml_context`
   - 调用 `buft_for_tensor()` 选择最佳 buffer type：
     - 检查 tensor 覆写规则（`tensor_buft_overrides`）
     - 通过 `select_weight_buft()` 根据 op 类型选择最优 buffer（GPU 优先，CPU 兜底）
   - 创建 `ggml_tensor` 并分配内存

5. **load_all_data() 调用** — 见下一节。

---

## 9. 第七阶段：权重数据加载 — `load_all_data`

位于 `llama-model-loader.cpp:1399`，是实际从磁盘读取权重数据到内存/显存的过程。

### 9.1 `init_mappings()` (第 1326 行)

若启用 `use_mmap`，为每个分片文件创建 `llama_mmap` 映射，可选择预取和 mlock。

### 9.2 异步上传机制（第 1416 行）

对于非 mmap 场景，使用 4 个 staging buffer（每个 1MB），通过 pinned memory + GPU events 实现异步上传。

### 9.3 数据加载循环（第 1514 行）

遍历所有张量，按三种路径加载：

**路径 A：mmap 方式（第 1529 行）**
```
mapping->addr() + weight->offs → 直接指针
若 GPU buffer 已分配：ggml_backend_tensor_alloc(buf_mmap, cur, data)
若已存在 data：ggml_backend_tensor_set(cur, data, 0, n_size)
```

- 张量 data 指针直接指向 mmap 地址（CPU 张量）
- 或通过 `tensor_alloc` 在 GPU buffer 内分配次区域
- mmap 利用了操作系统的按需分页（demand paging）

**路径 B：非 mmap + CPU 张量（第 1560 行）**
```
file->seek(weight->offs) → file->read_raw(cur->data, n_size)
```

**路径 C：非 mmap + GPU 张量（第 1568 行）**

优先使用异步上传（若支持）：
```
读对齐块到 pinned memory → ggml_backend_tensor_set_async() → event 同步
```

回退方案：
```
读完整块到临时 buffer → ggml_backend_tensor_set() 同步上传到 GPU
```

### 9.4 Tensor 数据验证（第 1537、1563、1627 行）

若 `check_tensors=true`，通过 `ggml_validate_row_data()` 异步验证每个张量的行数据格式正确性。

### 9.5 后处理（第 1637–1672 行）

- 同步并释放异步上传资源（events、host buffers、backend）
- 检查验证结果
- 卸载 mmap 未使用的文件片段（`unmap_fragment`）

---

## 10. 第八阶段：`llama_init_from_model` — 上下文创建

位于 `llama-context.cpp:2926`，从已加载的 `llama_model` 创建 `llama_context`：

1. 参数验证：`n_batch`、`n_ctx`、`flash_attn`、`pooling` 等
2. KV cache 初始化：根据 `n_ctx`、`n_layer`、`n_head_kv` 等分配 key/value cache
3. 计算缓冲区初始化：为推理分配临时 tensor 缓冲区
4. 设置采样器默认状态

---

## 11. 关键数据结构与类

| 类/结构体 | 文件 | 作用 |
|-----------|------|------|
| `llama_model` | `src/llama-model.h` | 模型对象，持有所有权重张量、超参数、词表、设备列表 |
| `llama_context` | `src/llama-context.h` | 推理上下文，持有 KV cache、batch、计算图等运行时状态 |
| `llama_model_params` | `include/llama.h` | 模型加载参数（GPU 层数、mmap、split_mode 等） |
| `llama_model_loader` | `src/llama-model-loader.h` | GGUF 文件加载器，管理文件映射、张量创建、数据加载 |
| `llama_file` | `src/llama-model-loader.h` | 文件 I/O 封装，支持对齐读取和 direct I/O |
| `llama_mmap` | `src/llama-mmap.h` | 内存映射文件封装（平台适配） |
| `llama_tensor_weight` | `src/llama-model-loader.h` | 张量权重索引（文件、偏移、元数据） |
| `llama_device` | `src/llama-model.h` | 设备封装（是否 meta device + backend device 指针） |
| `common_params` | `common/common.h` | 高层参数结构体，CLI/server 共用 |
| `common_init_result` | `common/common.h` | 加载结果封装（model + context 对） |
| `llama_vocab` | `src/llama-vocab.h` | 词表对象，支持多种 tokenizer 类型 |

---

## 12. 架构支持的扩展机制

llama.cpp 的模型加载是**高度可扩展**的：

1. **架构枚举**（`llama-arch.cpp`）：`LLM_ARCH_NAMES` 映射字符串名到 `llm_arch` 枚举。
2. **架构特定加载**（`llama-model.cpp`）：`load_tensors()` 中的 `switch(model.arch)` 分支处理每种架构的 tensor 布局。
3. **模型特定文件**（`src/models/`）：各架构的前向推理实现独立成文件，如 `llama.cpp`（Llama 架构）、`qwen3.cpp`、`gemma3.cpp`、`mamba.cpp` 等。

---

## 13. 关键设计决策与优化

### 13.1 内存映射（mmap）

- **默认启用**，通过操作系统虚拟内存按需加载权重，省去显式读取步骤
- mmap 下 GPU 张量通过 `ggml_backend_tensor_alloc()` 在 GPU buffer 内分配次区域，避免数据拷贝
- 加载完成后通过 `unmap_fragment()` 释放未使用的映射区域
- mmap 与 direct I/O 互斥（第 557 行）

### 13.2 异步 GPU 上传

- 非 mmap 场景下，使用 4 个环形缓冲区 + pinned memory 实现文件读取与 GPU 上传的流水线并行
- 这是针对 NVMe 硬盘优化的设计

### 13.3 设备发现与张量拆分

- 支持多种 GPU 后端：CUDA、Vulkan、Metal、SYCL
- 支持 RPC 远程 GPU
- `LLAMA_SPLIT_MODE_TENSOR` 通过 Meta Device 抽象多 GPU 张量并行
- `LLAMA_SPLIT_MODE_NONE` / `LLAMA_SPLIT_MODE_LAYER` 控制拆分策略

### 13.4 精度选择

- `select_weight_buft()` 根据张量的计算操作（MUL_MAT、ADD 等）选择最优 buffer type
- 支持 F32、F16、BF16、Q4_0、Q8_0 等多种量化格式

---

## 14. 涉及的源文件清单

| 文件路径 | 角色 |
|----------|------|
| `tools/cli/cli.cpp` | CLI 入口 |
| `tools/server/server-context.cpp` | Server 上下文（模型加载的调度层） |
| `common/common.cpp` | 公共参数处理 + `common_init_from_params` |
| `src/llama.cpp` | 公共 API + `llama_model_load` + `llama_model_load_from_file_impl` |
| `src/llama-model.cpp` | 模型类的 `load_arch/load_hparams/load_vocab/load_tensors` |
| `src/llama-model-loader.cpp` | GGUF 加载器、张量创建与数据加载 |
| `src/llama-context.cpp` | `llama_init_from_model` — 上下文创建 |
| `src/llama-vocab.cpp` | 词表加载 |
| `src/llama-arch.cpp` | 架构枚举与名称映射 |
| `include/llama.h` | 公共 API 声明 |
| `src/llama-model.h` | 模型对象定义 |
| `src/llama-model-loader.h` | 加载器定义 |
| `src/gguf.h` | GGUF 格式解析库 |

---

## 15. 总结

llama.cpp 的模型加载是一个**多层次、高度模块化**的过程：

1. **入口层**（`cli.cpp`）→ 参数解析
2. **调度层**（`server-context.cpp`, `common.cpp`）→ 高层参数转换与编排
3. **编排层**（`llama.cpp`）→ 设备枚举、六步加载流程
4. **解析层**（`llama-model-loader.cpp`）→ GGUF 格式解析、张量创建
5. **数据层**（`llama-model-loader.cpp`）→ mmap/异步读取权重数据

整个流程充分利用了操作系统的 mmap 机制和 GPU 的异步传输能力，在性能与内存效率之间取得了良好的平衡。架构支持通过枚举 + switch 的方式实现了超过 100 种模型的兼容，保持了良好的扩展性。

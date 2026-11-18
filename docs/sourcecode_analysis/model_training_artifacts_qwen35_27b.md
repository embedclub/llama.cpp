# 大模型训练的最终产物是什么

> 以 **Qwen3.5-27B** 为例，结合 Hugging Face 发布格式与 llama.cpp 的转换、量化、加载路径进行说明。

## 0. 结论先行

大模型训练结束后，最终交付的通常**不是一个单独的文件**，而是一个**模型目录（checkpoint / model repository）**。它至少包含四类信息：

1. **权重（weights）**：网络每一层的参数张量，通常保存为 `safetensors`，大模型会切分成多个 shard。
2. **结构配置（config）**：告诉推理框架如何把权重重新搭成网络，例如层数、隐藏维度、注意力头数、RoPE、词表大小等。
3. **分词与对话协议（tokenizer / chat template）**：负责文本与 token 的互相转换，以及把多轮对话、工具调用渲染成模型训练时见过的 prompt。
4. **运行时配置（generation / preprocessor）**：默认生成参数；多模态模型还包括图像、视频的预处理参数。

因此，训练产物提供的能力不是“直接双击运行”，而是：**可被推理框架加载、可被微调、可被量化、可被部署、可被复现实验**。

对于 Qwen3.5-27B，官方发布的是 **Hugging Face Transformers 格式的 post-trained checkpoint**，约 **55.6 GB**，由 **11 个 BF16 safetensors 分片**加配置、分词器、聊天模板等文件组成；再经过 `convert_hf_to_gguf.py` 和 `llama-quantize`，才能变成 llama.cpp 可直接运行的 GGUF 文件。

---

## 1. 训练产物与发布产物的区别

实际训练过程中，训练框架（Megatron、DeepSpeed、FSDP、TorchTitan 等）会周期性保存 checkpoint，其中可能包含：

- 模型参数；
- 优化器状态（Adam 的一阶、二阶矩）；
- 学习率调度器状态；
- 数据加载器与随机数状态；
- 分布式切分信息（TP/PP/DP/EP 的 shard 布局）。

这些文件用于**断点续训**，体积通常是模型权重的数倍，一般不会对外发布。

对外发布的“最终结果”通常会做一次**权重合并与格式转换**：

```text
训练框架原生分片 checkpoint
        │  合并 / 去掉优化器状态 / 统一 dtype / 转换为 HF 布局
        ▼
Hugging Face Transformers checkpoint
        │  可选：量化、蒸馏、转换
        ▼
GGUF / AWQ / GPTQ / FP8 / MLX / TensorRT-LLM 等部署格式
```

所以严格来说：

- **训练结束的原始产物**：包含权重和训练状态的 checkpoint 目录；
- **公开发布的最终产物**：只保留推理与再训练所需的模型权重和配置；
- **部署产物**：在发布权重基础上再做量化或图优化，例如 GGUF。

Qwen3.5-27B 的 Hugging Face 仓库属于第二类：可以直接被 Transformers、vLLM、SGLang、KTransformers 等加载，也可以作为微调起点。

---

## 2. 一个标准发布目录里有什么

以 Qwen3.5-27B 官方仓库为例，目录内容如下（大小为十进制 GB/MB）：

| 文件 | 大小 | 作用 |
|---|---:|---|
| `model.safetensors-00001-of-00011.safetensors` | 5.26 GB | 权重分片 1 |
| `model.safetensors-00002-of-00011.safetensors` | 5.35 GB | 权重分片 2 |
| ... | ... | ... |
| `model.safetensors-00010-of-00011.safetensors` | 5.35 GB | 权重分片 10 |
| `model.safetensors-00011-of-00011.safetensors` | 2.15 GB | 权重分片 11 |
| `model.safetensors.index.json` | 127 KB | 张量名到分片文件的映射表 |
| `config.json` | 4.0 KB | 网络结构与超参数 |
| `generation_config.json` | 244 B | 默认生成参数 |
| `tokenizer.json` | 12.2 MB | 完整分词器（词表、合并规则、特殊 token） |
| `tokenizer_config.json` | 16.3 KB | 分词器配置、特殊 token、chat template 关联 |
| `vocab.json` | 6.4 MB | BPE 词表 |
| `merges.txt` | 3.35 MB | BPE 合并规则 |
| `chat_template.jinja` | 7.6 KB | 对话、思考模式、工具调用的 Jinja 模板 |
| `preprocessor_config.json` | 390 B | 图像预处理参数 |
| `video_preprocessor_config.json` | 385 B | 视频预处理参数 |
| `LICENSE` | 11.1 KB | Apache 2.0 许可证 |
| `README.md` | 90.3 KB | 模型卡、用法、评测与限制 |

合计约 **55.6 GB**。

### 2.1 权重分片与索引

`safetensors` 是一种只存张量、不执行任意 Python 代码的格式，比旧的 `pytorch_model.bin` 更安全，加载也更快。

大模型无法放进单个文件，因此按固定大小切分为多个 shard。`model.safetensors.index.json` 中的 `weight_map` 记录每个张量在哪个分片，例如：

```json
{
  "metadata": { "total_size": 59600000000 },
  "weight_map": {
    "model.layers.0.self_attn.q_proj.weight": "model.safetensors-00001-of-00011.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model.safetensors-00001-of-00011.safetensors"
  }
}
```

推理框架先读索引，再按需加载分片，避免一次性把全部权重读入内存。**索引文件与分片缺一不可**；重命名分片时也必须同步修改索引。

### 2.2 `config.json`：权重的“施工图”

`safetensors` 只说明“有哪些张量、形状是多少”，并不说明“这些张量如何组成网络”。`config.json` 提供重建计算图所需的信息：

- `architectures`：例如 `Qwen3_5ForConditionalGeneration`；
- `model_type`：例如 `qwen3_5`；
- `text_config`：文本主干的层数、隐藏维度、注意力头数、FFN 维度、RoPE、词表大小等；
- `vision_config`：视觉编码器的层数、patch size、图像尺寸等；
- `torch_dtype`：训练/保存精度，Qwen3.5-27B 为 `bfloat16`；
- 量化或并行相关字段（如果存在）。

没有 `config.json`，即使权重完整，框架也不知道该如何实例化模型。

### 2.3 分词器与聊天模板：模型“会说话”的前提

权重负责计算，但文本进入网络前必须先变成 token。Qwen3.5 使用 BPE 分词器，相关文件分工如下：

- `tokenizer.json`：完整序列化分词器，包含词表、合并规则、预切分正则、特殊 token；
- `vocab.json` + `merges.txt`：传统 BPE 词表与合并规则；
- `tokenizer_config.json`：`bos/eos/pad`、是否添加特殊 token、消息角色格式等；
- `chat_template.jinja`：把 `system/user/assistant/tool` 消息渲染为模型训练时使用的格式。

`chat_template.jinja` 常被低估，但它直接决定：

- 多轮对话如何拼接；
- 是否默认开启 thinking 模式；
- 工具调用（tool calling）的 JSON/XML 结构；
- 生成提示如何结尾，例如是否追加 `<|im_start|>assistant`。

同样的权重，聊天模板用错，输出质量会明显下降，甚至出现“不会停”或“格式错乱”。

### 2.4 `generation_config.json`：默认解码策略

该文件保存默认生成参数，例如：

- `eos_token_id` / `pad_token_id`；
- `temperature`、`top_p`、`top_k`；
- `repetition_penalty`；
- 是否使用 thinking 模式等模型特有字段。

它不参与前向计算，但决定“怎么采样下一个 token”。不同推理框架对它的支持程度不同，生产环境通常仍会显式传入采样参数。

### 2.5 多模态配置与视觉权重

Qwen3.5-27B 是原生多模态模型，支持文本、图像和视频输入。因此产物中还有：

- `preprocessor_config.json`：图像 resize、归一化均值/方差、patch 尺寸等；
- `video_preprocessor_config.json`：视频抽帧与预处理参数；
- 视觉编码器与投影层权重：通常合并在同一组 safetensors 中。

在 llama.cpp 中，文本权重和视觉投影器会被拆成两个 GGUF：

- 主模型：`Qwen3.5-27B-*.gguf`；
- 多模态投影器：`mmproj-Qwen3.5-27B-*.gguf`。

运行时通过 `--mmproj` 把两者关联起来。

---

## 3. Qwen3.5-27B 的架构速览

Qwen3.5-27B 是 Qwen3.5 系列中的 **Dense（稠密）27B** 模型，不是 MoE。系列中的 35B-A3B、122B-A10B、397B-A17B 才是 MoE 版本，llama.cpp 中分别对应 `qwen35moe` 等架构。

| 项目 | 值 |
|---|---|
| 参数规模 | 27B 量级（实际权重约 27.8B，含视觉编码器与 MTP 头） |
| 架构 | 混合注意力：Gated DeltaNet（线性注意力）+ Gated Attention（全注意力） |
| 层数 | 64 |
| 隐藏维度 | 5120 |
| FFN 中间维度 | 17408 |
| 层布局 | 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |
| 线性注意力 | 48 个 V head，16 个 QK head，head dim 128 |
| 全注意力 | 24 个 Q head，4 个 KV head（GQA），head dim 256 |
| RoPE 维度 | 64 |
| 词表 | 248,320（padded） |
| 原生上下文 | 262,144 token |
| 扩展上下文 | 约 1,010,000 token（YaRN） |
| 多模态 | 文本 + 图像 + 视频输入 |
| 训练特性 | Multi-Token Prediction（MTP，多步） |
| 许可证 | Apache 2.0 |

### 3.1 为什么 27B 会占约 55.6 GB

BF16 每个参数占 2 字节：

```text
27.8 × 10^9 参数 × 2 字节 ≈ 55.6 × 10^9 字节 ≈ 55.6 GB
```

换算成 GiB 约 **51.8 GiB**。这解释了为什么官方仓库是 11 个约 5 GB 的分片，而不是一个 27 GB 的文件——27 GB 是 FP8/INT8 量级，不是 BF16。

官方还提供了 FP8 版本 `Qwen3.5-27B-FP8`，同样是 11 个分片，总文件体积约 **30.8 GB**，其中权重本体约 28 GB 量级，其余是缩放因子等辅助张量。

### 3.2 混合注意力对文件与运行时的影响

Qwen3.5 的 3:1 混合结构意味着：

- 每 4 层中有 3 层是 Gated DeltaNet：使用 1D 卷积和递归状态，不需要传统 KV Cache；
- 每 4 层中有 1 层是 Gated Attention：仍是标准 GQA + RoPE，需要 KV Cache。

因此运行时使用**混合内存**：递归状态缓存 + KV Cache 并存。按公开配置估算，27B 的全注意力层共 16 层，BF16 KV Cache 约为 **64 KB/token**，16K 上下文约 **1 GB**；这比“64 层全部是全注意力”的模型小得多，也是长上下文部署成本较低的原因之一。

---

## 4. 这些文件分别提供什么功能

| 能力 | 依赖文件 | 说明 |
|---|---|---|
| 重建网络并前向计算 | safetensors + `config.json` + index | 权重给出参数，config 给出计算图 |
| 文本分词 / 反分词 | tokenizer 系列文件 | 文本 ↔ token id |
| 多轮对话 | `chat_template.jinja` + tokenizer 特殊 token | 渲染 system/user/assistant/tool 消息 |
| 默认采样行为 | `generation_config.json` | eos、temperature、top_p 等 |
| 图像 / 视频输入 | 视觉权重 + `preprocessor_config.json` + `video_preprocessor_config.json` | 图像编码、投影到文本空间 |
| 微调（SFT / LoRA） | safetensors + config + tokenizer | HF 格式可直接被 PEFT/TRL 等加载 |
| 量化与端侧部署 | safetensors + config | 转换为 GGUF/AWQ/GPTQ 等 |
| 投机解码 | MTP 头 + 运行时支持 | Qwen3.5 训练时带多步 MTP；llama.cpp 当前未启用，转换时会跳过 MTP 张量 |
| 合规与复现 | `LICENSE` + `README.md` | 许可证、评测、限制、引用 |

需要强调：

- **safetensors 不是可执行程序**，它只是张量容器；
- **GGUF 也不是“另一种训练结果”**，而是把同一组权重按 llama.cpp 的布局重新打包并可选量化；
- **量化不是重新训练**，而是用更低比特表示原权重，必然有精度损失，只是 Q4_K_M 以上通常损失较小。

---

## 5. 从 HF 产物到 llama.cpp 可运行文件

### 5.1 转换链路

```text
Qwen/Qwen3.5-27B（HF 目录）
  ├── 11 × *.safetensors + index.json
  ├── config.json
  ├── tokenizer.json / vocab.json / merges.txt / tokenizer_config.json
  ├── chat_template.jinja
  └── preprocessor_config.json / video_preprocessor_config.json
                    │
                    │ convert_hf_to_gguf.py
                    ▼
  ┌───────────────────────────────────────────┐
  │ Qwen3.5-27B-BF16.gguf      文本主干权重    │
  │ mmproj-Qwen3.5-27B-BF16.gguf  视觉投影器    │
  └───────────────────────────────────────────┘
                    │
                    │ llama-quantize
                    ▼
  Qwen3.5-27B-Q4_K_M.gguf
                    │
                    │ llama-server / llama-cli / llama-mtmd-cli
                    ▼
                 推理服务
```

### 5.2 llama.cpp 中的对应实现

- 架构注册：`src/llama-arch.cpp:42` 中 `LLM_ARCH_QWEN35` 对应字符串 `"qwen35"`；MoE 版本为 `"qwen35moe"`。
- 模型构建：`src/llama-model.cpp:9014` 根据架构创建 `llm_build_qwen35`。
- 计算图：`src/models/qwen35.cpp`，其中 `build_layer_attn_linear`（约 198 行）处理 Gated DeltaNet，`build_layer_attn`（约 117 行）处理全注意力，`build_layer_ffn`（约 372 行）处理 FFN。
- 混合内存：线性注意力层使用递归状态，全注意力层使用 KV Cache。
- 分词器：`src/llama-vocab.cpp:2029` 识别 `qwen35` 预切分规则。
- HF 转换：`convert_hf_to_gguf.py:5421` 注册 `Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM`，输出 `qwen35` 架构。
- 线性注意力重排：转换器继承 `_LinearAttentionVReorderBase`，会按 `num_k_heads` / `num_v_heads` 重排 V 相关权重、`A_log`、`dt_bias`、`conv1d`、`out_proj` 等张量，使其匹配 ggml 的布局。**这也是不能手工改张量名代替转换的原因。**
- MTP：转换器基类 `Qwen3NextModel` 对 `name.startswith("mtp")` 的张量直接跳过；`src/llama-arch.cpp:757` 也注明 NextN/MTP 张量目前被忽略，保留给未来实现。因此 Qwen3.5 的 MTP 头在 llama.cpp 中暂不参与投机解码。

### 5.3 转换与运行命令

```bash
# 1. 文本主干：BF16 GGUF
python3 convert_hf_to_gguf.py ../Qwen3.5-27B \
    --outfile Qwen3.5-27B-BF16.gguf \
    --outtype bf16

# 2. 多模态投影器（视觉编码器）
python3 convert_hf_to_gguf.py ../Qwen3.5-27B \
    --mmproj \
    --outfile mmproj-Qwen3.5-27B-BF16.gguf \
    --outtype f16

# 3. 量化（示例：Q4_K_M）
./build/bin/llama-quantize \
    Qwen3.5-27B-BF16.gguf \
    Qwen3.5-27B-Q4_K_M.gguf \
    Q4_K_M

# 4. 启动多模态服务
./build/bin/llama-server \
    -m Qwen3.5-27B-Q4_K_M.gguf \
    --mmproj mmproj-Qwen3.5-27B-BF16.gguf \
    -c 32768
```

说明：

- `--outtype` 支持 `f32`、`f16`、`bf16`、`q8_0`、`tq1_0`、`tq2_0`、`auto`；
- 只做纯文本推理时可以不加载 `mmproj`；
- 上下文越长，KV Cache 和递归状态占用越大，应按硬件调整 `-c`；
- 超过 262,144 token 需要 YaRN 等长度外推配置，不能只改 `-c`。

### 5.4 常见 GGUF 体积

以下是社区量化版本的实际文件大小（不同上传者的混合精度策略不同，存在约 ±1 GB 差异）：

| 格式 | 体积 | 适用场景 |
|---|---:|---|
| BF16 | 54.66 GB | 精度基准、64 GB 以上内存/显存 |
| Q8_0 | 29.12 GB | 近无损，32 GB 显存 |
| Q6_K | 23.68 GB | 质量很高，24 GB 显存可尝试 |
| Q5_K_M | 20.97 GB | 质量与体积平衡 |
| Q4_K_M | 17.98 GB | 默认推荐，24 GB 显存较舒适 |
| Q4_K_S | 16.93 GB | 更省空间，质量略降 |
| Q3_K_XL | 16.60 GB | 低内存场景 |
| Q4_0 | 16.56 GB | 传统格式，ARM/AVX 有在线重排优化 |
| `mmproj`（BF16） | 0.93 GB | 多模态视觉投影器 |

如果使用 24 GB 显存，Q4_K_M 加 16K 上下文通常是比较稳妥的起点；如果要处理 128K 以上上下文，需要把 KV Cache 和递归状态一起计入预算。

---

## 6. 如何判断一份训练产物是否完整

拿到一个 HF 模型目录后，可以按以下顺序检查：

1. **索引与分片是否齐全**

   ```bash
   ls model.safetensors-*.safetensors | wc -l
   ```

   官方 Qwen3.5-27B 应为 11 个分片；再用 `model.safetensors.index.json` 的 `weight_map` 核对每个张量都能找到对应文件。

2. **总大小是否合理**

   BF16 下可用“参数量 × 2 字节”快速估算。27B 量级应在 50 GB 以上；如果只有 27 GB，可能是 FP8 版本；如果只有 15–18 GB，可能是 4-bit 量化版本。

3. **配置是否匹配**

   检查 `config.json` 中的 `architectures`、`model_type`、`text_config.num_hidden_layers`、`hidden_size`、`vocab_size` 是否符合预期。

4. **分词器与聊天模板是否存在**

   至少应有 `tokenizer.json`（或 `vocab.json` + `merges.txt`）和 `tokenizer_config.json`；对话模型还应检查 `chat_template.jinja` 或 `tokenizer_config.json` 中的 `chat_template` 字段。

5. **多模态文件是否存在**

   如果要做图像/视频推理，检查 `preprocessor_config.json`、`video_preprocessor_config.json`，以及转换后是否生成了 `mmproj-*.gguf`。

6. **许可证是否允许你的用途**

   Qwen3.5-27B 使用 Apache 2.0，但仍应阅读 `README.md` 中的使用限制与评测说明。

---

## 7. 常见误解

| 误解 | 事实 |
|---|---|
| “训练完就是一个 `.bin` 文件” | 现代大模型是目录，包含分片权重、索引、配置、分词器、模板等 |
| “有 safetensors 就能直接跑” | 还需要 config 重建网络、tokenizer 处理文本、chat template 组织对话 |
| “GGUF 是训练结果” | GGUF 是部署格式，由 HF 权重转换并可选量化得到 |
| “27B 模型就是 27 GB” | BF16 下约 55.6 GB；27 GB 是 FP8/INT8 量级；4-bit 约 16–18 GB |
| “量化就是压缩，不会影响效果” | 量化有精度损失，Q4_K_M 通常较小，低比特量化可能明显退化 |
| “发布权重可以原样续训” | 公开权重通常不含优化器状态、学习率调度器和数据状态，无法精确恢复训练 |
| “MTP 头会自动加速所有框架” | 需要运行时显式支持；llama.cpp 当前未启用，转换时会跳过 MTP 张量 |

---

## 8. 总结

大模型训练的最终结果，本质上是一份**可重建、可推理、可再训练、可部署的模型资产包**：

- **safetensors 分片 + index** 保存“学到了什么”；
- **config.json** 说明“网络长什么样”；
- **tokenizer + chat template** 说明“如何把文本变成模型认识的输入”；
- **generation / preprocessor 配置** 说明“默认如何生成、如何处理多模态输入”；
- **LICENSE / README** 说明“能怎么用、效果如何、有什么限制”。

以 Qwen3.5-27B 为例，官方产物是约 **55.6 GB、11 个 BF16 分片**的 HF checkpoint，包含 64 层混合注意力主干、视觉编码器、MTP 头、248,320 词表分词器和完整的对话模板。它不是“一个文件”，而是一套能被 Transformers、vLLM、SGLang、llama.cpp 等框架消费的完整描述。

在 llama.cpp 路径中，这份产物先被 `convert_hf_to_gguf.py` 拆成文本 GGUF 与 `mmproj`，再经 `llama-quantize` 压缩为 Q4_K_M 等格式，最终由 `llama-server` / `llama-cli` 加载运行。理解这些文件的职责，才能正确选择精度、上下文长度和部署方式，也才能在微调、量化与评测之间做出可复现的决策。

---

## 参考

- Qwen3.5-27B 官方模型卡：<https://huggingface.co/Qwen/Qwen3.5-27B>
- Qwen3.5-27B FP8：<https://huggingface.co/Qwen/Qwen3.5-27B-FP8>
- Transformers Qwen3.5 文档：<https://huggingface.co/docs/transformers/model_doc/qwen3_5>
- GGUF 量化示例（bartowski）：<https://huggingface.co/bartowski/Qwen_Qwen3.5-27B-GGUF>
- 本仓库相关实现：
  - `convert_hf_to_gguf.py`（Qwen3.5 转换与线性注意力重排）
  - `src/llama-arch.cpp`（`qwen35` / `qwen35moe` 架构注册）
  - `src/llama-model.cpp`（模型构建路由）
  - `src/models/qwen35.cpp`（混合注意力计算图）
  - `src/llama-vocab.cpp`（`qwen35` 分词预切分）
  - `tests/snapshots/qwen3.5-27b.schema`（27B 量化张量布局快照）

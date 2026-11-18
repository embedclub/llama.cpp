# llama.cpp 中一个 Token 的生成之旅 以 `llama-cli` 工具、Qwen3.5 模型、纯 CPU 推理为例（基于本仓库源码）

本文以 **llama-cli**（聊天界面入口）为例，跟踪 **一次前向推理产生一个新 Token** 的完整流程：从用户敲下回车，到文字被逐字打印到终端，再到该 Token 被回填供下一轮使用。所有代码位置均为本仓库当前 checkout 的真实文件与行号。

> **总体流程图**   用户输入 → 聊天模板(chat template) → **分词 tokenize** → 任务入队(server task) → 组装 batch → **llama_decode()** → 内存模块(hybrid: 卷积/状态缓存 + KV Cache) → **构建计算图**(Qwen3.5: 门控DeltaNet线性注意力 + 全注意力) → **ggml CPU 后端多线程计算** → 拷回 logits → **采样(temperature→top_k→top_p→dist)** → 得到一个 token id → **反分词 detokenize** → UTF-8 校验/停词检查 → 流式打印到终端 → token 回填 batch，进入下一轮循环

## 第 0 步 预备：加载模型，初始化纯 CPU 后端

入口在 tools/cli/cli.cpp 346 的 `main()`。这里创建 `cli_context`（内部复用了 server 的基础设施），并启动推理线程：

```cpp
// tools/cli/cli.cpp:346
int main(int argc, char ** argv) {
    common_params params;
    ...
    llama_backend_init();                       // :366 初始化 ggml 后端（纯 CPU 时只有 CPU 后端被注册）
    llama_numa_init(params.numa);               // :367 NUMA 亲和
    ...
    if (!ctx_cli.ctx_server.load_model(params)) // :387 加载 GGUF 模型
        return 1;
    std::thread inference_thread([&ctx_cli]() { // :396 推理在独立线程中循环执行
        ctx_cli.ctx_server.start_loop();
    });
    ...
}
```

### 0.1 模型加载与 Qwen3.5 架构识别

`load_model` 最终走 `common_init_from_params` → `llama_model_load_from_file`：

- tools/server/server-context.cpp 752：`llama_init = common_init_from_params(params_base);`
- common/common.cpp 1143/1234：加载模型后调用 `llama_init_from_model(model, cparams)` 创建 `llama_context`。
- src/llama.cpp 843：`llama_backend_init()` 注册 CPU 后端；931/1128：`llama_model_load_from_file_impl` → `llama_model_load`，使用 mmap 读取 GGUF（src/llama-mmap.cpp）。

加载时从 GGUF 元数据读出 `arch = "qwen3.5"`，在架构表里查到对应枚举，随后路由到 Qwen3.5 的建图器：

```cpp
// src/llama-arch.cpp:42
{ LLM_ARCH_QWEN35,           "qwen3.5"    },

// src/llama-model.cpp:9016
case LLM_ARCH_QWEN35:
    llm = std::make_unique<llm_build_qwen35>(*this, params);  // 模型计算图构建器
    break;
```

> **Qwen3.5 的关键结构（本仓库实现）**：它是 **混合架构**——部分层是「门控 DeltaNet 线性注意力」（gated delta net，带 1D 卷积 + 递归状态，不需要传统 KV Cache），部分层是标准「全注意力」（GQA + RoPE，需要 KV Cache）。因此它使用 `llama_memory_hybrid`（混合内存），由 `llama-memory-recurrent` 和 `llama-kv-cache` 两部分组成。见 src/models/qwen35.cpp:21、src/llama-hparams.h。

### 0.2 纯 CPU 的线程配置

因为没有 GPU，`ggml_backend_sched` 调度器里只有 CPU 后端，全部计算在 CPU 上完成。线程数来自启动参数 `-t/--threads`：

```cpp
// src/llama-context.cpp:45
cparams.n_threads        = params.n_threads;       // 生成阶段线程数
cparams.n_threads_batch  = params.n_threads_batch; // prompt 处理阶段线程数
```

## 第 1 步 用户输入 → 聊天模板 → 得到最终 prompt 字符串

用户敲下的文字被追加为一条 `user` 消息（tools/cli/cli.cpp 610-616），然后调用 `generate_completion`（618）。`format_chat()` 会把整个消息历史套进 Qwen3.5 的 chat template（Jinja 模板，加载自 GGUF）：

```cpp
// tools/cli/cli.cpp:205-223
common_chat_params format_chat() {
    ...
    inputs.add_generation_prompt = true;   // 模板末尾追加 "<|im_start|>assistant\n" 之类的生成提示
    ...
    // Apply chat template to the list of messages
    return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
}
```

- 模板应用实现：common/chat.cpp 2308（Jinja 路径在 2133，Jinja 引擎在 common/jinja/）。

## 第 2 步 分词：文本变成 llama_token 数组

tokenize 由 server 侧完成。llama-cli 构造 `server_task` 后投递（tools/cli/cli.cpp 88-121 `rd.post_task(...)`），任务队列处理时先做分词：

```cpp
// tools/server/server-context.cpp:1738 (tokenize_cli_input)
task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
// tools/server/server-common.cpp:793
std::vector<server_tokens> tokenize_input_prompts(...) { ... }
```

调用链：`common_tokenize`（common/common.cpp 1552）→ `llama_tokenize`（src/llama.cpp 的 C API，src/llama-vocab.cpp 4072）→ `llama_vocab::impl::tokenize`（src/llama-vocab.cpp 3066）。Qwen3.5 用的是 BPE 分词器：

```cpp
// src/llama-vocab.cpp:3066  —— 先切分特殊 token，再按词表类型分派
std::vector<llama_token> llama_vocab::impl::tokenize(...) {
    fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
    tokenizer_st_partition(fragment_buffer, parse_special);   // 处理 <|im_start|> 等特殊 token
    switch (get_type()) {
        case LLAMA_VOCAB_TYPE_BPE: { ... tokenizer.encode(text) ... }   // BPE 编码（:1435）
        ...
    }
}

// src/llama-vocab.cpp:2029 —— qwen3.5 专属的正则 pre-split 规则
} else if (tokenizer_pre == "qwen35") {
    pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN35;
    clean_spaces = false;
}
```

分词结果形如 `[151644, 872, 198, ...]`——prompt 阶段的所有 token 已经就绪。注意：**只有最后一个 token 的 `logits=1` 会要求输出**，因为此刻我们只需要「下一个词」的分布。

## 第 3 步 任务循环与 batch 组装

推理线程进入主循环：tools/server/server-queue.cpp 125 `server_queue::start_loop()` → 触发 `update_slots()`（tools/server/server-context.cpp 2118）。每个用户对话占用一个 `slot`，prompt 处理完后 slot 进入 `SLOT_STATE_GENERATING` 状态，之后每轮循环只回填一个已采样 token：

```cpp
// tools/server/server-context.cpp:332 (server_slot::update_batch)
if (spec_draft.empty()) {
    // no speculative decoding —— 纯 CPU 默认不开投机解码
    i_batch = batch.n_tokens;
    common_batch_add(batch, sampled,              // 把上一轮采样出的 token 放进 batch
                     prompt.tokens.pos_next(),    // 位置 = 已有序列长度
                     { this->id }, true);         // logits_only_for_this_token
}

// common/common.cpp:1511  —— 把一个 token 写进 llama_batch 结构
void common_batch_add(...) {
    batch.token  [batch.n_tokens] = id;
    batch.pos    [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_id.size();
    batch.seq_id [batch.n_tokens] = seq_ids.data();
    batch.logits [batch.n_tokens] = logits;
    batch.n_tokens++;
}
```

> **decode 一步只算一个新 token**：自回归生成中，历史 token 的 K/V（或 DeltaNet 的递归状态）已经缓存在内存模块里，所以每轮 decode 的 batch 通常只有 1 个 token。

## 第 4 步 llama_decode：进入推理核心

batch 组装完成后，server 调用 `llama_decode`（tools/server/server-context.cpp 2788）：

```cpp
const int ret = llama_decode(ctx, batch_view);   // tools/server/server-context.cpp:2788
```

C API 只是薄封装，真正的工作在 `llama_context::decode`：

```cpp
// src/llama-context.cpp:3451
int32_t llama_decode(llama_context * ctx, llama_batch batch) {
    return ctx->decode(batch);
}

// src/llama-context.cpp:1533
int llama_context::decode(const llama_batch & batch_inp) {
    ...
    balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all);  // :1582 batch 校验/预分配
    ...
    memory_update(false);                                 // :1617 处理挂起的 KV 变更
    mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);  // :1622 让内存模块准备本次 batch
    ...
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status); // :1692
    ...
    // :1738-1748  把 logits 从计算后端拷回主存
    ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0,
                                  n_outputs*n_vocab*sizeof(float));
}
```

### 4.1 混合内存：递归状态 + KV Cache 同时就位

Qwen3.5 走 `llama_memory_hybrid::init_batch`（src/llama-memory-hybrid.cpp 62），它同时管理两个子模块：

- **递归内存**（线性注意力层）：src/llama-memory-recurrent.cpp 379 `init_batch` / 429 `prepare`。每个序列只需保存每层的**卷积状态**（`r_l`）和 **SSM/DeltaNet 状态矩阵**（`s_l`），与序列长度无关，内存占用恒定。
- **KV Cache**（全注意力层）：src/llama-kv-cache.cpp，按 `pos` 存 K/V 并推进 `kv_head`；其上下文 `apply()` 在 2408。

## 第 5 步 构建计算图：Qwen3.5 的前向结构

`process_ubatch`（src/llama-context.cpp 1171）先尝试复用上一轮的图（生成阶段拓扑不变，可复用，1185），否则调用 `model.build_graph(gparams)`（1204；实现 src/llama-model.cpp 8539），进入 Qwen3.5 专属建图器：

```cpp
// src/models/qwen35.cpp:5  —— 计算图主干
llm_build_qwen35::llm_build_qwen35(...) : llm_build_delta_net_base(params), model(model) {
    ...
    inpL = build_inp_embd(model.tok_embd);       // :17 词嵌入查表
    auto * inp = build_inp_mem_hybrid();         // :21 混合内存输入（递归状态 + KV）
    ggml_tensor * inp_pos = build_inp_pos();     // :23 RoPE 位置
    for (int il = 0; il < n_layer; ++il) {
        cur = build_norm(inpL, model.layers[il].attn_norm, ..., LLM_NORM_RMS, il);  // :29 RMSNorm
        if (hparams.is_recurrent(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);   // :37 线性注意力层（门控DeltaNet）
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il); // :40 全注意力层
        }
        cur = ggml_add(ctx0, cur, inpSA);        // :49 残差
        attn_post_norm = build_norm(cur, ..., LLM_NORM_RMS, il);              // :56
        cur = build_layer_ffn(attn_post_norm, il);                            // :60 SwiGLU FFN
        cur = ggml_add(ctx0, cur, ffn_residual);                              // :64 残差
        inpL = cur;
    }
    cur = build_norm(cur, model.output_norm, ..., LLM_NORM_RMS, -1); // :76 最终 RMSNorm
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);       // :82 LM head：n_embd → n_vocab 投影
    res->t_logits = cur;                          // :85 这就是 logits 张量！
```

### 5.1 输入嵌入：把 token id 变成向量

```cpp
// src/llama-graph.cpp:1704
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    ...
    cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);  // :1729 按 id 从 embedding 表取行
```

### 5.2 线性注意力层（门控 DeltaNet）—— Qwen3.5 的特色

```cpp
// src/models/qwen35.cpp:198  build_layer_attn_linear
auto qkvz = build_qkvz(cur, il);                              // :219 QKV+门控 的联合投影 (wqkv, wqkv_gate)
beta  = build_lora_mm(model.layers[il].ssm_beta, cur, ...);   // :223 Δ 门控系数 → sigmoid
alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, ...);  // :230 alpha → +dt → softplus
gate  = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a); // :238 -A·softplus(dt)

conv_states = build_rs(inp, conv_states_all, ..., n_seqs);    // :248 从递归内存取上一轮卷积状态
qkv_mixed   = ggml_transpose(ctx0, qkv_mixed);
ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0); // :262 拼接历史卷积窗口

// :272-277 把 conv_input 的最后一段写回缓存（下一轮要用）
ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));

ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel); // :283 1D 卷积
conv_output_silu = ggml_silu(ctx0, conv_output_proper);                          // :286 SiLU 激活
q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);                 // :320 归一化 q/k
k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);                 // :321

auto attn_out = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il); // :339 门控DeltaNet核心
// :346-350 把新的递归状态写回内存（下一轮再用）
ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, ...));
attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);  // :356 z 门控 RMSNorm
cur = build_lora_mm(model.layers[il].ssm_out, final_output, ...);               // :363 输出投影
```

> 注意 347 与 277：状态回写（`ggml_cpy` 到内存模块张量）被显式编进了计算图，由 ggml 保证在计算时执行——这就是「递归状态更新」的实现方式，完全免除了传统 KV Cache 逐位置追加的开销。

### 5.3 全注意力层（标准 GQA + RoPE + KV Cache）

```cpp
// src/models/qwen35.cpp:117  build_layer_attn
ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, ...); // :129 Q+gate 联合投影
Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, ...); // :132 切出 Q
Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, ..., LLM_NORM_RMS, il); // :138 QK-Norm（Qwen3 系特有）
Kcur = build_lora_mm(model.layers[il].wk, cur, ...);                     // :141
Vcur = build_lora_mm(model.layers[il].wv, cur, ...);                     // :144
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, ..., LLM_NORM_RMS, il); // :149 K 也做 RMSNorm
Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, ..., sections, ...);          // :162 M-RoPE
Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, ..., sections, ...);          // :168
cur = build_attn(inp, ..., Qcur, Kcur, Vcur, ..., kq_scale, il);          // :181 注意力 + KV 写入
cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));                      // :186-189 sigmoid 门控
cur = build_lora_mm(model.layers[il].wo, cur, ...);                       // :192 输出投影
```

`build_attn`（src/llama-graph.cpp 2174 → `build_attn_mha` 1932）完成：把 K/V 写入 KV Cache（由 `llm_graph_input_attn_kv::set_input` 在 src/llama-graph.cpp 447 同步掩码/索引）、计算 QKᵀ/softmax/加权和。生成阶段 batch 只有 1 个 token，所以注意力掩码退化得很简单，主要工作就是**与历史 K/V 做点积**。

### 5.4 FFN 层（SwiGLU）

```cpp
// src/models/qwen35.cpp:372
cur = build_ffn(cur,
    model.layers[il].ffn_up,   ..., LLM_FFN_SILU, LLM_FFN_PAR, il); // :376-381 gate/up 双投影 + SiLU 门控
// build_ffn 实现：src/llama-graph.cpp:1141
```

## 第 6 步 CPU 多线程执行计算图

建好图之后，在 `process_ubatch` 中（src/llama-context.cpp 1226-1231）先设置输入张量（token id、位置、注意力掩码、递归状态切片等——`res->set_inputs(&ubatch)` 分派到 src/llama-graph.cpp 中各 input 的 `set_input()`，如 75、302、447），然后：

```cpp
// src/llama-context.cpp:1231
const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
// src/llama-context.cpp:2191
auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
// ggml/src/ggml-backend.cpp:1889
enum ggml_status ggml_backend_sched_graph_compute_async(...) { ... ggml_backend_sched_compute_splits ... }
```

纯 CPU 时只有一个后端，整张图交给 CPU 后端一次性计算：

```cpp
// ggml/src/ggml-cpu/ggml-cpu.cpp:170
static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) { ... }

// ggml/src/ggml-cpu/ggml-cpu.c:3237 —— 核心执行器
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ...
    // 按 n_threads 启动工作线程（ggml_threadpool），主线程也参与计算
    // :2962 static thread_ret_t ggml_graph_compute_thread(void * data)
    // 每个节点按 ggml_get_n_tasks()（:2181）决定拆给几个线程，
    // 线程间用 barrier 同步（:563 ggml_barrier）
}
```

### 6.1 热点算子：矩阵乘 + 量化向点积

语言模型 95% 的算力花在 `ggml_mul_mat`（QKV/LM head/FFN 投影）上。CPU 后端会按行块把矩阵乘拆成多线程的**量化向量点积**：

```cpp
// ggml/src/ggml-cpu/ggml-cpu.c:1241
void ggml_compute_forward_mul_mat(struct ggml_compute_params * params, struct ggml_tensor * dst) { ... }
// :1151  ggml_compute_forward_mul_mat_one_chunk —— 分块调用 vec_dot
// :1425      ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, ...)

// ggml/src/ggml-cpu/quants.c:639 —— 例如 Q4_K 量化权重的点积内核
void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, ...) { ... }
```

> **量化**：GGUF 模型权重通常以 Q4_K/Q5_K/Q8_0 等格式存储，CPU 上每次点积先把权重块解包再与激活值做 SIMD 内积（Apple 上走 Accelerate/Metal SIMD，x86 走 AVX2/AVX512，见 ggml/src/ggml-cpu/arch/）。这就是 llama.cpp 在纯 CPU 上跑得快的根本原因。

计算结束后，回到 src/llama-context.cpp 1738-1748，把 shape 为 `[n_vocab]` 的 logits 行拷回 `logits.data` 主存缓冲。

## 第 7 步 采样：从 n_vocab 个 logits 里挑出「那一个」token

decode 返回后，server 对每个 slot 调用采样器（tools/server/server-context.cpp 2916-2920）：

```cpp
llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx, tok_idx);  // :2916
common_sampler_accept(slot.smpl.get(), id, true);                            // :2920 接受并记录（供重复惩罚）
```

采样链在任务启动时按参数建好（tools/server/server-context.cpp 1293 `common_sampler_init`，实现 common/sampling.cpp 185，典型链为 `penalties → temperature → top_k → top_p → min_p → dist`）。核心逻辑：

```cpp
// src/llama-sampler.cpp:806
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    ...
    const auto * logits = llama_get_logits_ith(ctx, idx);   // :849 拿到刚才拷回的 logits
    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }
    ...
    llama_sampler_apply(smpl, &cur_p);       // :864 依次过采样链的每个 sampler
    auto token = cur_p.data[cur_p.selected].id;  // :868 最终选中的 token id
    llama_sampler_accept(smpl, token);        // :870
    return token;
}
```

- softmax：src/llama-sampler.cpp 289 `llama_sampler_softmax_impl`；
- 最终随机抽样（temperature/top_p 调整后的分布）：src/llama-sampler.cpp 1036 `llama_sampler_dist_apply`。

至此，**一个完整的 token id 诞生了**——它是 `n_vocab`（Qwen3.5 词表约 15 万+）个候选中被选中的那一个。

## 第 8 步 反分词：token id → 文字片段

拿到 id 后立刻转成可打印文字（tools/server/server-context.cpp 2937）：

```cpp
result.text_to_send = common_token_to_piece(slot.ctx, result.tok, accept_special_token(slot, result.tok));
// common/common.cpp:1572
std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) { ... }
// → src/llama-vocab.cpp:4083 llama_token_to_piece → :3383 impl::detokenize
// BPE 词表直接查 id→piece 缓存，再做 UTF-8 字节解码（llama_decode_text，:3046）
```

### 8.1 停词 / EOS / UTF-8 校验，并流式下发

```cpp
// tools/server/server-context.cpp:1337  process_token
bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();  // :1349
    // ↑ 一个中文字符可能被拆成多个 BPE token，凑齐完整 UTF-8 才输出，避免乱码
size_t stop_pos = slot.find_stopping_strings(str_test, ...);                        // :1358 停词检测
...
} else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok)) {          // :1364 EOS 检查
...
slot.add_token(result);
if (slot.task->params.stream) {
    send_partial_response(slot, result, false);                                      // :1381 流式推送
}
...
```

增量结果以 `server_task_result_cmpl_partial` 消息形式回到 llama-cli 主线程，逐字打印到终端：

```cpp
// tools/cli/cli.cpp:151-163
auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
if (res_partial) {
    for (const auto & diff : res_partial->oaicompat_msg_diffs) {
        if (!diff.content_delta.empty()) {
            curr_content += diff.content_delta;
            console::log("%s", diff.content_delta.c_str());   // ← 屏幕上「蹦」出的那一个词
            console::flush();
        }
        ...
    }
}
```

## 第 9 步 闭环：这个 token 成为下一轮的输入

`process_token` 返回 `true` 表示继续生成（tools/server/server-context.cpp 2944）。该 token id 已存进 `slot.sampled`（1340），下一轮 `update_slots()` 时又被 `slot.update_batch`（384）放入新的 batch——回到**第 4 步**，周而复始。

终止条件（src 侧判断）：context 用尽（1390）、`n_predict` 预算用完（1400）、命中 stop 字符串或 EOS（1358-1366）。终止后发送 final 响应，llama-cli 把整段回复作为一条 `assistant` 消息存回历史（tools/cli/cli.cpp 619-622），等待用户的下一句话。

## 附录 A：一次 token 生成的关键代码位置速查表

| 阶段 | 做什么 | 代码位置 |
| --- | --- | --- |
| 入口 | llama-cli 主函数 / 推理线程 | tools/cli/cli.cpp:346, 396 |
| 加载 | backend 初始化 / 模型加载 | src/llama.cpp:843, 931；tools/server/server-context.cpp:752 |
| 架构 | 识别 qwen3.5 / 选择建图器 | src/llama-arch.cpp:42；src/llama-model.cpp:9016 |
| 模板 | chat template 应用 | tools/cli/cli.cpp:205；common/chat.cpp:2308 |
| 分词 | BPE 编码 | src/llama-vocab.cpp:3066, 1435, 2029；common/common.cpp:1552 |
| 调度 | 任务循环 / 组 batch | tools/server/server-queue.cpp:125；server-context.cpp:2118, 332, 384；common/common.cpp:1511 |
| 解码 | llama_decode 主流程 | src/llama-context.cpp:3451, 1533, 1622, 1692 |
| 内存 | 混合内存 / 递归状态 / KV | src/llama-memory-hybrid.cpp:62；src/llama-memory-recurrent.cpp:379；src/llama-kv-cache.cpp:2408 |
| 建图 | Qwen3.5 计算图 | src/models/qwen35.cpp:5（整体）、198（DeltaNet 层）、117（注意力层）、372（FFN）、76-85（LM head） |
| 建图辅助 | 嵌入查表 / norm / attn / ffn | src/llama-graph.cpp:1704, 1028, 2174/1932, 1141；src/llama-model.cpp:8539 |
| 计算 | 图调度 / CPU 执行 / 线程 | src/llama-context.cpp:1226-1231, 2191；ggml/src/ggml-backend.cpp:1889；ggml/src/ggml-cpu/ggml-cpu.cpp:170；ggml-cpu.c:3237, 2962, 1241；quants.c:639 |
| 采样 | 采样链 / softmax / 抽样 | tools/server/server-context.cpp:2916, 1293；common/sampling.cpp:185, 526；src/llama-sampler.cpp:806, 289, 1036 |
| 输出 | 反分词 / 流式打印 | common/common.cpp:1572；src/llama-vocab.cpp:4083, 3383；server-context.cpp:1337, 1381；tools/cli/cli.cpp:151-163 |
| 闭环 | token 回填下一轮 | server-context.cpp:1340 → 384 → 2916（循环） |

## 附录 B：为什么纯 CPU 也能跑 Qwen3.5？

1. **混合架构省显存**：线性注意力层只保存固定大小的递归状态（src/llama-memory-recurrent.cpp），长上下文下内存不随长度线性增长，全注意力层仅对部分层维护 KV Cache。
2. **量化点积**：权重以 4/5/8-bit 量化存储，CPU 内核（ggml/src/ggml-cpu/quants.c、arch/）用 SIMD 逐块解包计算，带宽需求大幅降低。
3. **多线程 + 图调度**：ggml 把整张计算图按节点拆任务，`ggml_graph_compute`（ggml-cpu.c:3237）用线程池 + barrier 并行执行，算子内按 `n_tasks` 继续分块。
4. **图复用**：生成阶段每轮只喂 1 个 token，计算图拓扑完全相同，`process_ubatch`（src/llama-context.cpp:1185）直接复用上一轮的图，省去重复建图/分配的时间。

本文档基于本仓库源码生成，行号对应当前 HEAD（git 1081bf1a9）。llama.cpp 的 C API 层（src/llama.cpp）为薄封装，实际逻辑在 llama-context / llama-model / llama-graph / llama-vocab / llama-sampler / llama-memory-* / ggml 中。

# llama-cli 的 prefill 阶段源码分析

> 分析对象：`tools/cli/cli.cpp`（本仓库 commit `0a2d12fcc`）
> 相关文件：`tools/server/server-context.cpp`、`tools/server/server-queue.cpp`、`tools/server/server-common.cpp`、`common/common.cpp`、`src/llama-context.cpp`

## 0. 先给结论

`cli.cpp` 本身**不直接调用 `llama_decode()`**。它复用 server 的推理引擎（`server_context` / `server_context_impl`）：

- `cli.cpp` 只负责：读取用户输入 → 套 chat template 生成 prompt 字符串 → 构造 `SERVER_TASK_TYPE_COMPLETION` 任务 → 投递到队列 → 消费流式结果。
- 真正的 **prefill（prompt 处理）** 发生在 `tools/server/server-context.cpp` 的 `update_slots()` 中：把 prompt token 组进 `llama_batch`，然后调用 `llama_decode()`。
- `cli.cpp` 与 server 的唯一区别是任务上打了 `task.cli = true`，于是服务端在 `process_single_task()` 里先做一次 CLI 专用 tokenize。

因此“cli 的 prefill”要沿着下面这条链去看：

```
tools/cli/cli.cpp
  main()
    └─ cli_context::generate_completion()              cli.cpp:75
         ├─ format_chat()                              cli.cpp:197
         │    └─ common_chat_templates_apply()         cli.cpp:215   ← 生成 prompt 字符串
         ├─ 构造 server_task(COMPLETION), cli=true     cli.cpp:80
         └─ rd.post_task()                             cli.cpp:113   ← 投递任务
              │
tools/server/server-queue.cpp
  server_queue::start_loop()                           server-queue.cpp:125
    ├─ callback_new_task → process_single_task()       server-context.cpp:1833
    │    ├─ tokenize_cli_input()                       server-context.cpp:1738
    │    │    └─ tokenize_input_prompts()              server-common.cpp:793
    │    └─ launch_slot_with_task()                    server-context.cpp:1217
    │         └─ slot.state = SLOT_STATE_STARTED
    └─ callback_update_slots → update_slots()          server-context.cpp:2118
         ├─ STARTED → PROCESSING_PROMPT                server-context.cpp:2266
         ├─ prompt cache 复用 / KV shift / checkpoint  server-context.cpp:2342-2494
         ├─ 逐 token 填入 llama_batch                  server-context.cpp:2627
         ├─ DONE_PROMPT, 只取最后一个 logits           server-context.cpp:2678
         └─ llama_decode(ctx, batch_view)              server-context.cpp:2788
              └─ llama_context::decode()               src/llama-context.cpp:1533
```

---

## 阶段 1：输入 → chat template → prompt 字符串

入口在交互主循环 `main()`：

```cpp
// tools/cli/cli.cpp:602-610
if (add_user_msg) {
    ctx_cli.messages.push_back({ {"role","user"}, {"content", cur_msg} });
    cur_msg.clear();
}
result_timings timings;
std::string assistant_content = ctx_cli.generate_completion(timings);
```

`generate_completion()` 第一件事是把对话消息套上模板：

```cpp
// tools/cli/cli.cpp:197-215（节选）
common_chat_params format_chat() {
    auto meta = ctx_server.get_meta();
    auto & chat_params = meta.chat_params;

    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = chat_params.use_jinja;

    return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
}
```

**这一步的产物**：`chat_params.prompt` —— 一整段纯文本（已包含 BOS、系统提示、角色标记、`add_generation_prompt` 等）。它还不是 token，只是字符串。

---

## 阶段 2：构造任务并投递（cli 与 server 分叉点）

```cpp
// tools/cli/cli.cpp:76-113（节选）
server_response_reader rd = ctx_server.get_response_reader();
auto chat_params = format_chat();
{
    server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
    task.id         = rd.get_new_id();
    task.index      = 0;
    task.params     = defaults;           // 采样参数等
    task.cli_prompt = chat_params.prompt; // 字符串 prompt
    task.cli_files  = input_files;        // 图片/音频
    task.cli        = true;               // ← 关键：标记为 CLI 输入
    ...
    rd.post_task({std::move(task)});
}
```

要点：

1. `task.cli_prompt` 是**原始字符串**，服务端还没 tokenize。
2. `task.cli = true` 决定后面走 `tokenize_cli_input()` 分支。
3. `post_task()` 把任务放到 `server_queue`；`rd`（`server_response_reader`）负责之后按 id 收结果。
4. 真正的模型推理在另一个线程里跑：

```cpp
// tools/cli/cli.cpp:388-390
std::thread inference_thread([&ctx_cli]() {
    ctx_cli.ctx_server.start_loop();
});
```

`server_context::start_loop()` 只是转发到 `server_queue::start_loop()`（`server-context.cpp:3078`）。

---

## 阶段 3：队列调度（每轮 = 收任务 + 跑一次 update_slots）

```cpp
// tools/server/server-queue.cpp:139-163（节选）
while (true) {
    while (true) {
        ...
        server_task task = std::move(queue_tasks.front());
        queue_tasks.pop_front();
        callback_new_task(std::move(task));   // → process_single_task()
    }
    callback_update_slots();                  // → update_slots()，真正推理
    ...
}
```

回调在 `server-context.cpp:978-982` 注册：

```cpp
queue_tasks.on_new_task([this](server_task && task) {
    process_single_task(std::move(task));
});
queue_tasks.on_update_slots([this]() {
    update_slots();
});
```

所以模型是**按“调度轮次”推进的**：新任务先登记到某个 slot，下一轮 `update_slots()` 才开始 prefill。

---

## 阶段 4：CLI 专用 tokenize

`process_single_task()` 对 CLI 任务先做 tokenize：

```cpp
// tools/server/server-context.cpp:1833-1846（节选）
void process_single_task(server_task && task) {
    switch (task.type) {
        case SERVER_TASK_TYPE_COMPLETION:
            ...
            if (task.cli) {
                if (!tokenize_cli_input(task)) {
                    break;
                }
            }
```

```cpp
// tools/server/server-context.cpp:1738-1753
bool tokenize_cli_input(server_task & task) {
    try {
        auto & prompt = task.cli_prompt;
        if (mctx != nullptr) {
            task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files); // 多模态
        } else {
            task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
        }
        task.cli_prompt.clear();
        task.cli_files.clear();
    } catch (...) { ... }
    return true;
}
```

- `add_special = true`、`parse_special = true`：会加 BOS/EOS 类特殊 token，并解析 prompt 里的特殊 token 文本。
- 结果存进 `task.tokens`（`server_tokens`，内部是 `llama_tokens` + 多模态索引表，见 `server-common.h:128-227`）。
- 之后 `slot.task->n_tokens()` 就等于这段 token 数。

---

## 阶段 5：分配 slot 并“挂载”任务

```cpp
// tools/server/server-context.cpp:1851-1887（节选）
server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);
if (slot == nullptr) { queue_tasks.defer(...); break; }
...
} else if (!launch_slot_with_task(*slot, std::move(task))) { ... }
```

`launch_slot_with_task()`（`server-context.cpp:1217`）做 prefill 前的准备：

1. LoRA / aLoRA 处理（`1219-1281`），aLoRA 会记录 `alora_invocation_start`，影响后面能缓存到哪。
2. token 合法性检查：`task.tokens.validate(ctx)`（`1283`）。
3. 初始化采样器 `common_sampler_init(...)`；必要时设置 backend sampling（`1291-1319`）。
4. 把 `task` 移入 slot，并设置初始状态（`1324-1328`）：

```cpp
slot.task  = std::make_unique<const server_task>(std::move(task));
slot.state = slot.task->is_child()
    ? SLOT_STATE_WAIT_OTHER   // 子任务等父任务先 prefill
    : SLOT_STATE_STARTED;     // 父任务从这里开始 prefill
```

注意：**此时还没算任何东西**，只是“任务已绑定到 slot”。真正的计算在下一轮 `update_slots()`。

---

## 阶段 6：`update_slots()` 的 prefill 前半段（准备）

`update_slots()` 是 server 的核心循环（`server-context.cpp:2118`）。每一轮它把当前所有 slot 的工作组装成一个（或多个）`llama_batch`，然后 decode 一次。

### 6.1 状态切换与计时起点

```cpp
// tools/server/server-context.cpp:2259-2273（节选）
if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
    const auto & input_tokens = slot.task->tokens;   // 目标 prompt
    ...
    if (slot.state == SLOT_STATE_STARTED) {
        slot.t_start_process_prompt = ggml_time_us();   // prefill 计时开始
        slot.t_start_generation     = 0;
        slot.state = SLOT_STATE_PROCESSING_PROMPT;
        SLT_INF(slot, "new prompt, n_ctx_slot = %d, ... task.n_tokens = %d\n", ...);
    }
```

### 6.2 prompt cache 复用（同一 session 的第二轮对话会命中）

若 `cache_prompt` 打开，先求新旧 prompt 的公共前缀：

```cpp
// tools/server/server-context.cpp:2342-2360（节选）
if (slot.task->params.cache_prompt) {
    // 复用与新 prompt 相同的历史 token
    n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

    if (slot.alora_invocation_start > 0) {
        n_past = std::min(n_past, slot.alora_invocation_start - 1);
    }
    ...
}
```

`n_past` = 已经存在于 KV cache 中、可以跳过的 token 数。

### 6.3 `n_cache_reuse`：KV cache 片段平移

当问题是“中间相同、结尾不同”时，可以把旧 KV 片段 shift 到新位置复用（`2363-2409`）：

```cpp
const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;
llama_memory_seq_rm (llama_get_memory(ctx), slot.id, head_p, head_c);
llama_memory_seq_add(llama_get_memory(ctx), slot.id, head_c, head_c + n_match, kv_shift);
```

### 6.4 checkpoint 恢复

若前缀不再完整匹配但 SWA（滑窗注意力）场景下旧 checkpoint 仍可用，会恢复上下文快照：

```cpp
// tools/server/server-context.cpp:2491-2494
const size_t n = llama_state_seq_set_data_ext(ctx, it->data.data(), checkpoint_size,
                                              slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
```

### 6.5 超长 prompt 的截断 / 清理

在填充 batch 之前还有若干保护（`2550-2570` 附近）：`task->n_tokens()` 超过 `n_batch` 需要分批；超过上下文则报错或按策略截断，并可能 `slot.prompt_clear(true)` 重置。

---

## 阶段 7：把 prompt token 填进 `llama_batch`

这是 prefill 的核心“装箱”步骤：

```cpp
// tools/server/server-context.cpp:2626-2672（节选，有删减）
// add prompt tokens for processing in the current batch
while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
    llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
    if (cur_tok == LLAMA_TOKEN_NULL) {
        break; // 遇到多模态 chunk，稍后单独处理
    }
    ...
    // 嵌入任务要求所有 token 都输出
    common_batch_add(batch,
        cur_tok,
        slot.prompt.tokens.pos_next(),
        { slot.id },
        slot.task->need_embd());
    slot.prompt.tokens.push_back(cur_tok);

    slot.n_prompt_tokens_processed++;
    ...
}
```

关键点：

- **每次最多 `n_batch` 个 token**（逻辑 batch 上限）。超长 prompt 会自动分成多轮 `update_slots()`，每轮 decode 一批。
- `slot.prompt.tokens` 记录“已经送进模型的 token”；`slot.prompt.n_tokens()` 就是进度指针。
- `pos` 用 `slot.prompt.tokens.pos_next()`，对多模态（M-RoPE）也能正确处理位置不连续。
- `common_batch_add()`（`common/common.cpp:1511`）只写指针数组并自增 `batch.n_tokens`，非常轻量。

### 多模态分支

遇到 `LLAMA_TOKEN_NULL` 表示此处是图片/音频 chunk：

```cpp
// tools/server/server-context.cpp:2604-2624（节选）
while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
    int32_t res = input_tokens.process_chunk(ctx, mctx, slot.prompt.n_tokens(),
                                             slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
    ...
    slot.n_prompt_tokens_processed += n_tokens_out;
    slot.prompt.tokens.push_back(chunk.get());
}
```

`process_chunk()` 内部会跑视觉/音频编码器，把 embedding 写进上下文，因此这部分也属于广义 prefill。

---

## 阶段 8：整段 prompt 装完 → DONE_PROMPT

```cpp
// tools/server/server-context.cpp:2674-2690（节选）
const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

// entire prompt has been processed
if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
    slot.state = SLOT_STATE_DONE_PROMPT;

    GGML_ASSERT(batch.n_tokens > 0);

    // extract the logits only for the last token
    batch.logits[batch.n_tokens - 1] = true;

    slot.n_decoded = 0;
    slot.i_batch   = batch.n_tokens - 1;

    slot.init_sampler();
    SLT_INF(slot, "prompt processing done, n_tokens = %d, batch.n_tokens = %d\n", ...);
}
```

这是 prefill 与 generation 的分界点：

- prefill 期间只有**最后一个 prompt token** 需要 logits（`batch.logits[last] = true`），其余 token 只更新 KV。
- `slot.i_batch` 记录“这个 slot 的 logits 在 batch 中的下标”，decode 之后靠它取结果。
- 如果 prompt 还没装完（`else` 分支，`2691-2714`），状态保持 `PROCESSING_PROMPT`，下一轮继续。

另外，装批过程中可能顺便创建 KV checkpoint（`2731-2733`），给后续 prompt 复用 / 投机解码回滚用：

```cpp
if (do_checkpoint) {
    create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
}
```

---

## 阶段 9：真正的前向计算 —— `llama_decode()`

```cpp
// tools/server/server-context.cpp:2746-2788（节选）
SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

if (slot_batched) {
    common_set_adapter_lora(ctx, slot_batched->lora);
    ...
    llama_set_embeddings(ctx, slot_batched->task->need_embd());
}

int32_t i_next = 0;
for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
    const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

    llama_batch batch_view = {
        n_tokens,
        batch.token    + i,
        nullptr,
        batch.pos      + i,
        batch.n_seq_id + i,
        batch.seq_id   + i,
        batch.logits   + i,
    };

    const int ret = llama_decode(ctx, batch_view);
    metrics.on_decoded(slots);
    ...
}
```

要点：

1. 传入的是 `batch_view`（batch 的一个切片），不是整个 batch。物理层还会被 `llama_context::decode()` 按 `n_ubatch` 再切。
2. `n_batch`（逻辑批）与 `n_ubatch`（物理批）不同：**prompt 装箱用 `n_batch` 控制，真正图执行用 `n_ubatch` 切**。
3. 返回值：
   - `0` 成功；
   - `1` KV cache 找不到空位 / 上下文超限 → server 会把 `n_batch` 减半重试，或清理空闲 slot（`2831-2838`）；
   - `-1` 非法输入批；`<-1` 计算错误（`2792-2829`）。

### 下探到 llama_context

```cpp
// src/llama-context.cpp:3451
int32_t llama_decode(llama_context * ctx, llama_batch batch) {
    const int ret = ctx->decode(batch);
    ...
    return ret;
}
```

`llama_context::decode()`（`src/llama-context.cpp:1533`）内部主要做：

1. 参数/批次校验，处理 embeddings 输出、backend sampling；
2. 申请 KV cache 槽位（memory module），必要时做 cache defrag/optimize；
3. 按 `n_ubatch` 逐个 `process_ubatch()`（`src/llama-context.cpp:1692`）：
   - 构建计算图（`llm_graph_type = LLM_GRAPH_TYPE_DECODER`）；
   - 调度 ggml 后端执行；
   - 把 K/V 写回 `llama_memory`；
   - 按 `batch.logits` 把需要的 logits 写入输出缓冲。
4. 失败时回滚对应位置的 KV，并返回错误码。

对 prefill 来说，这里就是“一次性把这段 prompt 的 KV 算出来”。

---

## 阶段 10：decode 之后 —— 首 token 采样与计时

`llama_decode()` 返回后，`update_slots()` 遍历 slot：

```cpp
// tools/server/server-context.cpp:2870-2931（节选）
if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
    continue;
}

if (slot.state == SLOT_STATE_DONE_PROMPT) {
    ...
    slot.state = SLOT_STATE_GENERATING;

    if (slot.can_speculate()) {
        common_speculative_begin(slot.spec.get(), slot.prompt.tokens.get_text_tokens());
    }
}
...
const int tok_idx = slot.i_batch - i;
llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx, tok_idx);
...
slot.n_decoded += 1;

if (slot.n_decoded == 1) {
    slot.t_start_generation   = t_current;
    slot.t_prompt_processing  = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
    metrics.on_prompt_eval(slot);   // 统计 prompt eval 吞吐
}
```

因此：

- prefill 的**产物**就是最后一个 prompt token 的 logits；
- 用它在 `GENERATING` 状态下采出第一个生成 token；
- `t_prompt_processing` 在这一刻定格，后面都是 decode/generation 的计时。

---

## 阶段 11：结果流式回传 CLI

生成 token 时：

```cpp
// tools/server/server-context.cpp:1337-1382（节选）
slot.add_token(result);
if (slot.task->params.stream) {
    send_partial_response(slot, result, false);   // → queue_results.send()
}
```

CLI 侧同步消费：

```cpp
// tools/cli/cli.cpp:122-173（节选）
console::spinner::start();
server_task_result_ptr result = rd.next(should_stop);

while (result) {
    ...
    auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
    if (res_partial) {
        out_timings = std::move(res_partial->timings);
        for (const auto & diff : res_partial->oaicompat_msg_diffs) {
            ... console::log("%s", diff.content_delta.c_str()); ...
        }
    }
    auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
    if (res_final) {
        out_timings = std::move(res_final->timings);
        break;
    }
    result = rd.next(should_stop);
}
```

最后 `main()` 打印用户看到的 prompt/generation 速度：

```cpp
// tools/cli/cli.cpp:617-621
console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n",
             timings.prompt_per_second, timings.predicted_per_second);
```

其中 `prompt_per_second` 来自 slot 的 `t_prompt_processing` / `n_prompt_tokens_processed`（`server-context.cpp:435-440`）。

---

## 关键数据结构对照

| 结构 | 位置 | 作用 |
| --- | --- | --- |
| `server_task` | `tools/server/server-task.h` | 一次请求（CLI 时带 `cli_prompt`、`cli=true`） |
| `server_task::tokens` | `server_tokens` | tokenize 后的 prompt（含多模态索引） |
| `server_slot` | `server-context.cpp:82` 起 | 一个并行序列的运行时状态，持有 `prompt`、`task`、`smpl` |
| `server_slot::prompt` | `server_prompt` | “已送进模型的 token + KV checkpoint 元数据” |
| `slot.prompt.tokens` | `server_tokens` | 已处理 token 的追加记录，兼作进度指针 |
| `llama_batch` | `include/llama.h` | 本轮到 `llama_decode` 的 token/pos/seq_id/logits 数组 |
| `n_batch` / `n_ubatch` | `llama_n_batch()` / `llama_n_ubatch()` | 逻辑批 / 物理批 |
| `llama_memory` | `llama_get_memory(ctx)` | KV cache（含 SWA / recurrent 的实现细节） |

### prefill 的进度判定

```cpp
slot.prompt.n_tokens()   // 已处理的 token 数
slot.task->n_tokens()    // 目标 prompt 总 token 数
```

两者相等 → prefill 结束；否则下一轮 `update_slots()` 继续。**不要**把 `batch.n_tokens` 当成 prompt 总长度，它只是当前这一批的大小。

---

## 时序总览（单轮对话）

```
用户输入
  │
  ├─ cli.cpp: format_chat() → prompt 字符串
  ├─ cli.cpp: post_task(COMPLETION, cli=true)
  │
  └─ 推理线程 server_queue::start_loop()
       │
       ├─ process_single_task()
       │    ├─ tokenize_cli_input() → task.tokens
       │    └─ launch_slot_with_task() → state=STARTED
       │
       └─ update_slots()
            ├─ STARTED → PROCESSING_PROMPT
            ├─ cache_prompt: get_common_prefix() → n_past
            ├─ 填 batch（≤ n_batch 个 token/轮）
            ├─ 装完 → DONE_PROMPT，logits[last]=true
            ├─ llama_decode() → llama_context::decode() → process_ubatch()
            └─ 采样首 token → GENERATING → 流式返回
  │
  └─ cli.cpp: rd.next() 打印 content_delta / timings
```

---

## 调优相关的参数（和 prefill 直接相关）

| 参数 | 影响 |
| --- | --- |
| `-b/--batch-size`（`n_batch`） | 每轮最多处理多少 prompt token；越大 prefill 轮数越少，但显存峰值越高 |
| `-ub/--ubatch-size`（`n_ubatch`） | 真正图执行的物理批；影响单次 kernel/显存占用 |
| `--cache-prompt` / `cache_prompt` | 是否复用历史 prompt 的 KV（多轮对话提速的关键） |
| `--cache-reuse N`（`n_cache_reuse`） | 是否把旧 KV 片段平移复用（中间相同、结尾不同时有效） |
| `--no-context-shift` | 影响上下文满时的截断策略（prefill 前会检查） |
| `--swa-full` / `n_ctx_checkpoints` | 与 SWA checkpoint 恢复相关 |
| `-np/--parallel`、`--cont-batching` | 多 slot 时能否把多个 prompt 合并进同一批 prefill |

---

## 一句话总结

`cli.cpp` 的 prefill 不是本地循环，而是“**把 chat 模板渲染成字符串 → 打成 COMPLETION 任务 → 由 server_context 的 `update_slots()` 做 prompt 装箱 → `llama_decode()` 一次性算完 prompt 的 KV → 只取最后一个 logits 采首 token**”。理解这条链的关键分界点是 `SLOT_STATE_DONE_PROMPT` 和 `batch.logits[last] = true`。

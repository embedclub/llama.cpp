# `ctx_cli.generate_completion` 推理实现分析

> 基于本仓库当前 checkout 的源码；文中文件与行号均对应当前版本。

## 1. 结论先行

`cli_context::generate_completion()` 定义在 `tools/cli/cli.cpp:75`。它**本身并不执行神经网络推理**，而是一个"进程内客户端"：

1. 把当前对话历史套用聊天模板，得到最终 prompt；
2. 把 prompt 封装成一个 `server_task`（类型 `SERVER_TASK_TYPE_COMPLETION`）投递到 `server_context` 的任务队列；
3. 以**流式**方式从结果队列消费 `server_task_result`，把增量文本打印到终端，最后返回完整回复。

真正的推理发生在 `main()` 里另起的一个线程中：该线程调用 `ctx_server.start_loop()`，由 server 的调度循环完成分词、slot 调度、`llama_decode()`、采样和结果回传。

因此，`generate_completion` 的实现本质上是 **"生产者 + 消费者"**：生产一个任务，消费一串结果。它完整复用了 `llama-server` 的 slot / batch / KV cache / 采样 / chat parser 基础设施，只是省掉了 HTTP 层。

全链路概览：

```
CLI 主线程                           推理线程 (start_loop)
-----------                          ---------------------
format_chat()                        从 queue_tasks 取任务
  ↓                                   process_single_task()
构造 server_task                       tokenize_cli_input()
  ↓                                     ↓
rd.post_task() ─────────────────────→ launch_slot_with_task()（初始化采样器）
                                        ↓
                                     update_slots()（组装 batch）
                                        ↓
                                     llama_decode()（CPU/GPU 前向）
                                        ↓
                                     common_sampler_sample()（采样）
                                        ↓
                                     common_token_to_piece()（反分词）
                                        ↓
                                     process_token()（停词/UTF-8/停止条件）
                                        ↓
rd.next() ←────────────────────────  queue_results.send()（partial/final）
  ↓
打印 content_delta / reasoning_content_delta
  ↓ （循环直到 final）
返回完整文本 + timings
```

## 2. 组件与线程模型

| 组件 | 作用 |
| --- | --- |
| `cli_context` | CLI 侧状态：`server_context ctx_server`、`json messages`、`input_files`、默认 `task_params`。见 `tools/cli/cli.cpp:47-217` |
| CLI 主线程 | 读取用户输入、维护消息历史、调用 `generate_completion`、打印流式结果。`main()` 位于 `tools/cli/cli.cpp:338` |
| 推理线程 | `main()` 中 `std::thread` 启动，执行 `ctx_server.start_loop()`（`cli.cpp:388-390`） |
| `server_context` / `server_context_impl` | 持有 model、`llama_context`、vocab、slots、任务/结果队列，并注册回调（`tools/server/server-context.cpp:978-986`） |
| `server_queue` | 任务队列 `queue_tasks`，`post()` 加锁入队并 `notify_one`（`tools/server/server-queue.cpp:22-39`） |
| `server_response_reader` | 一次请求的句柄：注册等待的 task id、`next()` 轮询结果、把原始 chunk 解析成 chat diff、析构时取消未完成任务（`tools/server/server-queue.h:164-205`、`server-queue.cpp:343-450`） |
| `server_response` | 结果队列 `queue_results`，`send()` 入队并唤醒，`recv_with_timeout()` 阻塞等待（`server-queue.cpp:289-337`） |
| `server_slot` | 一条并行序列的运行时状态：prompt tokens、已生成 token、采样器、状态机。`update_batch()` 在 `server-context.cpp:332` |

关键点：CLI 与推理不在同一线程。二者通过两个带互斥锁/条件变量的队列通信，`generate_completion` 只是消费者的一端。

## 3. 逐步代码分析

### 3.1 获取 response reader

```cpp
// tools/cli/cli.cpp:75-77
std::string generate_completion(result_timings & out_timings) {
    server_response_reader rd = ctx_server.get_response_reader();
    auto chat_params = format_chat();
```

`get_response_reader()` 返回一个绑定到该 `server_context` 的读者：

```cpp
// tools/server/server-context.cpp:3062-3064
server_response_reader get_response_reader() {
    return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
}
```

`HTTP_POLLING_SECONDS = 1`（`server-context.cpp:36`），即 `next()` 每秒轮询一次，用来检查 `should_stop`（Ctrl+C）。

### 3.2 应用聊天模板

`format_chat()`（`cli.cpp:197-216`）把 `messages` 转成模板输入，并调用 `common_chat_templates_apply()`：

```cpp
// tools/cli/cli.cpp:201-215
common_chat_templates_inputs inputs;
inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
inputs.use_jinja             = chat_params.use_jinja;
inputs.add_generation_prompt = true;
inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
...
return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
```

返回的 `common_chat_params` 同时包含：最终 `prompt` 字符串、模板自带的 `parser`、`generation_prompt`、`thinking_start_tag` / `thinking_end_tag`，后续用于流式 chat diff 解析和 reasoning budget。

### 3.3 构造并投递任务

```cpp
// tools/cli/cli.cpp:80-113（节选）
server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
task.id         = rd.get_new_id();
task.index      = 0;
task.params     = defaults;           // copy
task.cli_prompt = chat_params.prompt; // copy
task.cli_files  = input_files;        // copy
task.cli        = true;

task.params.chat_parser_params = common_chat_parser_params(chat_params);
task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
if (!chat_params.parser.empty()) {
    task.params.chat_parser_params.parser.load(chat_params.parser);
}

// reasoning budget sampler
if (!chat_params.thinking_end_tag.empty()) {
    const llama_vocab * vocab = llama_model_get_vocab(
        llama_get_model(ctx_server.get_llama_context()));
    task.params.sampling.reasoning_budget_tokens = reasoning_budget;
    task.params.sampling.generation_prompt       = chat_params.generation_prompt;
    ...
    task.params.sampling.reasoning_budget_end =
        common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
    ...
}

rd.post_task({std::move(task)});
```

注意三点：

- `task.cli = true` 且只填 `cli_prompt`、不填 `tokens`，因此分词被**推迟到推理线程**执行（因为 vocab / mtmd context 只存在于 `server_context` 内部）。
- `defaults` 在构造 `cli_context` 时设置：`stream = true`、`timings_per_token = true`，保证逐 token 流式返回并携带耗时（`cli.cpp:59-73`）。
- `post_task()` 会注册等待 id、为流式解析创建状态，然后入队：

```cpp
// tools/server/server-queue.cpp:343-350
void server_response_reader::post_task(server_task && task, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
    GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
    task.index = 0;
    id_tasks.insert(task.id);
    states.push_back(task.create_state());
    queue_results.add_waiting_task_id(task.id);
    queue_tasks.post(std::move(task), front);
}
```

`create_state()` 返回 `task_result_state`，其中保存了 chat parser 参数和增量解析状态，供后续 `next()` 生成 `oaicompat_msg_diffs`。

### 3.4 推理线程事件循环

`main()` 中的推理线程：

```cpp
// tools/cli/cli.cpp:388-390
std::thread inference_thread([&ctx_cli]() {
    ctx_cli.ctx_server.start_loop();
});
```

`server_context::start_loop()` 委托给 `server_queue::start_loop()`（`server-context.cpp:3078-3081`），其主循环（`server-queue.cpp:125-209`）为：

1. 反复从 `queue_tasks` 取任务，逐个调用 `callback_new_task`；
2. 队列排空后调用 `callback_update_slots`（即 `update_slots()`），驱动本轮推理；
3. 再进入等待：有新任务、需要休眠、或终止时退出。

回调在 `init()` 中注册：

```cpp
// tools/server/server-context.cpp:978-983
queue_tasks.on_new_task([this](server_task && task) {
    process_single_task(std::move(task));
});
queue_tasks.on_update_slots([this]() {
    update_slots();
});
```

`process_single_task()` 对完成类任务的处理（`server-context.cpp:1833-1896`）：

- CLI 任务先分词：`tokenize_cli_input(task)`（`1738`）→ `tokenize_input_prompts(vocab, mctx, prompt, true, true)`（`tools/server/server-common.cpp:793`）→ `common_tokenize` → `llama_tokenize`。
- 选 slot：`get_slot_by_id()` 或 `get_available_slot(task)`；没有空闲 slot 则 `defer()` 稍后重试。
- `launch_slot_with_task()`（`1217`）：初始化采样器 `common_sampler_init(model, task.params.sampling)`（`1293`）；若启用后端采样则 `llama_set_sampler(ctx, slot.id, ...)`（`1314`）；把 task 移入 slot，状态置为 `SLOT_STATE_STARTED`。

### 3.5 `update_slots()`：组装 batch

`update_slots()`（`server-context.cpp:2118`）是推理循环核心：

1. 若所有 slot 空闲，直接返回。
2. 投递一个 `SERVER_TASK_TYPE_NEXT_RESPONSE` 空任务（`2140-2142`；在 `process_single_task` 的 `1907-1910` 中不做任何事），用于让任务队列保持非空、循环在连续生成期间持续运转，而不是进入空闲等待。
3. 需要时做 context shift（`2147-2204`）。
4. `common_batch_clear(batch)` 开始新 batch（`2207`）。
5. 对处于 `SLOT_STATE_GENERATING` 的 slot，把上一轮采样出的 token 加入 batch：

```cpp
// tools/server/server-context.cpp:380-384（server_slot::update_batch）
if (spec_draft.empty()) {
    i_batch = batch.n_tokens;
    common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);
}
```

`common_batch_add()`（`common/common.cpp:1511`）把 token、pos、seq_id 和 `logits=true` 写入 `llama_batch`，因此只有新 token 需要输出 logits。

6. 对 `STARTED` / `PROCESSING_PROMPT` 的 slot，按 `n_batch` / `n_ubatch` 把 prompt token 分块加入 batch（`2240` 起），并处理是否复用缓存、是否创建 checkpoint 等。
7. 执行解码（见下）。

### 3.6 调用 `llama_decode()`

```cpp
// tools/server/server-context.cpp:2775-2788
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
    ...
}
```

- 返回 0 表示成功；`ret == 1` 表示 KV 无空位（代码会把 `n_batch` 减半重试或清理空闲 slot）；`ret == -1` 无效 batch；`ret < -1` 计算错误（`2792-2839`）。
- 成功后 `i_next = i + n_tokens`，继续处理剩余 batch。
- 解码完成后，logits 已由 `llama_context::decode()` 拷回主存（`src/llama-context.cpp`）。

### 3.7 采样与反分词

解码成功后，对 `i_batch` 落在本次 chunk 范围内的 slot：

```cpp
// tools/server/server-context.cpp:2900-2944（节选）
slot.state = SLOT_STATE_GENERATING;

const int tok_idx = slot.i_batch - i;
llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx, tok_idx);
slot.i_batch = -1;
common_sampler_accept(slot.smpl.get(), id, true);

slot.n_decoded += 1;
...
completion_token_output result;
result.tok          = id;
result.text_to_send = common_token_to_piece(slot.ctx, result.tok,
                                            accept_special_token(slot, result.tok));

if (!process_token(result, slot)) {
    slot.print_timings();
    send_final_response(slot);
    metrics.on_prediction(slot);
    slot.release();
    continue;
}
```

- `common_sampler_sample()`（`common/sampling.cpp:526`）从第 `tok_idx` 个输出取 logits，依次应用采样链并返回选中的 token。
- `common_sampler_accept()`（`common/sampling.cpp:432`）把 token 记入采样器历史，供重复惩罚/语法等使用。
- `common_token_to_piece()`（`common/common.cpp:1572`）把 token id 转成文本片段。

### 3.8 `process_token()`：流式输出与停止条件

`process_token()`（`server-context.cpp:1337-1466`）决定"是否继续生成"：

- 记录 `slot.sampled`、追加文本、`n_decoded` 计数；
- UTF-8 完整性检查：一个中文/emoji 可能被拆成多个 token，未凑齐完整 UTF-8 前不发送（`1349`、`1385`）；
- 停止字符串检测 `find_stopping_strings()`（`1358`）；
- 上下文用尽（`1390`）、`n_predict` 预算（`1400`）、缩进/时间限制（`1407-1454`）；
- EOS/EOG 检测 `llama_vocab_is_eog()`（`1456`）；
- 若仍继续且 `stream == true`，调用 `send_partial_response()`（`1381`）。

`send_partial_response()`（`1554-1592`）构造 `server_task_result_cmpl_partial`，写入 `content`、`tokens`、`timings`、`n_decoded` 等，然后 `queue_results.send()`。

### 3.9 结果回传与 reader 消费

`server_response::send()`（`server-queue.cpp:319-332`）把结果放入 `queue_results`，并 `condition_results.notify_all()`。

CLI 侧的 `rd.next()`（`server-queue.cpp:378-407`）循环：

```cpp
server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
if (result == nullptr) {
    if (should_stop()) return nullptr;   // Ctrl+C
} else {
    if (result->is_error()) { stop(); return result; }
    if (!states.empty()) {
        const size_t idx = result->index;
        result->update(states[idx]);     // 生成 chat diff
    }
    if (result->is_stop()) received_count++;
    return result;
}
```

`result->update()` 对 partial/final 都会调用 `task_result_state::update_chat_msg()`（`server-task.cpp:158`），用 `common_chat_parse()` 解析当前累计文本，再 `compute_diffs()` 得到本 chunk 的 `oaicompat_msg_diffs`。这就是 CLI 能拿到 `content_delta` / `reasoning_content_delta` 的原因。

### 3.10 CLI 侧的打印循环

```cpp
// tools/cli/cli.cpp:130-174（节选）
while (result) {
    if (should_stop()) break;
    if (result->is_error()) { ... return curr_content; }

    auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
    if (res_partial) {
        out_timings = std::move(res_partial->timings);
        for (const auto & diff : res_partial->oaicompat_msg_diffs) {
            if (!diff.content_delta.empty()) {
                ...
                curr_content += diff.content_delta;
                console::log("%s", diff.content_delta.c_str());
                console::flush();
            }
            if (!diff.reasoning_content_delta.empty()) {
                console::log("%s", diff.reasoning_content_delta.c_str());
                console::flush();
            }
        }
    }

    auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
    if (res_final) {
        out_timings = std::move(res_final->timings);
        break;
    }
    result = rd.next(should_stop);
}
g_is_interrupted.store(false);
return curr_content;
```

- partial：累加并逐字打印，同时拿到每 token timings；
- final：取最终 timings 并结束；最终响应在流模式下 `content` 为空（内容已在 partial 中发完），只带统计信息（`send_final_response`，`server-context.cpp:1594-1615`）。

### 3.11 收尾与取消

- `server_response_reader` 析构时调用 `stop()`（`server-queue.cpp:431-450`）：移除等待 id；若任务未完成，投递 `SERVER_TASK_TYPE_CANCEL`（高优先级 `front=true`）取消剩余任务。
- `slot.release()` 把 slot 重新标记为空闲，并更新统计数据。
- `main()` 结束时调用 `ctx_server.terminate()` 并 `join` 推理线程（`cli.cpp:632-633`）。

## 4. 一次调用的时序

| 步骤 | 线程 | 位置 | 动作 |
| --- | --- | --- | --- |
| 1 | CLI | `cli.cpp:77` | `format_chat()` 套模板得 prompt |
| 2 | CLI | `cli.cpp:80-113` | 构造 `server_task` 并 `post_task` |
| 3 | 推理 | `server-queue.cpp:157` | `callback_new_task` → `process_single_task` |
| 4 | 推理 | `server-context.cpp:1738` | CLI prompt 分词 |
| 5 | 推理 | `server-context.cpp:1217` | `launch_slot_with_task` 初始化 sampler/slot |
| 6 | 推理 | `server-context.cpp:2118` | `update_slots` 组 batch |
| 7 | 推理 | `server-context.cpp:2788` | `llama_decode` |
| 8 | 推理 | `server-context.cpp:2916` | `common_sampler_sample` |
| 9 | 推理 | `server-context.cpp:2937` | `common_token_to_piece` |
| 10 | 推理 | `server-context.cpp:1337` | `process_token` 判停/发 partial |
| 11 | CLI | `server-queue.cpp:378` | `rd.next()` 收结果并解析 diff |
| 12 | CLI | `cli.cpp:143-167` | 打印增量 |
| 13 | 推理 | `server-context.cpp:1594` | 满足停止条件后发 final |
| 14 | CLI | `cli.cpp:168-171` | 收到 final，返回完整文本 |

## 5. 关键设计点

1. **进程内复用 server 基础设施**
   `generate_completion` 没有自己的推理代码，而是复用 `server_context_impl` 的 slot 调度、batching、KV/内存管理、采样和 chat parser。CLI 相当于一个不带 HTTP 的客户端。

2. **流式优先**
   `defaults.stream = true`，每个 token 通过 `send_partial_response` 立即入队，主线程逐字打印；`timings_per_token = true` 让每个 partial 都带耗时统计。

3. **原始 token → chat diff**
   server 侧发出的是原始文本 chunk；reader 在 `update()` 中用 `common_chat_parse(..., is_partial=true)` 增量解析，算出 `content_delta` / `reasoning_content_delta` / 工具调用增量。这样 CLI 不必自己理解模板结构。

4. **后端采样可选**
   在 `launch_slot_with_task` 中，若满足条件（未开投机解码、未请求 probs），调用 `llama_set_sampler()` 让采样进入计算图/后端执行；否则在主机侧用 `common_sampler_sample()`。CLI 无需感知差异。

5. **取消与中断**
   `should_stop()` 通过 `g_is_interrupted` 原子标志传递 Ctrl+C；`next()` 每秒轮询一次；reader 析构会取消未完成任务，避免推理线程空转。

6. **多 slot / 并发**
   虽然 CLI 通常只有一个对话，但底层 `update_slots()` 支持多 slot 并行、按 `n_batch` 分块、把多个序列拼进同一个 `llama_batch`，与 `llama-server` 完全一致。

## 6. 关键代码位置速查

| 主题 | 位置 |
| --- | --- |
| `generate_completion` 定义 | `tools/cli/cli.cpp:75` |
| 获取 reader / 套模板 | `tools/cli/cli.cpp:76-77`、`197-216` |
| 构造并投递任务 | `tools/cli/cli.cpp:80-113` |
| CLI 打印循环 | `tools/cli/cli.cpp:130-174` |
| `main()` 与推理线程 | `tools/cli/cli.cpp:338`、`388-390` |
| 结果消费循环 | `tools/server/server-queue.cpp:378-407` |
| `post_task` | `tools/server/server-queue.cpp:343-351` |
| `start_loop` | `tools/server/server-queue.cpp:125-209` |
| 队列/结果发送 | `tools/server/server-queue.cpp:22-39`、`319-332` |
| 回调注册 | `tools/server/server-context.cpp:978-986` |
| `process_single_task` | `tools/server/server-context.cpp:1833` |
| CLI 分词 | `tools/server/server-context.cpp:1738-1753` |
| `launch_slot_with_task` | `tools/server/server-context.cpp:1217` |
| 采样器初始化 | `tools/server/server-context.cpp:1290-1322` |
| `update_slots` | `tools/server/server-context.cpp:2118` |
| `server_slot::update_batch` | `tools/server/server-context.cpp:332-409` |
| `llama_decode` 调用 | `tools/server/server-context.cpp:2788` |
| 采样 + 反分词 | `tools/server/server-context.cpp:2914-2944` |
| `process_token` | `tools/server/server-context.cpp:1337-1466` |
| 发 partial / final | `tools/server/server-context.cpp:1554-1592`、`1594-1615` |
| chat diff 解析 | `tools/server/server-task.cpp:158`、`1382` |
| reader 析构取消 | `tools/server/server-queue.cpp:431-450` |

## 7. 与 HTTP server 路径的关系

`llama-server` 的 HTTP handler 也使用同一套 `server_response_reader` / `server_queue` / `update_slots` / `llama_decode` 流程，区别仅在于：

- HTTP 路径在 HTTP 线程里先把 prompt 分词，再投递任务；CLI 则设置 `task.cli = true`，由推理线程分词。
- HTTP 路径把结果序列化成 SSE/JSON 发回网络；CLI 直接把 `oaicompat_msg_diffs` 打印到终端。
- 两者共享 slot 调度、batching、采样、停止条件与取消机制。

因此阅读 `generate_completion` 是理解 llama.cpp server 推理主循环的一个很好的切入点：它的调用链几乎就是整个 server 的推理骨架。

## 8. 总结

`ctx_cli.generate_completion()` 是一个轻量的进程内客户端：

- **生产**：`format_chat()` → `server_task` → `post_task()`；
- **推理**：独立线程的 `start_loop()` → `process_single_task()` → `update_slots()` → `llama_decode()` → `common_sampler_sample()` → `common_token_to_piece()` → `process_token()`；
- **消费**：`rd.next()` → chat diff → 终端流式打印 → final 结束。

它把"交互式命令行"与"高性能推理服务"解耦，通过任务/结果队列连接，既复用了 server 的全部能力，又保持了 CLI 代码的简洁。

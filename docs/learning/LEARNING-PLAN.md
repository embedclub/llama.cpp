# llama.cpp 源码学习计划（24 周 × 4h/周）

> 定案于 2026-11-10，经 grilling 确认。目标：能修改 llama.cpp 源码实现新功能，或优化纯 CPU 推理的性能与功耗。

## 0. 已确认的约束与决定

| 项 | 结论 |
|---|---|
| 方向 | 性能/功耗优先，新功能在后（P3 → P4） |
| 周期 | 24 周，每周 4h（建议拆 2+2 两个时段，每时段固定件事） |
| 切入 | 自顶向下（跟一个 token 的路径），后期对热 kernel 自底向上 |
| 测试床 | 自己下载小模型（推荐 Qwen3-1.7B 或 Llama-3.2-1B 的 Q4_K_M，≈1GB）+ 现有 Qwen3.8-27B-UD-Q8_K_XL 做真实负载 |
| 编译 | clang（Apple clang 21, arm64），实验版 build 已配置：`GGML_METAL=OFF, GGML_ACCELERATE=ON, GGML_NATIVE=ON, GGML_OPENMP=OFF` |
| 功耗 | **无 sudo**：电池模式 `pmset -g batt` 读系统瓦数；交流电用活动监视器"能源影响"做代理 |
| 仓库 | 双仓库（见 ADR-0001）：`agent_workspace`=阅读版，`local_agentspace`=实验版，本计划产物全部在实验版 |
| 学习面 | 主线 `llama-cli` + `llama-bench`；`llama-server` 只做黑盒浅层（P4 第 23 周） |
| 自评 | C++ 3 / NEON·内存层次 3 / Transformer 推理原理 2 / profiling 工具 2 → P0 补后两项 |

**非目标**：Metal/多 backend 路径、分布式/RPC、GGML_BACKEND_DL、Vulkan；server 端 API 细节。

## 1. 无 sudo 测量协议（贯穿全程，先固化）

1. **时间维度** `llama-bench`：统一跑 `pp512` 与 `tg128`，固定 `--threads-p`/`--threads-t`，每配置 warmup 一次。
2. **功率维度（精确）**：插拔到电池，跑同样的 llama-bench，记录 `pmset -g batt` 显示的系统瓦数 W，令
   - 能效 ≈ W / tok/s = J/token。
3. **功率维度（粗）**：交流电日常，活动监视器「能源」标签看 `llama-cli` 进程"能源影响"，只做相对比较。
4. 一切对比用 `--seed` 固定 + `llama-cli -p` 固定 prompt，暖机后取样，同 machine 状态（勿在重度任务并行时取）。

> 测量脚本骨架放在 `docs/learning/scripts/energy-sweep.sh`（P2 周制作），输出 CSV：threads, pp512, tg128, W, J/tok。

## 2. 阶段与周计划

### P0 预备（w1–2）——补齐自评中的两个 2 分项

- **w1 原理速成**：
  - 弄懂并能在 3 句话内解释：prefill vs decode 为何性能特征完全不同（算强 vs 内存带宽瓶颈）、KV cache 为什么必须存在、量化 Q4_K 相对 fp16 省的是什么（体积和带宽，算的反而可能更费）。
  - 资料：官方 docs、llama.cpp 仓库内 `docs/`；可看 Karpathy 的 LLM101n/llm.c 视频作对照。
  - 检验题：27B Q8 模型 decode 慢的主因是算力不够还是带宽不够？（答：带宽——权重流式读取）
- **w2 工具链**：
  - 学会读 `llama-cli -v` 的全部日志（tensor 加载、mmap、图构建、计时各条的含义）；
  - 跑通 `llama-bench`，理解其输出列；用 Instruments 的 Time Profiler 对 `llama-cli` tg128 采样一张火焰图，记下前 5 个热点函数名（不需看懂实现）。
  - 下载小模型并固化基准命令。

### P1 自顶向下：一个 token 的旅程（w3–7）

- **w3 启程**：`tools/main/main.cpp` → `common::init_backend` → tokenize → `llama_decode`
  - 产出：调用链地图 v0（`docs/learning/call-chain-map.md`，只要求主干 10 个函数+一句话职责）。
- **w4 一次 decode 的内部**：`src/llama-context.cpp` 的 decode → 逻辑 batch → ubatch 切分 → `llama-graph.cpp` 的 build（embed→每层 transformer→norm）→ 提交给 backend 队列 → 待执行。
  - 检验题：为什么 ubatch 较大时 prefill 快、tg 时无所谓？
- **w5 输出侧**：backend 执行结果 → logits → `common/sampling.*` 的 sampler 链 → detokenize → 回到循环。
  - 动手：写一个 `-v` 打开时看清每层耗时构架的实验；改 1 处日志重新编译以验证你会改代码并复编译。
- **w6 模型加载与 GGUF**：`ggml/src/gguf.cpp`、`src/llama-model-loader.*`、mmap 语义、张量元数据（metadata）的名字规律；用 `llama-gguf` 工具 dump 你小模型的 extras 验证认知。
- **w7 里程碑 M1**：把地图补完（decode 尾→采样→下一轮），写一页"每层的张量形状账本"（embed、Wq/Wk/Wv、KV、vis）。自测：不看代码说出 token 从 prompt 到输出经过的 10 个关键函数及数据形状变化。

### P2 ggml CPU backend 深挖（w8–12）

- **w8 ggml 分层**：`ggml.c`（图与算子定义）→ `ggml-backend*.cpp`（调度抽象）→ `ggml/src/ggml-cpu/`（CPU 算子）；树目录以 `ggml/src/ggml-cpu/` 为主。
  - 检验题：一次 decode 里每层各出现多少个 op？排序前 3？（用 `-v`/debug 计数验证）
- **w9 量化家族**：读 `ggml-quants.c/h`：Q4_0 / Q8_0 / Q4_K 块结构差异；vec_dot 系列函数；量化要点在整数化 scale 几乎不占用循环主体。
  - 动手：llama-bench 转换小模型为 Q4_0 / Q8_0 / Q6_K 各一份，比速度，与理论带宽模型对账。
- **w10 Accelerate 与 NEON**：`ggml-cpu/arch/arm/` 下 NEON 实现；`ggml-cpu.c`/`ops.cpp` 中何时走 Accelerate（BLAS）分支；用 `GGML_ACCELERATE=OFF` 重新 build 一版做速度对比，理解"哪些 op 受 BLAS 替换影响"。
- **w11 线程与调度**：`ggml-cpu/cpus.cpp`（thread pool）、`n_threads` 的影响路径；Apple 双簇（P/E cores）由 `sysctl -n hw.perflevel*.*logicalcpu` 查看；写 `energy-sweep.sh`（线程 1…P 数、batch 扫）。
- **w12 里程碑 M2**：产出基线报告 `docs/learning/baseline.md`：
  - 小模型/27B 两档的 pp512、tg128、瓦数、J/token；
  - Instruments 热点榜 top 10（分去重到函数级）；
  - 结论：下一步可挖掘的两个油矿（候选写明：某 vec_dot、某 elementwise op、线程策略）。

### P3 优化实战（w13–18）

- **w13–14 项目 A：能效实验集（低风险）**
  - 跑并分析：线程数 sweep（含只用 P 核 affinity 实验）、`n_batch`/`ubatch` sweep、批种子基准；
  - 产出：数据表 +「这台机器上 decode 能效最优线程数」结论；如有反直觉现象（如超线程后功耗猛增）归因。
  - 检验：解释为什么 tg 阶段线程超过某数后 J/token 变差（带宽封顶 + 功率上升）。
- **w15–17 项目 B：单 kernel NEON 优化**
  - 从 M2 热点榜选一个（默认候选：`vec_dot_q4_K_q8_K` 或 `mul_mat` 内积路径）；
  - 方法：先读现实现→写一个独立 micro-bench harness→逐版重写（用 `vdotq_*`、dotprod 扩展）→每版跑 `test-backend-ops` 与 llama-bench、并 llama-cli 同 seed 比对输出一致；
  - 提交到实验版仓库，留 before/after 数据。
  - 红线：结果数据必须由 llama-bench 产出，禁止凭感觉说快了。
- **w18 复盘 + 决定 (c)**：视 P2/P3 热点与兴趣决定是否加做 KV cache 布局/预取实验；里程碑 M3 = 一份"已验证的优化"列表（含数据与代码 diff 链接）。

### P4 新功能项目与收尾（w19–24）

- **w19–20 选题与设计**（已定 c：届时再定）：
  - 候选 A：sampling 层新功能（如新 sampler/惩罚规则）——要读 `common/sampling.*`；
  - 候选 B：graph/op 层新功能（新增 op 或改 build_graph）——要读 `llama-graph.cpp` 与 op 注册全链；
  - 写半页设计文档（接口、改动文件清单、验证方式）。
- **w21–22 实现**：小步提交；写/build 对应测试（参照 `tests/` 既有风格）；llama-cli 端到端验证。
- **w23 server 黑盒浅层 + 收个口**：跑 `llama-server`，理解其与 CLI 共用哪些模块、哪里不同（HTTP/队列/多 context）；决定是否尝试上游 PR——先读 `CONTRIBUTING.md`（注意：本项目不接受纯 AI 生成的 PR，并要求你能亲自解释与维护你的每一行代码）。
- **w24 总复盘**：更新 GLOSSARY.md、call-chain-map、baseline；写"还行不知道清单"作为第二期计划；留实验仓库整洁（分支名带 `learn/` 前缀）。

## 3. 里程碑一览

| 里程碑 | 周 | 产物 | 验收 |
|---|---|---|---|
| M1 | w7 | `call-chain-map.md` + 每层张量形状账本 | 不看代码复述 token 全旅程 |
| M2 | w12 | `baseline.md` + `energy-sweep.sh` + 热点榜 | 能解释每列数据来源 |
| M3 | w18 | 已验证优化列表（含 before/after 数据） | 每条有 llama-bench 数据支撑 |
| M4 | w24 | 新功能 + 测试 + 复盘文档 | 能向他人讲清实现与取舍 |

## 4. 核心文件地图（阅读优先级序）

```
tools/main/main.cpp              入口、主循环
common/common.cpp|sampling.*     tokenizer、sampler 链
src/llama-context.cpp            decode/prefill 编排
src/llama-graph.cpp              计算图构建
src/llama-kv-cache.cpp           KV cache 生命周期
src/llama-model*.cpp             架构与张量加载
src/llama-arch.cpp               架构名→tensor map
src/llama-sampling.cpp           llama 层 sampler 支持
src/llama-server → tools/server/ 浅层
ggml/src/gguf.cpp                GGUF 容器
ggml/src/ggml.c                  图定义/算子注册（老派 C，重读难）
ggml/src/ggml-backend*.cpp       调度抽象
ggml/src/ggml-cpu/               ★ CPU backend 主体（ops.cpp, cpus.cpp）
ggml/src/ggml-cpu/arch/arm/      ★ NEON 实现
ggml/src/ggml-quants.c|h         ★ 量化块与 vec_dot
```

## 5. 每周节奏与失败模式

- 每周 4h = 2h 阅读做笔记 / 2h 动手验证；**动手时间不少于 40%**，卡在"读懂"不是学习，改一行代码看现象才是。
- 常见失败模式：通读式学习（会忘）、改代码没固定 seed（误判效果）、只测小模型（27B 的带宽瓶颈现象才出得来）、笔记只贴代码不写自己的话。
- 计划追赶机制：单周超支不超过 30%，则压缩阅读深度优先保证里程碑产物成立。

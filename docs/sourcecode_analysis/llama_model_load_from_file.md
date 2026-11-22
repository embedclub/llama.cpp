# `llama_model_load_from_file` 逐行梳理

> 源码基线：本仓库当前检出（`src/llama.cpp`、`src/llama-model.cpp`、`src/llama-model-loader.cpp`、`src/llama-vocab.cpp`）。
> 文中行号均对应当前工作区文件，方便直接跳转（格式 `文件:行号`）。

---

## 0. 一句话结论

`llama_model_load_from_file()` 本身只是一个**很薄的公共 API 包装**，真正的工作在 `llama_model_load_from_file_impl()` 与 `llama_model_load()` 中完成：

```
llama_model_load_from_file()                       // src/llama.cpp:1163  公共入口
  └─ llama_model_load_from_file_impl()             // src/llama.cpp:931   参数校验 / 设备选择 / 生命周期
       └─ llama_model_load()                       // src/llama.cpp:874   加载流程编排
            ├─ llama_model_loader 构造             // src/llama-model-loader.cpp:510  GGUF 解析/分片/权重索引
            ├─ model.load_arch()                   // src/llama-model.cpp:683        识别模型架构
            ├─ model.load_hparams()                // src/llama-model.cpp:693        读取超参数
            ├─ model.load_vocab()                  // src/llama-model.cpp:2952       构建词表/分词器
            ├─ model.load_stats() + print_info()   // src/llama-model.cpp:678/8096   统计与打印
            └─ model.load_tensors()                // src/llama-model.cpp:2958       建张量/分配后端 buffer/搬运权重
```

整个加载过程可以概括为 **“解析元数据 → 识别架构 → 读超参 → 建词表 → 建张量并绑定到设备 → 把权重数据搬到显存/内存”**。

---

## 1. 公共 API 入口

`include/llama.h` 中的声明：

```c
// include/llama.h:474-479
// Load a model from a file
// If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
LLAMA_API struct llama_model * llama_model_load_from_file(
                         const char * path_model,
          struct llama_model_params   params);
```

实现（`src/llama.cpp:1163`）：

```cpp
struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};                                   // 1166
    return llama_model_load_from_file_impl(                                 // 1167
        nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}
```

要点：

1. `splits` 传空 `vector`，表示“由 loader 自己按 `<name>-%05d-of-%05d.gguf` 规则推导分片文件名”。
2. `metadata`、`set_tensor_data`、`file` 都传 `nullptr`，即“从磁盘文件加载”的标准路径。

同一实现还被下面几个入口复用：

| 入口 | 位置 | 与标准路径的差异 |
|---|---|---|
| `llama_model_load_from_splits` | `src/llama.cpp:1170` | 调用方显式给出分片路径列表，`splits.front()` 作为主文件 |
| `llama_model_load_from_file_ptr` | `src/llama.cpp:1186` | 从已打开的 `FILE *` 加载，不传路径 |
| `llama_model_init_from_user` | `src/llama.cpp:1144` | 从外部 `gguf_context` + 回调构造，`use_mmap/use_extra_bufts` 强制关闭 |
| `llama_load_model_from_file`（已废弃） | `src/llama.cpp:1157` | 直接转发到新 API |

`llama_model_params` 的默认值（`src/llama-model.cpp:9064`）：

```cpp
llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,       // 自动选设备
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,            // 负数 = 全部层放 GPU
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_direct_io               =*/ false,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
    };
    return result;
}
```

字段定义见 `include/llama.h:286-322`。理解这张表是理解后面分支的前提：`n_gpu_layers`、`split_mode`、`use_mmap`、`vocab_only`、`no_alloc`、`progress_callback` 会分别影响不同阶段。

---

## 2. `llama_model_load_from_file_impl()`：参数校验与设备选择

`src/llama.cpp:931-1142`。函数签名：

```cpp
static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params);
```

### 2.1 三种输入源必须“有且只有一个”

```cpp
// 939-954
int n_sources_defined = 0;
if (metadata != nullptr) n_sources_defined++;
if (!path_model.empty()) n_sources_defined++;
if (file != nullptr)     n_sources_defined++;
if (n_sources_defined != 1) {
    LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
    return nullptr;
}
```

即模型来源只能三选一：**外部 GGUF 元数据 / 文件路径 / 已打开的文件指针**。

### 2.2 初始化计时与后端检查

```cpp
// 955-960
ggml_time_init();

if (!params.vocab_only && ggml_backend_reg_count() == 0) {
    LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() ...\n", __func__);
    return nullptr;
}
```

- `ggml_time_init()`：初始化高精度计时（Windows 下尤其需要）。
- 只要不是 `vocab_only`，就要求至少注册了一个 ggml backend，否则无法建 buffer。

### 2.3 默认进度回调

```cpp
// 962-977
unsigned cur_percentage = 0;
if (params.progress_callback == NULL) {
    params.progress_callback_user_data = &cur_percentage;
    params.progress_callback = [](float progress, void * ctx) {
        unsigned * cur_percentage_p = (unsigned *) ctx;
        unsigned percentage = (unsigned) (100 * progress);
        while (percentage > *cur_percentage_p) {
            *cur_percentage_p = percentage;
            LLAMA_LOG_CONT(".");                 // 每涨 1% 打一个点
            if (percentage >= 100) LLAMA_LOG_CONT("\n");
        }
        return true;                             // 永不取消
    };
}
```

如果调用方没给回调，就装一个“打点式”默认回调；回调返回 `false` 会让加载提前取消（对应后面的 `-2` 状态）。

### 2.4 创建模型对象

```cpp
// 979
llama_model * model = new llama_model(params);
```

构造函数（`src/llama-model.cpp:668`）只做一件事：记录 `params`，并判断是否存在 tensor buffer type 覆盖：

```cpp
llama_model::llama_model(const llama_model_params & params)
    : params(params), pimpl(std::make_unique<impl>()) {
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}
```

### 2.5 构建要使用的设备列表（核心分支）

这一段（`982-1101`）决定“权重最终放到哪些设备上”。

#### 分支 A：调用方显式给了 `params.devices`

```cpp
// 982-1006
if (params.devices) {
    if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        // 统计设备数量（NULL 结尾）
        size_t n_devs = 0;
        while (params.devices[n_devs]) n_devs++;
        if (n_devs == 0) { ... return nullptr; }

        // 多设备做张量并行时，用一个 "Meta device" 包住所有设备
        model->get_split_state_ud.n_devices = n_devs;
        model->get_split_state_ud.model     = model;
        model->devices.push_back({
            true, ggml_backend_meta_device(params.devices, n_devs,
                                           llama_meta_device_get_split_state,
                                           &model->get_split_state_ud)
        });
    } else {
        // 普通模式：逐个加入
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev)
            model->devices.push_back({false, *dev});
    }
}
```

- `llama_device` 的 `is_meta` 标志表示这是一个逻辑聚合设备。
- `LLAMA_SPLIT_MODE_TENSOR`（张量并行）会走 Meta device，并在后面用 `llama_meta_device_get_split_state` 回调决定每个张量切到哪。

#### 分支 B：自动选设备（`params.devices == nullptr`）

先分成三类候选：`gpus`、`igpus`、`rpc_servers`（`1011-1013`）。

- **张量并行模式**（`1015-1042`）：遍历所有 backend 设备，跳过 CPU（`buffer_type == ggml_backend_cpu_buffer_type()`），剩余设备组成 Meta device。
- **普通模式**（`1043-1089`）：遍历设备并按类型分流：
  - `CPU` / `ACCEL`：跳过（CPU 单独处理）。
  - `GPU`：如果 backend 名字是 `RPC`，进 `rpc_servers`；否则用 `device_id` 去重，避免同一物理 GPU 被多个 backend 重复注册（如 CUDA + Vulkan 同时可见）。
  - `IGPU`：进 `igpus`。
  - `META`：`GGML_ABORT`（不应出现在这里）。

组装顺序（`1091-1100`）：

```cpp
model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end()); // RPC 放最前，减少网络传输
model->devices.insert(model->devices.end(),   gpus.begin(), gpus.end());               // 再放独显
if (model->devices.empty())                                                            // 一个都没有才用核显
    model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
```

> 设计意图：RPC 放最前、独显优先、核显兜底。

### 2.6 单卡模式裁剪设备

```cpp
// 1103-1117
if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
    if (params.main_gpu < 0) {
        model->devices.clear();              // main_gpu < 0：完全不用 GPU
    } else {
        if (params.main_gpu >= (int)model->devices.size()) {
            LLAMA_LOG_ERROR(...);
            llama_model_free(model);          // 这里正确释放了 model
            return nullptr;
        }
        llama_device main_gpu = model->devices[params.main_gpu];
        model->devices.clear();
        model->devices.push_back(main_gpu);   // 只保留主 GPU
    }
}
```

### 2.7 打印设备信息并调用真正的加载函数

```cpp
// 1119-1126
for (const auto & dev : model->devices) {
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev.dev, &props);
    LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", ...);
}

// 1128-1141
const int status = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud,
                                    path_model, splits, file, *model, params);
GGML_ASSERT(status <= 0);
if (status < 0) {
    if (status == -1)      LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
    else if (status == -2) LLAMA_LOG_INFO ("%s: cancelled model load\n", __func__);
    llama_model_free(model);                 // 失败统一释放
    return nullptr;
}
return model;                                // 成功返回裸指针，调用方负责 llama_model_free
```

> 注意：设备选择阶段的少数早期 `return nullptr`（如 `n_devs == 0`、张量并行无设备）发生在 `new llama_model` 之后但没有 `llama_model_free`。这些属于参数非法路径，正常调用不会触发；阅读时可留意。

---

## 3. `llama_model_load()`：加载流程编排

`src/llama.cpp:874-929`。返回约定：

```
0  = 成功
-1 = 出错（异常）
-2 = 被 progress_callback 取消
```

### 3.1 计时起点

```cpp
// 877-882
model.t_load_us = 0;
time_meas tm(model.t_load_us);     // RAII：析构时把耗时写入 t_load_us
model.t_start_us = tm.t_start_us;
```

注释说明：真实加载时间会在首次 eval 后重算，以把 mmap 的缺页开销算进去。

### 3.2 构造 `llama_model_loader`（GGUF 解析 + 权重索引）

```cpp
// 884-888
try {
    llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud,
                          fname, splits, file,
                          params.use_mmap, params.use_direct_io,
                          params.check_tensors, params.no_alloc,
                          params.kv_overrides, params.tensor_buft_overrides);
    ml.print_info();
```

这一行是整条链路的信息来源，详见第 4 节。

### 3.3 依次加载 架构 → 超参 → 词表 → 统计

```cpp
// 890-913
model.hparams.vocab_only = params.vocab_only;
model.hparams.no_alloc   = params.no_alloc;

try { model.load_arch(ml); }    catch (...) { throw ... "error loading model architecture" ...; }
try { model.load_hparams(ml); } catch (...) { throw ... "error loading model hyperparameters" ...; }

if (model.arch == LLM_ARCH_CLIP)
    throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");

try { model.load_vocab(ml); }   catch (...) { throw ... "error loading model vocabulary" ...; }

model.load_stats(ml);
model.print_info();
```

顺序不能乱：`load_hparams` 需要 `arch`，`load_vocab` 依赖元数据，`load_tensors` 又依赖前三者。

### 3.4 `vocab_only` 提前返回

```cpp
// 915-918
if (params.vocab_only) {
    LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
    return 0;                       // 只加载词表，不碰权重
}
```

### 3.5 加载张量

```cpp
// 920-922
if (!model.load_tensors(ml)) {
    return -2;                      // 仅当被 progress_callback 取消时返回 false
}
```

### 3.6 统一异常处理

```cpp
// 923-928
} catch (const std::exception & err) {
    LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
    return -1;
}
return 0;
```

所有 `throw` 在这里收敛为 `-1`，由 impl 负责释放 `model`。

---

## 4. `llama_model_loader` 构造函数：把 GGUF 读成内存索引

`src/llama-model-loader.cpp:510-819`。这是最“重”的一步之一。

### 4.1 记录 KV 覆盖与 buffer type 覆盖

```cpp
// 524-535
int trace = 0;
if (getenv("LLAMA_TRACE")) trace = atoi(getenv("LLAMA_TRACE"));

if (param_overrides_p != nullptr) {
    for (const auto * p = param_overrides_p; p->key[0] != 0; p++) {
        kv_overrides.insert({std::string(p->key), *p});
    }
}
tensor_buft_overrides = param_tensor_buft_overrides_p;
```

- `LLAMA_TRACE>0` 会打印每个张量的详细信息。
- `kv_overrides` 允许调用方覆盖模型元数据（例如强制改某些超参）。

### 4.2 从文件加载主 GGUF（`!fname.empty()` 分支）

```cpp
// 537-552
struct ggml_context * ctx = NULL;
struct gguf_init_params params = {
    /*.no_alloc = */ true,     // 只建元数据，不分配张量数据
    /*.ctx      = */ &ctx,     // 同时返回一个 ggml_context，里面是张量“骨架”
};

metadata_ptr.reset(gguf_init_from_file(fname.c_str(), params));
metadata = metadata_ptr.get();
if (metadata == nullptr)
    throw std::runtime_error(format("%s: failed to load model from %s", __func__, fname.c_str()));

get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);   // 读 general.architecture
llm_kv = LLM_KV(llm_arch_from_string(arch_name));                  // 转成枚举
```

关键点：`no_alloc = true`，此时 `ctx` 中每个 `ggml_tensor` 只有形状/类型/名称，`data == nullptr`，真正的数据后面再搬运。

### 4.3 mmap 与 Direct I/O 的互斥处理

```cpp
// 554-569
files.emplace_back(new llama_file(fname.c_str(), "rb", use_direct_io));
contexts.emplace_back(ctx);

if (use_mmap && use_direct_io) {
    if (files.back()->has_direct_io()) {
        LLAMA_LOG_WARN("%s: direct I/O is enabled, disabling mmap\n", __func__);
        use_mmap = false;
    } else {
        LLAMA_LOG_WARN("%s: direct I/O is not available, using mmap\n", __func__);
        use_direct_io = false;
        files.pop_back();
        files.emplace_back(new llama_file(fname.c_str(), "rb", false));  // 用普通 fopen 重新打开以便 mmap
    }
}
```

两者同时开启时：Direct I/O 优先，否则回退 mmap 并重开文件。

### 4.4 建立 `weights_map`（张量名 → 文件/偏移）

```cpp
// 571-583
for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
    std::string tensor_name = cur->name;
    if (weights_map.find(tensor_name) != weights_map.end())
        throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ...));

    n_elements += ggml_nelements(cur);
    n_bytes    += ggml_nbytes(cur);
    weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
}
```

`llama_tensor_weight` 的构造函数（`src/llama-model-loader.h:39-49`）会做两件重要的事：

```cpp
offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size())
    throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ...));
```

即：**计算张量在文件中的绝对偏移，并做越界校验**（防损坏/截断模型）。

`weights_map` 使用 `weight_name_comparer`（`src/llama-model-loader.h:53-64`）排序，按 `blk.%d.` 的层号排序，让后续遍历更接近“逐层”顺序。

### 4.5 分片模型加载（`n_split > 1`）

```cpp
// 584-663
uint16_t n_split = 0;
get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

if (n_split > 1) {
    // 1) 必须从第 0 片开始加载
    uint16_t idx = 0;
    const std::string kv_split_no = llm_kv(LLM_KV_SPLIT_NO);
    get_key(kv_split_no, idx);
    if (idx != 0) throw std::runtime_error(... "model must be loaded with the first split" ...);

    // 2) 若调用方没给 splits，则按命名规则自动推导
    if (splits.empty()) splits = llama_get_list_splits(fname, idx, n_split);

    // 3) 校验数量
    if (n_split != (uint16_t)splits.size())
        throw std::runtime_error(format("invalid split count, given: %zu splits, but expected %d", ...));

    // 4) 逐个加载剩余分片
    for (idx = 1; idx < n_split; idx++) {
        gguf_context_ptr ctx_gguf { gguf_init_from_file(splits[idx].c_str(), split_params) };
        if (!ctx_gguf) throw ...;

        // 校验分片序号与期望一致
        const int kid = gguf_find_key(ctx_gguf.get(), kv_split_no.c_str());
        ...
        int idx_gguf = gguf_get_val_u16(ctx_gguf.get(), kid);
        if (idx_gguf != idx) throw ...;

        files.emplace_back(new llama_file(splits[idx].c_str(), "rb", use_direct_io));
        contexts.emplace_back(ctx);

        // 把该分片的张量也加入 weights_map（注意 idx 记录来源文件）
        for (ggml_tensor * cur = ...) {
            ...
            weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), idx, ctx_gguf.get(), cur));
        }
    }

    // 5) 张量总数校验
    get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);
    if (n_tensors != (int) weights_map.size())
        throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", ...));
}
```

要点：分片模式下 `weights_map` 中每个条目的 `idx` 记录了它来自哪个分片文件，后续按 `idx` 去对应文件 mmap/读取。

### 4.6 从 `FILE *` 加载分支

```cpp
// 664-693
} else if (file != nullptr) {
    metadata_ptr.reset(gguf_init_from_file_ptr(file, params));
    ...
    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));
    files.emplace_back(new llama_file(file));
    contexts.emplace_back(ctx);
    // 同样建立 weights_map（idx 恒为 0）
}
```

### 4.7 仅元数据分支（`llama_model_init_from_user`）

```cpp
// 694-697
} else {
    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));
}
```

此时没有 `files`，张量数据由调用方的 `set_tensor_data` 回调提供。

### 4.8 汇总元数据、推断文件类型、打印 KV

```cpp
// 699-705
n_kv      = gguf_get_n_kv(metadata);
n_tensors = weights_map.size();
fver      = (enum llama_fver) gguf_get_version(metadata);
LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n", ...);
```

**推断量化类型 `ftype`**（`709-778`）：统计每种 `ggml_type` 的张量数量，取数量最多的类型映射为 `LLAMA_FTYPE_*`，再打上 `LLAMA_FTYPE_GUESSED` 标记；如果 GGUF 里有 `general.file_type`，则以它为准覆盖：

```cpp
ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);
uint32_t ftype_val = 0;
if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false))
    ftype = (llama_ftype) ftype_val;
```

随后（`780-807`）把所有 KV 与类型统计打印出来（注意日志说明：此输出不应用 KV overrides）。

### 4.9 收尾：mmap 支持检查与成员赋值

```cpp
// 810-819
if (!llama_mmap::SUPPORTED) {
    LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
    use_mmap = false;
}
this->use_mmap      = use_mmap;
this->use_direct_io = use_direct_io;
this->check_tensors = check_tensors;
this->no_alloc      = no_alloc;
```

`print_info()`（`src/llama-model-loader.cpp:1687`）打印文件格式、文件类型、文件大小与 BPW。

---

## 5. `load_arch` / `load_hparams` / `load_vocab` / `load_stats`

### 5.1 `load_arch`（`src/llama-model.cpp:683`）

```cpp
void llama_model::load_arch(llama_model_loader & ml) {
    arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN)
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");

    if (!devices.empty() && devices[0].is_meta && !llm_arch_supports_sm_tensor(arch))
        throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
}
```

- 把 `general.architecture` 字符串（如 `llama`、`qwen2`、`deepseek2`）转成 `llm_arch` 枚举。
- 若启用了张量并行但该架构未实现对应切分逻辑，直接报错。

### 5.2 `load_hparams`（`src/llama-model.cpp:693`）

分两大部分：

1. **先把所有非数组 KV 以字符串形式存进 `gguf_kv`**（`696-705`），便于后续查询/调试。
2. **按架构读取超参数**。开头读通用字段：

```cpp
ml.get_key(LLM_KV_GENERAL_NAME, name, false);

// 词表模式或 CLIP 到此为止
if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) return;

ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,   false);
ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer);
ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,      false);
ml.get_key(LLM_KV_EXPERT_USED_COUNT,       hparams.n_expert_used, false);
...
```

随后读取逐层的 `n_ff`、`n_head`、`n_head_kv`（`get_key_or_arr` 支持“标量或数组”两种写法），以及 RoPE 相关参数：

```cpp
ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);
hparams.n_head_kv_arr = hparams.n_head_arr;
ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);
hparams.rope_freq_base_train = 10000.0f;
ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);
```

之后是一段很长的 `switch (arch)`，针对不同模型族设置默认值、读取专有字段、做一致性校验。核心原则：**通用键先读，架构特有键在 switch 中补齐，最后断言自洽**（例如专家数量、`n_rot` 与 head dim 的关系）。

### 5.3 `load_vocab`（`src/llama-model.cpp:2952`）

```cpp
void llama_model::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);
    vocab.load(ml, kv);
}
```

转发到 `llama_vocab::impl::load`（`src/llama-vocab.cpp:1749`），核心是：

1. 读 `tokenizer.ggml.model`、`tokenizer.ggml.pre`，确定分词器类型。
2. 按类型构建：
   - `llama` → SPM（SentencePiece）
   - `bert` → WPM（WordPiece）
   - `gpt2` / `gemma4` → BPE（读 `merges` 构建 `bpe_ranks`）
   - `t5` → UGM（读 `precompiled_charsmap`）
   - `rwkv`、`plamo2` 等 → 各自类型
   - `no_vocab`/`none` → `LLAMA_VOCAB_TYPE_NONE`，可创建占位 token
3. 读 token 列表、score、type、special tokens 等，最终得到 `id_to_token` / `token_to_id`。

### 5.4 `load_stats`（`src/llama-model.cpp:678`）

```cpp
void llama_model::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes    = ml.n_bytes;
}
```

把 loader 统计的元素数/字节数存到模型，供 `n_elements()` 等 API 使用。

`print_info()`（`src/llama-model.cpp:8096`）会打印架构、`n_ctx_train`、`n_embd`、`n_layer`、`n_head`、`n_head_kv`、`n_rot`、`n_swa` 等超参，是排查“模型是否被正确识别”的第一手日志。

---

## 6. `load_tensors()`：建张量、分配 buffer、搬运权重

`src/llama-model.cpp:2958-8035`。这是最长的一个函数，主体是**按架构创建各层张量**（几百行到数千行的 `switch (arch)`），但骨架步骤很清晰。可分成 7 个阶段。

### 6.1 准备 buffer type 列表

```cpp
// 2958-2978
const auto & split_mode   = params.split_mode;
const auto & use_mlock    = params.use_mlock;
const auto & tensor_split = params.tensor_split;
const int n_layer      = hparams.n_layer;
const int n_gpu_layers = this->n_gpu_layers();
const bool use_mmap_buffer = true;

LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s, direct_io = %s)\n", ...);

// CPU 侧可选 buffer 类型（含 extra bufts / host buffer 等）
pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);

// 每个 GPU 设备一份候选 buffer 类型，最后追加 CPU 类型作为 fallback
for (const auto & dev : devices) {
    buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
    buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
    pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
}
```

`n_gpu_layers()`（`src/llama-model.cpp:8065`）：

```cpp
return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer + 1;
```

即默认 `-1` 表示**全部层（含输出层）上 GPU**。

### 6.2 计算各设备切分比例

```cpp
// 2985-3016
bool all_zero = tensor_split == nullptr ||
                std::all_of(tensor_split, tensor_split + n_devices(), [](float x){ return x == 0.0f; });
std::vector<float> splits(n_devices());

if (all_zero) {
    // 默认按各设备“空闲显存”比例切
    for (size_t i = 0; i < n_devices(); ++i) {
        ggml_backend_dev_t dev = devices[i].dev;
        size_t total, free;
        ggml_backend_dev_memory(dev, &free, &total);
        if (free == 0 && total == 0)          // 设备不上报内存时用主机内存兜底
            ggml_backend_dev_memory(cpu_dev, &free, &total);
        splits[i] = free;
    }
} else {
    std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
}

// 前缀和 → 归一化 → 得到 [0,1] 的分割点
float split_sum = 0.0f;
for (size_t i = 0; i < n_devices(); ++i) { split_sum += splits[i]; splits[i] = split_sum; }
for (size_t i = 0; i < n_devices(); ++i) splits[i] /= split_sum;
```

### 6.3 决定每层落在哪个设备

```cpp
// 3018-3030
const int i_gpu_start   = std::max(int(hparams.n_layer) + 1 - n_gpu_layers, 0);
const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, int(n_layer) + 1);

auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
    const bool is_swa = il < int(hparams.n_layer) && hparams.is_swa(il);
    if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
        // 落在 CPU 的层
        return {cpu_dev, &pimpl->cpu_buft_list};
    }
    // 用分割点判断落在哪个 GPU
    const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(),
                                           float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
    auto * dev = devices.at(layer_gpu).dev;
    return {dev, &pimpl->gpu_buft_list.at(dev)};
};
```

然后分配：

```cpp
// 3032-3043
pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };   // 输入层始终 CPU
pimpl->dev_layer.resize(n_layer);
for (int il = 0; il < n_layer; ++il)
    pimpl->dev_layer[il] = get_layer_buft_list(il);
pimpl->dev_output = get_layer_buft_list(n_layer);        // 输出层
```

> 注释明确说明：输入层 offload 收益很小，所以固定放 CPU。

### 6.4 创建张量骨架（按架构 switch）

先定义统一的 `create_tensor` lambda：

```cpp
// 3073-3078
auto create_tensor = [&](const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor * {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
};
```

- `tn` 是 `LLM_TN`（tensor name），把逻辑张量名（如 `LLM_TENSOR_ATTN_Q`）+ 后缀 + 层号拼成 GGUF 里的真实名字。
- `ne` 是期望维度，`create_tensor` 会与文件里的元数据做形状校验。
- `flags` 支持：`TENSOR_NOT_REQUIRED`（可缺省）、`TENSOR_DUPLICATED`（复用/复制）、`TENSOR_SKIP`、`TENSOR_SKIP_IF_VIRTUAL`。

随后是巨大的 `switch (arch)`（从约 `3110` 行一直到 `7888` 行），为每种架构逐层创建 `tok_embd`、`output_norm`、`output`、`attn_norm`、`wq/wk/wv/wo`、`ffn_*`、`rope_freqs`、MoE 的 `ffn_*_exps`、SSM 相关张量等。以 LLaMA 为例（`3121-3166` 附近）：

```cpp
tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

for (int i = 0; i < n_layer; ++i) {
    auto & layer = layers[i];
    layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
    create_tensor_qkv(layer, i, n_embd, n_embd_head_k*n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
    layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k*n_head, n_embd}, 0);
    layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
    layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
    layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
    layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
}
```

辅助函数处理“合并/分离”写法：

- `create_tensor_qkv`（`3095`）：优先找合并的 `attn_qkv`，找不到再分别建 `wq/wk/wv`（兼容不同导出格式）。
- `create_tensor_gate_up_exps`（`3086`）：优先找合并的 `ffn_gate_up_exps`，否则分开建 gate/up。

> 这一段代码量最大，但模式高度重复：**按架构定义、按层号循环、必要时用 `TENSOR_NOT_REQUIRED` 兼容不同版本权重**。

### 6.5 收尾校验：张量数量必须匹配

```cpp
// 7890-7897
ml.done_getting_tensors();

// populate tensors_by_name
for (auto & [_, ctx_ptr] : ml.ctx_map) {
    for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur; cur = ggml_get_next_tensor(ctx_ptr.get(), cur))
        tensors_by_name.emplace_back(ggml_get_name(cur), cur);
}
```

`done_getting_tensors()`（`src/llama-model-loader.cpp:1315`）做关键一致性检查：

```cpp
void llama_model_loader::done_getting_tensors() const {
    if (n_created != n_tensors)
        throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
    if (n_tensors_moved > 0) {
        LLAMA_LOG_DEBUG("%s: tensor '%s' ... cannot be used with preferred buffer type %s, using %s instead\n", ...);
    }
}
```

含义：**模型里声明的张量必须全部被“认领”**，否则说明架构解析与实际权重不匹配（常见于不支持的变体或损坏文件）。

### 6.6 建立 mmap 映射并分配后端 buffer

```cpp
// 7899-7900
ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
pimpl->mappings.reserve(ml.mappings.size());
```

`init_mappings`（`src/llama-model-loader.cpp:1326`）在 `use_mmap` 时对每个文件建立 `llama_mmap`，可选 mlock。

核心分配循环（`7910-7991`）：

```cpp
for (auto & [buft, ctx_ptr] : ml.ctx_map) {          // 每个 buffer type 一个 ggml_context
    ggml_context * ctx = ctx_ptr.get();
    if (ggml_get_first_tensor(ctx) == nullptr) continue;

    // 判断能否用 mmap 的 host 指针直接做 buffer
    bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
    bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

    if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
        // 用 mmap 区间直接构造 backend buffer，零拷贝
        for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
            ml.get_mapping_range(&first, &last, &addr, idx, ctx);
            if (first >= last) continue;
            ggml_backend_buffer_t buf =
                ggml_backend_dev_buffer_from_host_ptr(dev, (char *)addr + first, last - first, max_size);
            ...
            buf_map.emplace(idx, buf);
        }
    } else {
        if (ml.no_alloc) {
            // 只模拟分配：给每个张量挂 dummy buffer
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ...; t; ...) t->buffer = buf;
        } else {
            // 真正按 ctx 里所有张量一次性分配 buffer
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (use_mlock && ggml_backend_buffer_is_host(buf)) { /* mlock */ }
        for (uint32_t idx = 0; idx < ml.files.size(); idx++) buf_map.emplace(idx, buf);
    }

    // 标记 buffer 用途为“权重”，供调度器优先把相关算子放到该设备
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
}
```

`get_mapping_range`（`src/llama-model-loader.cpp:1359`）会遍历该 `ctx` 中属于第 `idx` 个文件的张量，求出最小/最大偏移，从而**只映射真正用到的 mmap 区间**（对 Metal/Apple Silicon 的部分 offload 很关键）。

随后打印 offload 情况与各 buffer 大小（`7993-8015`）：

```cpp
LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", ...);
LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n", ...);
```

若 `no_alloc`，到此直接返回（`8017-8019`），只做“内存占用模拟”。

### 6.7 搬运权重数据

```cpp
// 8021-8034
for (auto & [ctx, buf_map] : ctx_buf_maps) {
    if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL,
                          params.progress_callback, params.progress_callback_user_data)) {
        return false;                 // 被取消
    }
}

if (use_mmap_buffer) {
    for (auto & mapping : ml.mappings)
        pimpl->mappings.emplace_back(std::move(mapping));   // 保留 mmap 生命周期
}
return true;
```

---

## 7. `llama_model_loader::create_tensor()`：张量如何绑定到 buffer type

`src/llama-model-loader.cpp:1045-1285`。这是连接“逻辑张量”和“物理设备”的关键函数。

### 7.1 每个 buffer type 一个 `ggml_context`

```cpp
auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
    auto it = ctx_map.find(buft);
    if (it == ctx_map.end()) {
        int max_n_tensors = n_tensors;
        max_n_tensors += 1;                  // duplicated output
        max_n_tensors += hparams.n_layer*2;  // duplicated rope freq
        if (files.empty()) max_n_tensors += hparams.n_layer*256;  // 外部构造留足余量
        const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;

        ggml_init_params params = { ctx_size, NULL, /*no_alloc=*/true };
        ggml_context * ctx = ggml_init(params);
        ctx_map.emplace(buft, ctx);
        return ctx;
    }
    return it->second.get();
};
```

为什么按 buffer type 分 ctx？因为 backend 分配 buffer 时是“按 ctx 批量分配”，同一 buffer type 的张量放在同一 ctx 里可以一次分配，且便于 mmap 零拷贝。

### 7.2 为单个张量选 buffer type

`buft_for_tensor`（`1078-1211`）依次做：

1. **缺失处理**：`t_meta == nullptr` 时，若 `TENSOR_NOT_REQUIRED` 返回 `nullptr`，否则抛错。
2. **重复张量特判**：`token_embd` 且 `TENSOR_DUPLICATED` 时按 `output` 处理。
3. **查张量信息**：`llm_tensor_info_for(tn_tensor)` 得到 `op` 和 `layer`（INPUT / OUTPUT / REPEATING）。
4. **跳过无用张量**：`info.op == GGML_OP_NONE || TENSOR_SKIP` → 记录日志、`size_data -= nbytes`、`n_created++`、返回 `nullptr`。
5. **bias 修正**：bias 张量的 `op` 从 `MUL_MAT` 改为 `ADD`，MoE 的 `MUL_MAT_ID` 改为 `ADD_ID`。
6. **层类型校验**：INPUT/OUTPUT 层不能带 `bid`，REPEATING 必须带 `bid`。
7. **选择候选列表**：按 `info.layer` 选 `buft_list_input` / `buft_list_output` / `buft_list_layer`。
8. **应用覆盖规则**：遍历 `tensor_buft_overrides`，用正则匹配张量名；若覆盖到 CPU buffer，还会调用 `select_weight_buft(...)` 选择更合适的 CPU extra buffer（并提示 mmap 下性能问题）。
9. **默认选择**：`select_weight_buft(hparams, t_meta, op, buft_list)`。
10. **mmap 特例**：`use_mmap` 时避免用 host buffer，改回 CPU buffer type。
11. **统计“被迫换 buffer”的张量**（`n_tensors_moved`）。

### 7.3 两种创建路径

**无文件（外部构造）**（`1213-1249`）：根据 `ne` 和 GGUF 中记录的类型手工构造 `ggml_tensor` 元数据，然后 `ggml_dup_tensor` 到目标 ctx。

**有文件（标准路径）**（`1251-1284`）：

```cpp
ggml_tensor * t_meta = get_tensor_meta(tn.str().c_str());
ggml_backend_buffer_type_t buft = buft_for_tensor(t_meta);
if (buft == nullptr) return nullptr;

ggml_context * ctx = ctx_for_buft(buft);

// 重复张量：若同 ctx 里已存在同名张量，直接复用
if (flags & TENSOR_DUPLICATED) {
    if (ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str())) return t;
}

// 形状校验（与 GGUF 元数据比对）
const ggml_tensor * cur = check_tensor_dims(tn.str(), ne, !(flags & TENSOR_NOT_REQUIRED));
if (cur == NULL) return NULL;

// 在目标 ctx 中创建同形状张量（no_alloc，无数据）
ggml_tensor * tensor = ggml_dup_tensor(ctx, cur);
ggml_set_name(tensor, ggml_get_name(cur));

if (flags & TENSOR_DUPLICATED) size_data += ggml_nbytes(cur);  // 复制也要占数据量
else                          n_created++;

return tensor;
```

`check_tensor_dims`（`src/llama-model-loader.cpp:862`）逐维比较期望 `ne` 与文件中的 `cur->ne`，不一致即抛“has wrong shape”，这是**加载期最早发现权重与架构不匹配的地方**。

---

## 8. `load_all_data()`：把权重搬进 buffer

`src/llama-model-loader.cpp:1399-1681`。根据是否 mmap、是否 host buffer、是否支持异步上传，走三条路径。

### 8.1 外部构造：直接回调

```cpp
// 1405-1410
if (files.empty()) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
        set_tensor_data(t, set_tensor_data_ud);       // 调用方填充数据
    return true;
}
```

### 8.2 尝试启用异步上传

```cpp
// 1416-1505
constexpr size_t n_buffers = 4;                        // 4 个 1MB 级 staging buffer
size_t alignment = max(所有文件 read_alignment());
const size_t buffer_size = alignment != 1 ? 64MB + 2*alignment : 1MB;

auto * upload_backend = [&](...) {
    if (use_mmap || check_tensors) return nullptr;     // mmap / 校验时不做异步
    // 需要：buffer 有 device、是默认 buffer type、设备支持 async/host_buffer/events
    // 然后创建 n_buffers 个 pinned host buffer + event + backend
}(...);
```

只有在**非 mmap、非校验、设备支持异步**时才会启用 pinned memory + event 的异步上传。

### 8.3 主循环：逐张量搬运

```cpp
// 1514-1635
for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
    const auto * weight = get_weight(ggml_get_name(cur));
    if (weight == nullptr) continue;                    // 分片专家模型可能出现

    if (progress_callback && !progress_callback((float)size_done/size_data, user_data))
        return false;                                   // 用户取消

    size_t n_size = ggml_nbytes(cur);

    if (use_mmap) {
        // 路径 A：mmap
        uint8_t * data = (uint8_t *)mapping->addr() + weight->offs;
        if (check_tensors) 异步校验;
        if (buf_mmap && cur->data == nullptr)
            ggml_backend_tensor_alloc(buf_mmap, cur, data);   // 直接把张量指向 mmap 地址（零拷贝）
        else
            ggml_backend_tensor_set(cur, data, 0, n_size);    // 否则复制到 buffer
    } else {
        const auto & file = files.at(weight->idx);
        if (ggml_backend_buffer_is_host(cur->buffer)) {
            // 路径 B：host buffer，普通 read
            file->seek(weight->offs, SEEK_SET);
            file->read_raw(cur->data, n_size);
        } else if (upload_backend) {
            // 路径 C：读入 pinned 内存 → 异步上传 GPU（分块、对齐、事件同步）
            ...
            ggml_backend_tensor_set_async(upload_backend, cur, ptr_data, data_read, data_to_copy);
            ggml_backend_event_record(events[buffer_idx], upload_backend);
        } else {
            // 路径 D：读入临时 buffer 再 set
            read_buf.resize(n_size);
            file->read_raw(read_buf.data(), n_size);
            ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
        }
    }

    size_done += n_size;
}
```

### 8.4 收尾：校验结果、释放临时资源、解除多余 mmap

```cpp
// 1637-1680
// 1) 等待所有异步校验 future
for (auto & future : validation_result) {
    auto result = future.get();
    if (!result.second) { LLAMA_LOG_ERROR(... "has invalid data"); validation_failed = true; }
}
if (validation_failed) throw std::runtime_error("found tensors with invalid data");

// 2) 全部加载完成后，把 mmap 中“未使用”的区间解除映射
if (size_done >= size_data) {
    for (...) {
        mapping->unmap_fragment(0, mmap_used.first);
        if (mmap_used.second != 0) mapping->unmap_fragment(mmap_used.second, mapping->size());
    }
    if (progress_callback) return progress_callback(1.0f, user_data);  // 允许最后一次取消
}
return true;
```

> 这是“懒加载”的体现：mmap 模式下只把真正用到的区间保留映射，其余在加载完成后释放，降低虚拟内存压力。

---

## 9. 错误处理与生命周期

| 返回/异常 | 含义 | 处理位置 |
|---|---|---|
| `status == 0` | 成功 | `llama_model_load` 返回 0 |
| `status == -1` | 加载异常（GGUF 损坏、形状不符、未知架构等） | `catch (std::exception)`，impl 中 `llama_model_free(model)` |
| `status == -2` | 被 `progress_callback` 取消 | `load_tensors` 返回 false / `load_all_data` 返回 false，impl 中释放 |
| `return nullptr`（impl） | 参数/设备非法或加载失败 | 调用方需判空 |

- 成功时返回 `llama_model *`，**由调用方负责 `llama_model_free()`**（`src/llama-model.cpp:9096` 即 `delete model`）。
- `llama_model` 的析构会释放 LoRA；`pimpl` 中的 `ctxs_bufs`、`mappings`、`mlock_*` 通过智能指针/容器自动释放。
- 加载中创建的 `ggml_context`、backend buffer、mmap 都通过 `pimpl` 持有，保证与模型同生命周期。

---

## 10. 关键数据结构速查

| 结构 | 位置 | 作用 |
|---|---|---|
| `llama_model_params` | `include/llama.h:286` | 加载参数（设备、GPU 层数、mmap、回调等） |
| `llama_model` | `src/llama-model.h` | 模型对象：`arch`、`hparams`、`vocab`、`layers`、`devices`、`pimpl` |
| `llama_model::impl` | `src/llama-model.cpp` | 持有 `ctxs_bufs`、`mappings`、`cpu/gpu_buft_list`、`dev_input/dev_layer/dev_output` |
| `llama_model_loader` | `src/llama-model-loader.h:31` | GGUF 解析结果：`metadata`、`files`、`contexts`、`weights_map`、`ctx_map` |
| `llama_tensor_weight` | `src/llama-model-loader.h:33` | 张量来源：文件索引 `idx`、偏移 `offs`、元数据 `tensor` |
| `llama_hparams` | `src/llama-hparams.h` | 超参：`n_ctx_train`、`n_embd`、`n_layer`、`n_head`、RoPE 等 |
| `llama_vocab` | `src/llama-vocab.h` | 分词器：`id_to_token`、`token_to_id`、special tokens |

---

## 11. 时序图（ASCII）

```
调用方
  │ llama_model_load_from_file(path, params)
  ▼
llama_model_load_from_file_impl
  │ 1. 校验 metadata/path/file 三选一
  │ 2. ggml_time_init + backend 检查
  │ 3. 安装默认 progress_callback
  │ 4. new llama_model(params)
  │ 5. 构建设备列表 (devices / 自动 / split_mode / main_gpu)
  ▼
llama_model_load
  │ 6. llama_model_loader 构造
  │      ├─ gguf_init_from_file(no_alloc)
  │      ├─ 读 general.architecture → llm_arch
  │      ├─ 处理 mmap/direct_io
  │      ├─ 建 weights_map（偏移 + 越界校验）
  │      ├─ 加载分片 (n_split)
  │      └─ 推断 ftype + 打印 KV
  │ 7. load_arch
  │ 8. load_hparams
  │ 9. (CLIP 拒绝)
  │10. load_vocab
  │11. load_stats + print_info
  │12. vocab_only? ──是──▶ 返回 0
  │13. load_tensors
  │      ├─ 建 cpu/gpu buft 列表
  │      ├─ 计算 split 比例
  │      ├─ 层 → 设备分配
  │      ├─ switch(arch) 创建张量骨架 (create_tensor)
  │      ├─ done_getting_tensors (数量校验)
  │      ├─ init_mappings
  │      ├─ 按 buft 分配 buffer（mmap 零拷贝 / alloc_ctx_tensors / dummy）
  │      └─ load_all_data（mmap / host read / async upload）
  ▼
返回 llama_model*  ──▶ 调用方后续 llama_model_free()
```

---

## 12. 阅读建议（按关注点）

- **只想理解整体**：第 0、2、3、11 节即可。
- **关注 GGUF/分片解析**：第 4 节 + `src/llama-model-loader.h`。
- **关注 GPU offload/多卡切分**：第 2.5、6.1–6.3 节。
- **关注张量创建与 buffer 选择**：第 6.4、6.6、7 节。
- **关注权重搬运性能（mmap/异步）**：第 8 节。
- **关注词表**：第 5.3 节 + `src/llama-vocab.cpp:1749`。

---

### 备注

- 本文件是对现有源码的**学习性梳理**，不修改任何代码。
- 若源码行号随上游更新发生变化，请以函数名为锚点（如 `llama_model_load_from_file_impl`、`llama_model::load_tensors`）重新定位。

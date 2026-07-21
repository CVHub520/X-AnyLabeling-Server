# 配置指南

X-AnyLabeling-Server 通过服务配置、模型清单和单模型配置三个层级管理运行行为。

| 文件 | 用途 | 默认位置 |
| --- | --- | --- |
| `server.yaml` | 服务、日志、安全、性能与并发 | `configs/server.yaml` |
| `models.yaml` | 控制启动时加载哪些模型 | `configs/models.yaml` |
| `{model_id}.yaml` | 单个模型的参数和客户端控件 | `configs/auto_labeling/{model_id}.yaml` |

## 1. 自定义配置路径

默认读取仓库中的 `configs/`。可通过命令行指定其他文件：

```bash
x-anylabeling-server \
  --config /path/to/custom/server.yaml \
  --models-config /path/to/custom/models.yaml
```

也可以使用环境变量：

```bash
export XANYLABELING_SERVER_CONFIG=/path/to/custom/server.yaml
export XANYLABELING_MODELS_CONFIG=/path/to/custom/models.yaml
x-anylabeling-server
```

> [!NOTE]
> 命令行参数优先于环境变量；自定义文件只覆盖显式提供的字段。

## 2. 服务配置

`configs/server.yaml` 的完整结构如下：

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

logging:
  level: "INFO"
  console_level: "INFO"
  file_enabled: true
  file_path: null
  rotation: "500 MB"
  retention: "30 days"
  format: "text"

security:
  api_key_enabled: false
  api_key: ""
  api_key_header: "Token"
  cors_origins:
    - "*"

performance:
  request_timeout: 300
  max_image_size: 0
  rate_limit_enabled: false
  rate_limit: "100/minute"

concurrency:
  max_workers: 4
  max_queue_size: 50
```

### 2.1 服务监听

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `host` | String | `"0.0.0.0"` | 监听地址；只允许本机访问时使用 `127.0.0.1` |
| `port` | Integer | `8000` | HTTP 端口 |
| `workers` | Integer | `1` | Uvicorn 工作进程数 |

每个进程可能独立加载模型。使用 GPU 模型时不要盲目增加 `workers`，否则会重复占用显存。

### 2.2 日志

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `level` | String | `"INFO"` | 文件日志级别 |
| `console_level` | String | `"INFO"` | 控制台日志级别 |
| `file_enabled` | Boolean | `true` | 是否写入日志文件 |
| `file_path` | String/null | `null` | 自定义日志路径；空值表示自动生成 |
| `rotation` | String | `"500 MB"` | 按大小或时间轮转 |
| `retention` | String | `"30 days"` | 日志保留时间 |
| `format` | String | `"text"` | `text` 或结构化 `json` |

生产环境建议启用轮转和保留期限，避免日志持续占满磁盘。

### 2.3 安全

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `api_key_enabled` | Boolean | `false` | 是否启用 API Key |
| `api_key` | String | `""` | API Key；优先使用环境变量 |
| `api_key_header` | String | `"Token"` | 携带 API Key 的 Header 名称 |
| `cors_origins` | List | `["*"]` | 允许访问的跨域来源 |

使用环境变量覆盖密钥：

```bash
export XANYLABELING_API_KEY="your-secret-key"
```

生产环境示例：

```yaml
security:
  api_key_enabled: true
  api_key: ""
  api_key_header: "Token"
  cors_origins:
    - "https://labeling.example.com"
```

> [!NOTE]
> 即使启用了 API Key，`/health` 也不要求身份验证。

### 2.4 性能限制

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `request_timeout` | Integer | `300` | 请求超时秒数；`0` 表示不限制 |
| `max_image_size` | Integer | `0` | 图像大小上限（MB）；`0` 表示不限制 |
| `rate_limit_enabled` | Boolean | `false` | 是否启用限流 |
| `rate_limit` | String | `"100/minute"` | `{次数}/{时间单位}` |

可用时间单位包括 `second`、`minute`、`hour` 和 `day`。

### 2.5 并发

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `max_workers` | Integer | `4` | 同时执行的推理任务数 |
| `max_queue_size` | Integer | `50` | 等待队列上限；达到上限时返回 HTTP 503 |

并发值应根据模型显存占用和单次推理耗时调节。增加并发并不一定提高 GPU 吞吐量。

## 3. 模型配置

### 3.1 全局模型清单

编辑 `configs/models.yaml` 控制启动时加载的模型：

```yaml
enabled_models:
  - yolo11n
  - yolo11n_seg
  - yolo11n_pose
  # - yolo11n_obb
  - qwen3vl_caption_transformers
```

- 模型按照列表顺序加载并显示。
- 注释掉不使用的模型可减少启动时间和资源占用。
- 每个 `model_id` 必须全局唯一。

### 3.2 单模型配置

在 `configs/auto_labeling/{model_id}.yaml` 中定义模型：

```yaml
model_id: your_model_id
display_name: "Your Model Name"
batch_processing_mode: "default"

capabilities:
  ppocr_pipeline: true

params:
  model_path: "path/to/weights.pt"
  device: "cuda:0"
  conf_threshold: 0.25

widgets:
  - name: button_run
    value: null
  - name: edit_conf
    value: 0.25
  - name: edit_iou
    value: 0.45
  - name: toggle_preserve_existing_annotations
    value: false
```

#### 基本字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `model_id` | 是 | 全局唯一，通常与文件名一致 |
| `display_name` | 是 | 桌面端显示名称 |
| `batch_processing_mode` | 否 | `default`、`text_prompt` 或 `video` |
| `capabilities` | 否 | 供专项客户端模块识别的稳定能力元数据 |
| `params` | 否 | 传递给模型实现的运行参数 |
| `widgets` | 否 | 桌面端展示的交互控件 |

#### 批处理模式

| 模式 | 说明 |
| --- | --- |
| `default` | 不需要文本提示的常规批处理 |
| `text_prompt` | 执行前要求用户输入文本提示 |
| `video` | 使用会话接口处理视频序列 |

#### 常用控件

| 控件 | 值类型 | 作用 |
| --- | --- | --- |
| `button_run` | null | 执行推理 |
| `button_send` | null | 提交文本提示 |
| `edit_text` | string | 文本提示输入框 |
| `edit_conf` | float | 置信度阈值，必须提供默认值 |
| `edit_iou` | float | NMS IoU 阈值，必须提供默认值 |
| `toggle_preserve_existing_annotations` | bool | 是否保留已有标注 |
| `button_add_point` | null | 添加正点提示 |
| `button_remove_point` | null | 添加负点提示 |
| `button_add_rect` | null | 添加矩形提示 |
| `button_clear` | null | 清空提示 |
| `button_finish_object` | null | 完成当前目标 |
| `mask_fineness_slider` | int | 掩码细化程度，范围 `1-100` |
| `add_pos_rect` | null | 添加正矩形提示 |
| `add_neg_rect` | null | 添加负矩形提示 |
| `button_run_rect` | null | 执行矩形提示推理 |
| `remote_task_select_combobox` | null | 多任务模型的任务选择器 |

### 3.3 多任务模型

Rex-Omni、PaddleOCR-VL 等模型可以通过 `get_metadata()` 返回 `available_tasks`。每个任务可声明：

- 唯一 `id`、显示名称和说明；
- 独立的 `batch_processing_mode`；
- 当前任务启用的控件；
- 控件占位符、提示和默认值。

桌面端通过 `remote_task_select_combobox` 切换任务，不需要为每个任务重复加载模型。

## 4. 故障排查

### 服务无法启动

- 使用 YAML 校验器检查语法和缩进。
- 确认端口未被占用：`lsof -i :8000`。
- 检查模型配置中的必填字段和文件路径。
- 检查 PyTorch、CUDA 与驱动版本是否匹配。

### 内存或显存占用过高

- 降低 `max_workers`。
- 减少 Uvicorn `workers`，避免重复加载模型。
- 禁用未使用的模型。
- 使用更小的模型或输入尺寸。

### 响应缓慢

- 检查队列长度和模型推理耗时。
- 根据硬件能力调整并发，而不是无限提高 `max_workers`。
- 对 API 模型检查网络延迟和上游限流。

### 队列已满

- 在资源允许时增加 `max_queue_size` 或 `max_workers`。
- 在客户端加入有限次数的退避重试。
- 高负载场景部署多个服务实例并使用负载均衡。

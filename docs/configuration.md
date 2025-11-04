# Configuration Guide

This guide covers all configuration options for X-AnyLabeling-Server.

| File | Purpose | Location |
|------|---------|----------|
| `server.yaml` | Server, logging, security, and performance settings | `configs/server.yaml` |
| `models.yaml` | Enable/disable models | `configs/models.yaml` |
| `{model_id}.yaml` | Individual model configuration | `configs/auto_labeling/{model_id}.yaml` |

## 0. Table of Contents

* [1. Server Configuration](#1-server-configuration)
   * [1.1 Server Settings](#11-server-settings)
   * [1.2 Logging Settings](#12-logging-settings)
   * [1.3 Security Settings](#13-security-settings)
   * [1.4 Performance Settings](#14-performance-settings)
   * [1.5 Concurrency Settings](#15-concurrency-settings)
* [2. Model Configuration](#2-model-configuration)
   * [2.1 Global Model Configuration](#21-global-model-configuration)
   * [2.2 Individual Model Configuration](#22-individual-model-configuration)
* [3. Troubleshooting](#3-troubleshooting)
   * [3.1 Server Won't Start](#31-server-wont-start)
   * [3.2 High Memory Usage](#32-high-memory-usage)
   * [3.3 Slow Response Times](#33-slow-response-times)
   * [3.4 Queue Full Errors](#34-queue-full-errors)

## 1. Server Configuration

Edit `configs/server.yaml` to configure server behavior.

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

### 1.1 Server Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `host` | String | `"0.0.0.0"` | Server bind address (use `127.0.0.1` for localhost only) |
| `port` | Integer | `8000` | Server port number |
| `workers` | Integer | `1` | Number of Uvicorn worker processes |

### 1.2 Logging Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | String | `"INFO"` | File log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `console_level` | String | `"INFO"` | Console log level (independent from file level) |
| `file_enabled` | Boolean | `true` | Enable file logging |
| `file_path` | String/null | `null` | Log file path (null = auto-generate with timestamp) |
| `rotation` | String | `"500 MB"` | Log rotation: by size (`"100 MB"`) or time (`"1 day"`) |
| `retention` | String | `"30 days"` | Log retention period (auto-delete old logs) |
| `format` | String | `"text"` | Log format: `json` (structured) or `text` (human-readable) |

### 1.3 Security Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_key_enabled` | Boolean | `false` | Enable API key authentication |
| `api_key` | String | `""` | API key (prefer environment variable) |
| `api_key_header` | String | `"Token"` | HTTP header name for API key |
| `cors_origins` | List | `["*"]` | Allowed CORS origins |

**Environment Variable Support:**

Set `XANYLABELING_API_KEY` environment variable to override `api_key` in config:

```bash
export XANYLABELING_API_KEY="your-secret-key"
```

**Examples:**

```yaml
# Development: No authentication
security:
  api_key_enabled: false
  cors_origins:
    - "*"

# Production: With authentication
security:
  api_key_enabled: true
  api_key: ""  # Set via environment variable
  api_key_header: "Token"
  cors_origins:
    - "https://yourdomain.com"
    - "https://app.yourdomain.com"

# Internal network: Restrict origins
security:
  api_key_enabled: false
  cors_origins:
    - "http://localhost:3000"
    - "http://192.168.1.100"
```

> ![NOTE]
> The `/health` endpoint does not require authentication, even when `api_key_enabled: true`.

### 1.4 Performance Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `request_timeout` | Integer | `300` | Request timeout in seconds (0 = no timeout) |
| `max_image_size` | Integer | `0` | Maximum image size in MB (0 = unlimited) |
| `rate_limit_enabled` | Boolean | `false` | Enable rate limiting |
| `rate_limit` | String | `"100/minute"` | Rate limit format: `"{count}/{unit}"` |

**Rate Limit Units:** `second`, `minute`, `hour`, `day`

**Examples:**

```yaml
# Development: No limits
performance:
  request_timeout: 0
  max_image_size: 0
  rate_limit_enabled: false

# Production: With limits
performance:
  request_timeout: 60
  max_image_size: 10
  rate_limit_enabled: true
  rate_limit: "50/minute"

# High-throughput: Longer timeout, higher rate limit
performance:
  request_timeout: 300
  max_image_size: 20
  rate_limit_enabled: true
  rate_limit: "500/minute"
```

### 1.5 Concurrency Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_workers` | Integer | `4` | Maximum concurrent inference tasks |
| `max_queue_size` | Integer | `50` | Maximum queued requests (returns 503 when full) |

## 2. Model Configuration

### 2.1 Global Model Configuration

Edit `configs/models.yaml` to control which models are loaded on startup:

```yaml
enabled_models:
  - yolo11n
  - yolo11n_seg
  - yolo11n_pose
  # - yolo11n_obb  # Comment out to disable
  - qwen3vl_caption_transformers
```

**Important:**
- Models are displayed in X-AnyLabeling UI in the order listed
- Comment out models you don't need to save resources
- Each `model_id` must be unique across all configurations

### 2.2 Individual Model Configuration

Create `configs/auto_labeling/{model_id}.yaml` for each model with the following structure:

```yaml
model_id: your_model_id           # Required: Must match filename and be globally unique
display_name: "Your Model Name"   # Required: Shown in X-AnyLabeling UI

params:                           # Optional: All accessible via self.params in model
  model_path: "path/to/weights.pt"
  device: "cuda:0"
  conf_threshold: 0.25
  # Add any custom parameters

widgets:                          # Optional: UI components (see table below)
  - name: button_run
    value: null
  - name: edit_conf
    value: 0.25                   # Must provide value for required widgets
  - name: edit_iou
    value: 0.45
  - name: toggle_preserve_existing_annotations
    value: false
```

**Widget Configuration:**

| Widget Name | Type | Range | Required | Description |
|------------|------|-------|----------------|-------------|
| `button_run` | null | - | - | Trigger inference button for auto-labeling |
| `button_send` | null | - | - | Submit button (required if using `edit_text`) |
| `input_conf` | null | - | - | Label for confidence threshold input |
| `edit_conf` | float | 0.0-1.0 | ✅ Must provide | Confidence threshold slider/input |
| `input_iou` | null | - | - | Label for IoU threshold input |
| `edit_iou` | float | 0.0-1.0 | ✅ Must provide | IoU threshold slider/input for NMS |
| `edit_text` | string | - | - | Text input field for prompts (requires `button_send`) |
| `toggle_preserve_existing_annotations` | bool | - | ✅ Must provide | Checkbox to keep existing annotations |
| `mask_fineness_slider` | int | 1-100 | ✅ Must provide | Mask detail level (for segmentation models) |

> [!TIP]
> Multiple models can share the same implementation class by using different `model_id` and `params`.

## 3. Troubleshooting

### 3.1 Server Won't Start

**Check:**
- YAML syntax errors (use a YAML validator)
- Port already in use (`lsof -i :8000`)
- Invalid configuration values
- Missing required fields in model configs

### 3.2 High Memory Usage

**Solutions:**
- Reduce `max_workers`
- Reduce `max_queue_size`
- Disable unused models
- Use smaller model variants

### 3.3 Slow Response Times

**Check:**
- `max_workers` too low (increase if GPU allows)
- Queue is full (check `max_queue_size`)
- Model loading time (optimize `load()` method)
- Network latency for API-based models

### 3.4 Queue Full Errors

**Solutions:**
- Increase `max_queue_size`
- Increase `max_workers` (if resources allow)
- Implement client-side retry logic
- Add more server instances (load balancing)
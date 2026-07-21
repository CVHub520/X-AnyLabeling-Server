# API 使用指南

X-AnyLabeling-Server 基于 FastAPI 提供 HTTP 接口，覆盖健康检查、模型发现、图像推理和交互式视频会话。

[`openapi.json`](../../openapi.json) 是 API 的权威契约。该文件直接从 FastAPI 应用导出；路由或数据模型发生变化时，CI 会检查它是否同步更新。

## 交互式文档

启动服务后，可打开 FastAPI 自带的接口页面：

- Swagger UI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`
- OpenAPI JSON：`http://127.0.0.1:8000/openapi.json`

官网中的 [API 参考](https://xanylabeling.com/zh-Hans/docs/server/api-reference) 同样由这份 Schema 自动生成。

## 基础地址

本文示例统一使用：

```text
http://127.0.0.1:8000
```

如果修改了 `configs/server.yaml` 或使用自定义配置文件，请相应调整主机和端口。

## 身份验证

API Key 验证默认关闭。启用时修改 `configs/server.yaml`：

```yaml
security:
  api_key_enabled: true
  api_key: ""
  api_key_header: "Token"
```

建议通过环境变量设置密钥，避免提交到仓库：

```bash
export XANYLABELING_API_KEY="your-secret-key"
```

随后在请求中加入配置的 Header：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Token: your-secret-key"
```

`/health` 始终无需身份验证。

## 常用请求

### 检查服务状态

```bash
curl http://127.0.0.1:8000/health
```

### 查看已加载模型

```bash
curl http://127.0.0.1:8000/v1/models
```

### 查看模型信息

```bash
curl http://127.0.0.1:8000/v1/models/yolo11n/info
```

### 执行图像推理

`image` 字段支持 Base64 字符串或 Data URI：

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11n",
    "image": "data:image/jpeg;base64,...",
    "params": {"conf_threshold": 0.3}
  }'
```

成功响应包含标注形状和可选的描述文本：

```json
{
  "success": true,
  "data": {
    "shapes": [],
    "description": "",
    "replace": false
  }
}
```

形状字段定义参见[用户指南](./user_guide.md#2-响应数据结构)。

## 视频会话

交互式视频模型采用基于会话的调用流程：

1. 使用 `POST /v1/video/init` 初始化会话。
2. 使用 `POST /v1/video/prompt` 添加文本或点提示。
3. 使用 `POST /v1/video/propagate` 启动传播，或通过 `POST /v1/video/propagate/stream` 流式接收结果。
4. 使用 `GET /v1/video/status/{task_id}` 获取进度。
5. 任务结束后取消任务或清理会话。

当前请求与响应 Schema 请以自动生成的 API 参考为准。

## 错误处理

应用层错误通常使用以下结构：

```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model is not loaded"
  }
}
```

请求队列达到上限时返回 HTTP `503`。客户端应采用有限次数的退避重试，避免立即重复发送请求。

## 更新 API 契约

修改 FastAPI 路由或 Pydantic Schema 后执行：

```bash
python scripts/export_openapi.py
python scripts/export_openapi.py --check
```

路由变更必须同时提交更新后的 `docs/openapi.json`；CI 会拒绝过期 Schema。

# 快速入门

本指南介绍如何安装 X-AnyLabeling-Server、启动远程推理服务，并将其连接到 X-AnyLabeling 桌面端。

## 1. 环境要求

- Python 3.10 或更高版本；部分模型需要 Python 3.12。
- 根据模型选择 CPU 或 CUDA 环境。
- PyTorch 的版本必须与本机 CUDA 和显卡驱动兼容。

建议先按照 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 安装匹配的 PyTorch，再安装服务端依赖。

### 使用 Conda

```bash
conda create --name x-anylabeling-server python=3.12 -y
conda activate x-anylabeling-server
```

### 使用 venv

```bash
python3 -m venv venv-server
source venv-server/bin/activate  # Linux/macOS
# venv-server\Scripts\activate  # Windows
```

## 2. 安装服务

克隆仓库：

```bash
git clone https://github.com/CVHub520/X-AnyLabeling-Server.git
cd X-AnyLabeling-Server
```

推荐使用 `uv` 安装依赖：

```bash
python -m pip install --upgrade uv
```

按使用场景选择安装方式：

```bash
# 只安装核心框架，适合部署自定义模型
uv pip install -e .

# 安装仓库提供的常用示例依赖
uv pip install -e .[all]
```

> [!TIP]
> 初次体验建议使用 `.[all]`；生产环境应只安装实际使用的模型依赖，以减少环境冲突和镜像体积。

部分模型需要额外环境：

| 模型或能力 | 建议 |
| --- | --- |
| SAM 2 / SAM 3 | 使用 Python 3.12、PyTorch 2.7+ 和 CUDA 12.6+；安装 `.[sam3]` |
| LocateAnything | 建议单独环境并安装 `.[locateanything]`，其 Transformers 版本要求为 `>=4.50,<5` |
| Rex-Omni / PaddleOCR-VL-1.5 | 建议使用 `.[all]` |
| PaddleOCR-VL-1.5 | 需要 `transformers>=5.0.0`；根据显存调整生成长度与图像尺寸参数 |
| PP-DocLayoutV3 | 需要与模型兼容的最新版 Transformers |

模型权重、后端和设备参数在 `configs/auto_labeling/*.yaml` 中配置。不要将密钥或机器相关的绝对路径提交到公共仓库。

## 3. 启动服务

使用默认配置启动：

```bash
x-anylabeling-server
```

也可以显式指定配置文件：

```bash
x-anylabeling-server \
  --config /path/to/custom/server.yaml \
  --models-config /path/to/custom/models.yaml
```

开发调试时也可直接使用 Uvicorn：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

配置项详见[配置指南](./configuration.md)。

## 4. 验证接口

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 查看已加载模型
curl http://127.0.0.1:8000/v1/models
```

启动后还可访问：

- Swagger UI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`
- OpenAPI JSON：`http://127.0.0.1:8000/openapi.json`

## 5. 连接桌面端

在 X-AnyLabeling 用户配置文件（默认 `~/.xanylabelingrc`）中加入：

```yaml
remote_server_settings:
  server_url: http://127.0.0.1:8000
  api_key: ""
```

如果服务端启用了 API Key，请在 `api_key` 中填写对应值。随后启动 X-AnyLabeling，按 `Ctrl+A` 打开 AI 自动标注面板，在 CVHub 提供商分组中选择 `Remote-Server`。

桌面端会从服务端读取已启用模型及其控件配置，并将预测结果转换为可编辑标注。

## 6. 后续阅读

- [配置指南](./configuration.md)：服务、安全、并发、日志和模型配置。
- [用户指南](./user_guide.md)：自定义模型接入、响应格式和故障排查。
- [API 使用指南](./router.md)：认证、常用请求和视频会话流程。

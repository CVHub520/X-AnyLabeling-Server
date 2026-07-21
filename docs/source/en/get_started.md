# Quick Start Guide

## 0. Table of Contents

* [1. Installation](#1-installation)
   * [1.1 Prerequisites](#11-prerequisites)
      * [1.1.1 Miniconda](#111-miniconda)
      * [1.1.2 Venv](#112-venv)
   * [1.2 Install Server](#12-install-server)
      * [1.2.1 Git Clone](#121-git-clone)
* [2. Further Reading](#2-further-reading)

## 1. Installation

### 1.1 Prerequisites

#### 1.1.1 Miniconda

**Step 0.** Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 1.** Create a conda environment with Python 3.10+ and activate it.

> [!NOTE]
> Other Python versions require compatibility verification on your own.

```bash
conda create --name x-anylabeling-server python=3.12 -y
conda activate x-anylabeling-server
```

#### 1.1.2 Venv

In addition to Miniconda, you can also use Python's built-in `venv` module to create virtual environments. Here are the commands for creating and activating environments under different configurations:

```bash
python3 -m venv venv-server
source venv-server/bin/activate  # Linux/macOS
# venv-server\Scripts\activate   # Windows
```

### 1.2 Install Server

#### 1.2.1 Git Clone

**Step a.** Clone the repository.

```bash
git clone https://github.com/CVHub520/X-AnyLabeling-Server.git
cd X-AnyLabeling-Server
```

**Step b.** Install dependencies.

For faster dependency installation and a more modern Python package management experience, we strongly recommend using [uv](https://github.com/astral-sh/uv) as your package manager. uv provides significantly faster installation speeds and better dependency resolution capabilities.

> [!NOTE]
> PyTorch requirements vary by operating system and CUDA requirements, so we recommend installing PyTorch first by following the [instructions](https://pytorch.org/get-started/locally/) at PyTorch.

> [!TIP]
> For users in China, you can use a pip mirror, e.g., `-i https://pypi.tuna.tsinghua.edu.cn/simple`, to accelerate downloads.

```bash
pip install --upgrade uv

# Option 1: Install core framework only (for deploying custom models)
uv pip install -e .

# Option 2: Install with example dependencies (recommended for first-time users)
# This includes ultralytics and transformers for running demo models
uv pip install -e .[all]
```

> [!TIP]
> If you're new to X-AnyLabeling-Server, we recommend using **Option 2** to install all dependencies so you can run the demo models out of the box. If you only plan to deploy your own custom models without these examples, **Option 1** is sufficient.

> [!NOTE]
> Model requirements differ:
>
> - `sam3`: Use Python 3.12+, PyTorch 2.7+, and a CUDA-compatible GPU with CUDA 12.6+. The same dependency group can be used to build `sam2`.
> - GLM and other API models: Supply credentials through environment variables such as `ZHIPU_API_KEY`, or through a private model configuration. They can also be deployed through vLLM or SGLang.
> - `locateanything`: Install the `[locateanything]` extra in a separate environment. It requires `transformers>=4.50,<5`.
> - `rexomni` and `paddleocr_vl_1_5`: Prefer the `[all]` installation mode.
> - `paddleocr_vl_1_5`: Install `transformers>=5.0.0`. Tune generation and image-size parameters in `configs/auto_labeling/paddleocr_vl_1_5.yaml` to match available GPU memory.
> - `pp_doclayout_v3`: Install a Transformers version compatible with the current upstream model release.
> - `geco2`: Download `CNTQG_multitrain_ca44.pth` and `sam2_hiera_base_plus.pt` from the [v0.0.9 assets](https://github.com/CVHub520/X-AnyLabeling-Server/releases/download/v0.0.9).

After installation, you can quickly start the service with the following command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> [!TIP]
> **Recommended:** Use the `x-anylabeling-server` command instead, which supports custom configuration files:
> 
> ```bash
> # Start with default configs
> x-anylabeling-server
> 
> # Start with custom config files
> x-anylabeling-server \
>   --config /path/to/custom/server.yaml \
>   --models-config /path/to/custom/models.yaml
> ```
> 
> For more configuration options, see the [Configuration Guide](./configuration.md).

---

Test the API to verify that your model is loaded correctly:

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Run inference
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "your_model_id", "image": "data:image/png;base64,...", "params": {}}'
```

Once the server is running, open the [X-AnyLabeling client](https://github.com/CVHub520/X-AnyLabeling) and follow these steps:

1. Configure the server connection in your user configuration file (default: `~/.xanylabelingrc`):
   ```yaml
   remote_server_settings:
     server_url: http://127.0.0.1:8000  # Update if using a different address or port
     api_key: ""  # Required if authentication is enabled
   ```
2. Launch X-AnyLabeling and press `Ctrl+A` to enable AI auto-labeling.
3. Open the model dropdown, navigate to the **CVHub** provider section, and select **Remote-Server**.

The desktop client can now discover the enabled remote models and convert their predictions into editable annotations.

> [!TIP]
> For detailed instructions on using the client, see the [X-AnyLabeling User Guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md).

## 2. Further Reading

- [User Guide](./user_guide.md): Step-by-step guide for integrating custom models, response schema specifications, and troubleshooting.
- [API Reference](./router.md): Complete REST API endpoint documentation.
- [Configuration Guide](./configuration.md): Detailed server, logging, and performance configuration.

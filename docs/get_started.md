# Quick Start Guide

## 0. Table of Contents

* [1. Installation](#1-installation)
   * [1.1 Prerequisites](#11-prerequisites)
      * [1.1.1 Miniconda](#111-miniconda)
      * [1.1.2 Venv](#112-venv)
   * [1.2 Install Server](#12-install-server)
      * [1.2.1 Git Clone](#121-git-clone)
* [2. Usage](#2-usage)

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

After installation, you can quickly start the service with the following command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Once the server is running, open the [X-AnyLabeling client](https://github.com/CVHub520/X-AnyLabeling) and follow these steps:

1. Configure server connection in [`remote_server.yaml`](https://github.com/CVHub520/X-AnyLabeling/blob/main/anylabeling/configs/auto_labeling/remote_server.yaml):
   - Set `server_url` if you changed the default address/port
   - Set `api_key` if you enabled authentication
2. Launch X-AnyLabeling and press `Ctrl+A` to enable AI auto-labeling
3. Open the model dropdown, navigate to the **CVHub** provider section, and select **Remote-Server**

You can now use your remotely deployed models for auto-labeling!

> [!TIP]
> For detailed usage instructions, refer to the [X-AnyLabeling User Guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md).

## 2. Usage

For detailed instructions on how to use X-AnyLabeling-Server, please refer to the corresponding [User Guide](./user_guide.md).
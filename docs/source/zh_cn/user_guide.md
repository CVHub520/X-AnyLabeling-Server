# 用户指南

本指南介绍如何接入自定义模型、向桌面端暴露模型能力，以及返回符合 X-AnyLabeling 约定的标注结果。

## 1. 接入新模型

接入过程包含四个步骤：

1. 在 `app/models/` 中实现模型类。
2. 使用装饰器注册模型 ID。
3. 在 `configs/auto_labeling/` 中创建模型配置。
4. 在 `configs/models.yaml` 中启用模型。

### 1.1 实现模型类

模型应继承 `BaseModel`，并实现加载、推理和释放逻辑：

```python
from app.models import BaseModel
from app.schemas.shape import Shape


class YourModel(BaseModel):
    def load(self):
        model_path = self.params.get("model_path")
        self.model = load_your_model(model_path)

    def predict(self, image, params):
        results = self.model(image)
        shapes = [
            Shape(
                label="object",
                shape_type="rectangle",
                points=[[10, 10], [100, 10], [100, 100], [10, 100]],
            )
        ]
        return {"shapes": shapes, "description": ""}

    def unload(self):
        del self.model
```

注意事项：

- 输入图像采用 OpenCV 的 BGR 格式。
- 运行时参数通过 `params` 传入，可覆盖部分默认配置。
- 返回值至少应包含 `shapes` 和 `description`。
- 模型资源必须在 `unload()` 中正确释放。

完整实现可参考 [`app/models/`](../../../app/models/)。

### 1.2 注册模型

使用 `@register_model` 注册一个或多个模型 ID：

```python
from app.core.registry import register_model


@register_model("your_model_id")
class YourModel(BaseModel):
    ...
```

多个配置可以复用同一实现：

```python
@register_model("model_small", "model_large")
class SharedModel(BaseModel):
    ...
```

服务会自动导入 `app/models/` 第一层目录中的模块。模型 ID 必须全局唯一，并与配置文件中的 `model_id` 一致。

### 1.3 创建模型配置

创建 `configs/auto_labeling/your_model_id.yaml`：

```yaml
model_id: your_model_id
display_name: "Your Model Name"
batch_processing_mode: "default"

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

`widgets` 决定桌面端远程模型面板展示哪些交互控件；完整列表参见[配置指南](./configuration.md#32-单模型配置)。

### 1.4 声明专项能力

`capabilities` 用于将模型路由到专用客户端模块，而不是普通的 Remote-Server 下拉列表：

```yaml
capabilities:
  ppocr_pipeline: true
```

能力字段应保持稳定且可机器读取。已经发布的字段不要随意改名，否则会影响客户端兼容性。

多任务模型可以通过元数据返回 `available_tasks`，并在配置中启用任务选择控件：

```yaml
widgets:
  - name: remote_task_select_combobox
    value: null
```

每个任务可独立声明批处理模式、活动控件和控件参数。

### 1.5 启用模型

将模型 ID 加入 `configs/models.yaml`：

```yaml
enabled_models:
  - yolo11n
  - your_model_id
```

模型会按照列表顺序加载并展示。注释掉暂时不用的模型可以减少启动时间和资源占用。

## 2. 响应数据结构

模型的 `predict()` 应返回：

```python
{
    "shapes": [],
    "description": "",
    "replace": None,
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `shapes` | List | 标注形状列表；文本描述任务可以为空 |
| `description` | String | 图像或结果描述；检测任务可以为空 |
| `replace` | Boolean/null | 是否替换已有标注；可选 |

每个 `Shape` 支持以下字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `label` | String | 是 | 标签名称 |
| `shape_type` | String | 是 | 形状类型 |
| `points` | List | 是 | 由 `[x, y]` 坐标组成的点集 |
| `score` | Float | 否 | `0.0` 到 `1.0` 的置信度 |
| `attributes` | Dict | 否 | 自定义属性 |
| `description` | String | 否 | 形状级描述 |
| `difficult` | Boolean | 否 | 是否为困难样本 |
| `direction` | Float | 否 | 方向弧度，范围为 `0` 到 `2π` |
| `flags` | Dict | 否 | 额外标记 |
| `group_id` | Integer | 否 | 关联形状的正整数分组 ID |
| `kie_linking` | List | 否 | 关键信息抽取关系 |

支持的 `shape_type`：

- `rectangle`
- `polygon`
- `quadrilateral`
- `rotation`
- `point`
- `line`
- `circle`
- `linestrip`

服务端可以返回额外字段，但应确保桌面端能够忽略未知字段，且不要破坏上述稳定字段的语义。

## 3. 视频模型

视频模型使用会话式接口，并在配置中声明：

```yaml
batch_processing_mode: video
```

模型实现可按需提供：

- `init_session()`：载入帧序列并初始化状态。
- `add_prompt()` 或 `add_point_prompt()`：添加文本或点提示。
- `start_propagation()`：启动异步传播任务。
- `propagate_stream()`：通过 SSE 流式返回结果。
- `get_task_status()`：查询传播状态。

接口调用顺序参见 [API 使用指南](./router.md#视频会话)。

## 4. 故障排查

| 问题 | 处理方式 |
| --- | --- |
| `Model 'xxx' not registered` | 确认模块位于 `app/models/` 第一层，并使用正确的 `@register_model` |
| 找不到配置文件 | 检查文件名、`model_id` 和 `configs/models.yaml` 是否一致 |
| 控件缺少默认值 | 为 `edit_conf`、`edit_iou` 等必需控件设置 `value` |
| 模型 ID 重复 | 保证所有配置的 `model_id` 全局唯一 |
| CUDA 内存不足 | 减小输入尺寸或批量、减少并发，或换用更小模型 |
| 桌面端看不到模型 | 检查服务日志、`enabled_models`、能力字段以及客户端与服务端版本 |

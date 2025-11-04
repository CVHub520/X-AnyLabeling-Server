# User Manual

## 0. Table of Contents

* [1. How to Support New Models](#1-how-to-support-new-models)
  * [1.1 Create Model Class](#11-create-model-class)
  * [1.2 Register Model](#12-register-model)
  * [1.3 Create Configuration](#13-create-configuration)
  * [1.4 Enable Model](#14-enable-model)
* [2. Response Schema](#2-response-schema)
* [3. Troubleshooting](#3-troubleshooting)
* [4. Further Reading](#4-further-reading)

## 1. How to Support New Models

Integrating a custom model requires 4 simple steps:

1. **Create** a model class in `app/models/your_model.py`
2. **Register** it in `app/core/registry.py`
3. **Configure** it in `configs/auto_labeling/your_model_id.yaml`
4. **Enable** it in `configs/models.yaml`

### 1.1 Create Model Class

Your model must inherit from `BaseModel` and implement three methods:

```python
from .base import BaseModel
from app.schemas.shape import Shape

class YourModel(BaseModel):
    def load(self):
        """Load model weights and initialize resources"""
        model_path = self.params.get("model_path")
        self.model = load_your_model(model_path)
    
    def predict(self, image, params):
        """Run inference and return results"""
        results = self.model(image)
        shapes = [Shape(label="...", shape_type="rectangle", points=[...])]
        return {"shapes": shapes, "description": ""}
    
    def unload(self):
        """Free resources on shutdown"""
        del self.model
```

**Key Points:**
- Input `image` is in BGR format (OpenCV style)
- Return a dict with `shapes` and `description` fields (see [Response Schema](#13-response-schema))
- Check [app/models/](../app/models/) for complete implementation examples

### 1.2 Register Model

Add your model to `app/core/registry.py`:

```python
def _build_registry(self):
    from app.models.your_model import YourModel
    return {
        "your_model_id": YourModel,
        # ... other models
    }
```

> [!TIP]
> Multiple model IDs can share the same class (e.g., `yolo11n` and `yolo11s` both use `YOLO11nDetection`)

### 1.3 Create Configuration

Create `configs/auto_labeling/your_model_id.yaml`:

```yaml
model_id: your_model_id          # Required: Must be globally unique
display_name: "Your Model Name"  # Required: Displayed in X-AnyLabeling UI

params:                          # Optional: All params are passed to your model's __init__
  model_path: "path/to/weights.pt"
  device: "cuda:0"
  conf_threshold: 0.25
  # Add any custom parameters your model needs

widgets:
  - name: button_run
    value: null
  - name: edit_conf
    value: 0.25          # Widgets with ✅ must have default values
  - name: edit_iou
    value: 0.45
  - name: toggle_preserve_existing_annotations
    value: false
```

See [Widget Reference](./configuration.md#model-configuration) for details.

### 1.4 Enable Model

Add to `configs/models.yaml`:

```yaml
enabled_models:
  - yolo11n
  - yolo11n_seg
  - your_model_id  # Models will be displayed in this order
  # - yolo11s      # Comment out to disable a model
```

> [!NOTE]
> - Models are displayed in X-AnyLabeling UI in the order listed here
> - Comment out any model you don't want to load to save resources

Then start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You can also test the API to verify your deployed model is loaded correctly:

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

## 2. Response Schema

Your model's `predict()` method must return a dictionary with two fields:

```python
{
    "shapes": [...],      # List of Shape objects (can be empty for caption tasks)
    "description": "..."  # Text description (can be empty for detection tasks)
}
```

Each `Shape` object should have the following properties:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `label` | String | ✅ Yes | Category label of the object |
| `shape_type` | String | ✅ Yes | Type of shape (see supported types below) |
| `points` | List | ✅ Yes | List of `[x, y]` coordinates defining the shape vertices |
| `score` | Float | No | Confidence score from model inference (default: `None`) |
| `attributes` | Dict | No | Custom object attributes (default: `{}`) |
| `description` | String | No | Optional text description for the shape |
| `difficult` | Boolean | No | Flag if object is difficult to identify (default: `False`) |
| `direction` | Float | No | Direction in radians, 0-2π (default: `0`) |
| `flags` | Dict | No | Additional flags or metadata |
| `group_id` | Integer | No | ID to group related shapes (e.g., pose keypoints) |
| `kie_linking` | List | No | Key Information Extraction linking data (default: `[]`) |

**Supported Shape Types:**

- **`rectangle`**: Horizontal bounding box defined by 4 corner points
- **`polygon`**: Closed polygon with 3+ vertices
- **`rotation`**: Oriented/rotated bounding box
- **`point`**: Single point coordinate
- **`line`**: Line segment with start and end points
- **`circle`**: Circle defined by center and radius point
- **`linestrip`**: Connected line segments (polyline)

For detailed shape specifications, see the [X-AnyLabeling User Guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md) and [`app/schemas/shape.py`](../app/schemas/shape.py).

## 3. Troubleshooting

| Error | Solution |
|-------|----------|
| `Model 'xxx' not registered` | Add to `_build_registry()` in `registry.py` |
| `Configuration file not found` | Check YAML file exists and `model_id` matches filename |
| `Widget 'edit_conf' requires a default value` | Set `value: 0.25` in widgets config |
| `Duplicate model_id found` | Each `model_id` must be unique |

## 4. Further Reading

- [API Reference](./router.md) - Complete REST API endpoint documentation
- [Configuration Guide](./configuration.md) - Detailed server, logging, and performance configuration
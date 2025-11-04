# API Reference

This document provides a complete reference for all REST API endpoints in X-AnyLabeling-Server.

## 0. Table of Contents

* [1. Endpoints](#1-endpoints)
  * [1.1 Health Check](#11-health-check)
  * [1.2 List All Models](#12-list-all-models)
  * [1.3 Get Model Info](#13-get-model-info)
  * [1.4 Run Inference](#14-run-inference)
* [2. Error Codes](#2-error-codes)
* [3. Authentication](#3-authentication)
  * [3.1 Enable Authentication](#31-enable-authentication)
  * [3.2 Using API Key](#32-using-api-key)
* [4. Client Examples](#4-client-examples)
  * [4.1 Python](#41-python)
  * [4.2 JavaScript](#42-javascript)

## 1. Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/health` | Health check | No |
| GET | `/v1/models` | List all models | Optional |
| GET | `/v1/models/{model_id}/info` | Get model details | Optional |
| POST | `/v1/predict` | Run inference | Optional |

### 1.1 Health Check

Check server health and status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "timestamp": "2025-11-04T10:00:00.000Z"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | String | Server status (always `"healthy"` if responding) |
| `models_loaded` | Integer | Number of models currently loaded |
| `timestamp` | String | Current server time in ISO 8601 format (UTC) |

**Example:**
```bash
curl http://localhost:8000/health
```

### 1.2 List All Models

Get metadata for all loaded models.

**Endpoint:** `GET /v1/models`

**Response:**
```json
{
  "success": true,
  "data": {
    "yolo11n": {
      "display_name": "YOLO11n Detection",
      "widgets": [
        {"name": "button_run", "value": null},
        {"name": "edit_conf", "value": 0.25},
        {"name": "edit_iou", "value": 0.70}
      ],
      "params": {
        "model_path": "yolo11n.pt",
        "device": "cpu"
      }
    },
    "yolo11n_seg": {
      "display_name": "YOLO11n Segmentation",
      "widgets": [...],
      "params": {...}
    }
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `data` | Object | Dictionary mapping model IDs to their metadata |
| `data[model_id].display_name` | String | Human-readable model name |
| `data[model_id].widgets` | Array | UI widget configurations |
| `data[model_id].params` | Object | Model parameters |

**Example:**
```bash
curl http://localhost:8000/v1/models
```

### 1.3 Get Model Info

Get detailed information about a specific model.

**Endpoint:** `GET /v1/models/{model_id}/info`

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | String | Yes | Unique model identifier |

**Success Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "yolo11n",
    "display_name": "YOLO11n Detection",
    "widgets": [
      {"name": "button_run", "value": null},
      {"name": "edit_conf", "value": 0.25},
      {"name": "edit_iou", "value": 0.70}
    ],
    "params": {
      "model_path": "yolo11n.pt",
      "device": "cpu"
    }
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'invalid_model' not loaded"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/v1/models/yolo11n/info
```

### 1.4 Run Inference

Execute model inference on an image.

**Endpoint:** `POST /v1/predict`

**Request Body:**
```json
{
  "model": "yolo11n",
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "params": {
    "conf_threshold": 0.3,
    "iou_threshold": 0.5
  }
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | String | Yes | Model ID to use for inference |
| `image` | String | Yes | Base64-encoded image (Data URI format or raw base64) |
| `params` | Object | No | Inference parameters (overrides config defaults) |

**Image Format:**
- Supports Data URI format: `data:image/png;base64,{base64_string}`
- Or raw base64 string without prefix
- Supported formats: PNG, JPEG, BMP, WEBP, TIFF

**Success Response (Detection):**
```json
{
  "success": true,
  "data": {
    "shapes": [
      {
        "label": "person",
        "shape_type": "rectangle",
        "points": [[100.5, 200.3], [150.2, 200.3], [150.2, 300.8], [100.5, 300.8]],
        "score": 0.95,
        "attributes": {},
        "description": null,
        "difficult": false,
        "direction": 0,
        "flags": null,
        "group_id": null,
        "kie_linking": []
      }
    ],
    "description": ""
  }
}
```

**Success Response (Caption):**
```json
{
  "success": true,
  "data": {
    "shapes": [],
    "description": "A person standing in front of a blue car on a sunny day."
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `data.shapes` | Array | List of detected objects (see [Shape Schema](./user_guide.md#13-response-schema)) |
| `data.description` | String | Generated text description (for caption tasks) |

**Model Not Found:**
```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'invalid_model' not loaded"
  }
}
```

**Invalid Image:**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Failed to decode image"
  }
}
```

**Inference Error:**
```json
{
  "success": false,
  "error": {
    "code": "INFERENCE_ERROR",
    "message": "CUDA out of memory"
  }
}
```

**Queue Full (503):**
```json
{
  "detail": "Task queue is full, please try again later"
}
```

**Examples:**

```bash
# Using curl
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11n",
    "image": "data:image/png;base64,iVBORw0KGgo...",
    "params": {"conf_threshold": 0.3}
  }'

# Using Python
import requests
import base64

with open("image.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/v1/predict",
    json={
        "model": "yolo11n",
        "image": f"data:image/jpeg;base64,{img_base64}",
        "params": {"conf_threshold": 0.3}
    }
)
print(response.json())
```

## 2. Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `MODEL_NOT_FOUND` | 200 | Requested model is not loaded or doesn't exist |
| `INVALID_IMAGE` | 200 | Image decoding failed (invalid format or corrupted) |
| `INFERENCE_ERROR` | 200 | Error during model inference |
| `UNAUTHORIZED` | 200 | Invalid or missing API key (when auth enabled) |
| `QUEUE_FULL` | 503 | Task queue is full, server overloaded |

> **Note:** Most errors return HTTP 200 with `success: false` in the response body. Only queue-related errors return 503 status.

## 3. Authentication

API key authentication is optional and disabled by default.

### 3.1 Enable Authentication

Edit `configs/server.yaml`:

```yaml
security:
  api_key_enabled: true
  api_key: "your-secret-key"
  api_key_header: "Token"
```

### 3.2 Using API Key

Include the API key in request headers:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "Token: your-secret-key" \
  -d '{...}'
```

**Authentication Failure:**
```json
{
  "success": false,
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API Key"
  }
}
```

**Exempt Endpoints:**
- `/health` does not require authentication

## 4. Client Examples

### 4.1 Python

```python
import requests
import base64
from pathlib import Path

class XAnyLabelingClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Token"] = api_key

    def health(self):
        """Check server health"""
        return requests.get(f"{self.base_url}/health").json()

    def list_models(self):
        """List all available models"""
        return requests.get(
            f"{self.base_url}/v1/models",
            headers=self.headers
        ).json()

    def predict(self, model_id, image_path, params=None):
        """Run inference on an image"""
        # Read and encode image
        img_bytes = Path(image_path).read_bytes()
        img_base64 = base64.b64encode(img_bytes).decode()

        # Prepare request
        payload = {
            "model": model_id,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "params": params or {}
        }

        # Send request
        response = requests.post(
            f"{self.base_url}/v1/predict",
            json=payload,
            headers=self.headers
        )
        return response.json()

# Usage
client = XAnyLabelingClient()
print(client.health())
result = client.predict("yolo11n", "image.jpg", {"conf_threshold": 0.3})
print(result)
```

### 4.2 JavaScript

```javascript
class XAnyLabelingClient {
  constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
    this.baseUrl = baseUrl;
    this.headers = { 'Content-Type': 'application/json' };
    if (apiKey) {
      this.headers['Token'] = apiKey;
    }
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async listModels() {
    const response = await fetch(`${this.baseUrl}/v1/models`, {
      headers: this.headers
    });
    return response.json();
  }

  async predict(modelId, imageFile, params = {}) {
    // Read file as base64
    const base64 = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.readAsDataURL(imageFile);
    });

    const response = await fetch(`${this.baseUrl}/v1/predict`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: modelId,
        image: base64,
        params: params
      })
    });
    return response.json();
  }
}

// Usage
const client = new XAnyLabelingClient();
const result = await client.predict('yolo11n', imageFile, {conf_threshold: 0.3});
console.log(result);
```
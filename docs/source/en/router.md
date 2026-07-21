# API Guide

X-AnyLabeling-Server exposes a FastAPI HTTP interface for health checks, model discovery, image inference, and interactive video sessions.

The canonical API contract is [`openapi.json`](../../openapi.json). It is exported directly from the FastAPI application and checked in CI whenever routes or schemas change.

## Interactive documentation

After starting the server, open one of the built-in interfaces:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

The website's [API Reference](https://xanylabeling.com/docs/server/api-reference) is generated from the same schema.

## Base URL

Examples in this guide use:

```text
http://127.0.0.1:8000
```

Change the host or port to match `configs/server.yaml` or your custom server configuration.

## Authentication

API key authentication is disabled by default. To enable it, update `configs/server.yaml`:

```yaml
security:
  api_key_enabled: true
  api_key: ""
  api_key_header: "Token"
```

Set the secret through the environment instead of committing it:

```bash
export XANYLABELING_API_KEY="your-secret-key"
```

Then include the configured header in requests:

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Token: your-secret-key"
```

The `/health` endpoint remains available without authentication.

## Common requests

### Check server health

```bash
curl http://127.0.0.1:8000/health
```

### List loaded models

```bash
curl http://127.0.0.1:8000/v1/models
```

### Inspect a model

```bash
curl http://127.0.0.1:8000/v1/models/yolo11n/info
```

### Run image inference

The `image` field accepts a base64 string or a Data URI:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11n",
    "image": "data:image/jpeg;base64,...",
    "params": {"conf_threshold": 0.3}
  }'
```

Successful requests return annotation shapes and an optional description:

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

See the [User Guide](./user_guide.md#2-response-schema) for the shape response contract.

## Video sessions

Interactive video models use a session-based flow:

1. Initialize a session with `POST /v1/video/init`.
2. Add a text or point prompt with `POST /v1/video/prompt`.
3. Start propagation with `POST /v1/video/propagate` or stream results from `POST /v1/video/propagate/stream`.
4. Read progress from `GET /v1/video/status/{task_id}`.
5. Cancel a task or clean up a session when it is no longer needed.

Use the generated API Reference for the current request and response schemas.

## Error handling

Application errors generally use this envelope:

```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model is not loaded"
  }
}
```

Queue saturation returns HTTP `503`. Clients should use bounded retries with backoff rather than immediately repeating the same request.

## Updating the API contract

After changing FastAPI routes or Pydantic schemas, run:

```bash
python scripts/export_openapi.py
python scripts/export_openapi.py --check
```

Commit the updated `docs/openapi.json` with the route change. CI rejects stale schemas.

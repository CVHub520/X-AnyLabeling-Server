# Unlimited-OCR-2 document serving

Install the backend dependency group listed in [Getting Started](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/docs/source/en/get_started.md). The Transformers backend runs in X-AnyLabeling-Server; the SGLang backend runs as a separate service and can be hosted on another machine. Use the customized SGLang wheel supplied by the `unlimited_ocr_2_sglang` extra: a stock SGLang release does not provide this model's document-window implementation.

## SGLang service

Start the customized service in its serving environment:

```bash
export SGLANG_UOCR_EANABLE_DOC_LIMIT=1
export SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW=2
export SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=2
export SGLANG_UOCR_MAX_PAGE_TOKENS=20000
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64
export SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1
export SGLANG_VLM_CACHE_SIZE_MB=0

python -m sglang.launch_server \
  --model-path baidu/Unlimited-OCR-2 \
  --revision c14071b549f4fcab10909f3c4acc6a2edb366947 \
  --served-model-name Unlimited-OCR-2 \
  --chat-template unlimited-ocr \
  --tokenizer-backend fastokens \
  --mem-fraction-static 0.72 \
  --max-running-requests 1 \
  --max-prefill-tokens 1024 \
  --enable-custom-logit-processor \
  --enable-streaming-session \
  --allow-auto-truncate \
  --disable-radix-cache \
  --mm-preprocess-cache-size-mb 0 \
  --disable-cuda-graph \
  --host 127.0.0.1 --port 10000
```

`SGLANG_UOCR_EANABLE_DOC_LIMIT` is spelled this way in upstream code. With document limits enabled, SGLang processes page windows and retains a context tail between them. Larger page windows retain more cross-page context and require more memory. Set it to `0` to parse the document in one window. The model selects its required `unlimited_ocr_v2` attention backend automatically; do not force a generic attention backend.

Configure `configs/auto_labeling/unlimited_ocr_2.yaml`:

```yaml
params:
  backend: sglang
  api_url: http://127.0.0.1:10000/v1
  served_model: Unlimited-OCR-2
  max_new_tokens: 8192
  crop_mode: null
  request_timeout: 3600
```

Keep the other configuration fields, including `capabilities`. Enable `unlimited_ocr_2` in `configs/models.yaml` and start X-AnyLabeling-Server normally. The adapter verifies that the named SGLang model is served before registering it as loaded. Configure the client with the **X-AnyLabeling-Server** address, then select Unlimited-OCR-2 in the PaddleOCR panel.

For authenticated SGLang deployments, configure authentication on the serving process and set `SGLANG_API_KEY` in the X-AnyLabeling-Server environment. The model's `api_key` setting is an alternative.

## Document requests

`POST /v1/predict/document` accepts ordered base64-encoded page images:

```json
{
  "model": "unlimited_ocr_2",
  "images": ["<page-1-base64>", "<page-2-base64>"],
  "params": {"max_new_tokens": 8192}
}
```

The PaddleOCR panel handles PDF rendering and submits its pages together. The endpoint itself accepts page images, not PDF bytes. The response contains `layoutParsingResults` in page order, page dimensions, global block/group identifiers, and document Markdown assembled by the upstream postprocessor. Cross-page text/table continuations share a group ID; table fragments are merged in the document-level Markdown. Page-local blocks remain available for review and editing in the panel.

`max_new_tokens` is a per-page budget. The adapter sends the total document budget to the backend, preserves the `<PAGE>` protocol, and rejects truncated responses or mismatched page counts. The SGLang response is streamed internally; the document endpoint returns the complete result after successful parsing.

The client uses its normal block-based export so manual corrections are retained. The original model-generated, cross-page-merged Markdown is also retained in the document JSON's top-level `markdown.text`; it is not automatically rewritten after manual block edits.

## Native Transformers

Set `backend: transformers` to run the pinned model inside X-AnyLabeling-Server. A CUDA GPU is required. All pages are supplied to the model in the same inference call; calls are serialized because the upstream implementation temporarily mutates model state. Native inference is limited by `max_model_len` and GPU memory. Use the SGLang windowed service for documents that exceed those limits.

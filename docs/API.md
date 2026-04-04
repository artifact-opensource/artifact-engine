# Artifact Engine — API Reference

> Version 0.1.0 · OpenAI-Compatible HTTP API · CONFIDENTIAL

Artifact Engine exposes an OpenAI-compatible REST API over HTTP/1.1. The server binds to `0.0.0.0:8080` by default (configurable via `--host` and `--port`).

All endpoints return JSON with `Content-Type: application/json`. All endpoints include CORS headers (`Access-Control-Allow-Origin: *`) for browser-based clients.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `GET` | `/v1/models` | List loaded model |
| `GET` | `/health` | Health check with VRAM stats |
| `OPTIONS` | `*` | CORS preflight (returns 200 with CORS headers) |

Any other path returns `404` with an error JSON body.

---

## POST /v1/chat/completions

Generate a chat completion from a sequence of messages. Supports both streaming (SSE) and non-streaming responses.

### Request

```http
POST /v1/chat/completions HTTP/1.1
Content-Type: application/json

{
  "model": "qwen2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 2048,
  "stream": false
}
```

### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | — | Model identifier (informational; the loaded model is always used) |
| `messages` | array | **Yes** | — | Array of message objects with `role` and `content` |
| `messages[].role` | string | Yes | — | One of: `"system"`, `"user"`, `"assistant"` |
| `messages[].content` | string | Yes | — | Message text content |
| `temperature` | float | No | 0.7 | Sampling temperature. 0.0 = greedy/deterministic |
| `top_p` | float | No | 0.9 | Nucleus sampling threshold |
| `max_tokens` | integer | No | 2048 | Maximum number of tokens to generate |
| `stream` | boolean | No | false | Enable Server-Sent Events streaming |

### Non-Streaming Response (stream=false)

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "chatcmpl-1712185200",
  "object": "chat.completion",
  "created": 1712185200,
  "model": "qwen2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

### Streaming Response (stream=true)

When `stream` is `true`, the server sends the response as Server-Sent Events (SSE):

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
Access-Control-Allow-Origin: *
```

Each token is sent as an SSE event:

```
data: {"id":"chatcmpl-1712185200","object":"chat.completion.chunk","created":1712185200,"model":"qwen2","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-1712185200","object":"chat.completion.chunk","created":1712185200,"model":"qwen2","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

data: {"id":"chatcmpl-1712185200","object":"chat.completion.chunk","created":1712185200,"model":"qwen2","choices":[{"index":0,"delta":{"content":" of"},"finish_reason":null}]}
```

The final chunk includes `finish_reason`:

```
data: {"id":"chatcmpl-1712185200","object":"chat.completion.chunk","created":1712185200,"model":"qwen2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Chat Template

Messages are formatted using the ChatML template before tokenization:

```
<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
```

The assistant's response is appended by the model during generation.

### Error Responses

**Missing messages:**
```json
{
  "error": {
    "message": "No messages provided",
    "type": "invalid_request"
  }
}
```
Status: `400 Bad Request`

**Tokenization failure:**
```json
{
  "error": {
    "message": "Tokenization failed",
    "type": "server_error"
  }
}
```
Status: `500 Internal Server Error`

---

## GET /v1/models

List the currently loaded model.

### Request

```http
GET /v1/models HTTP/1.1
```

### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "object": "list",
  "data": [
    {
      "id": "qwen2",
      "object": "model",
      "owned_by": "artifact-virtual",
      "context_length": 32768
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `data[].id` | string | Model architecture identifier (extracted from GGUF metadata) |
| `data[].object` | string | Always `"model"` |
| `data[].owned_by` | string | Always `"artifact-virtual"` |
| `data[].context_length` | integer | Maximum context window (from model metadata) |

---

## GET /health

Health check endpoint. Returns the loaded model name and current VRAM usage.

### Request

```http
GET /health HTTP/1.1
```

### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "ok",
  "model": "qwen2",
  "vram_used": 7113539584,
  "vram_total": 10737418240
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` when the server is running |
| `model` | string | Architecture name from GGUF metadata |
| `vram_used` | integer | GPU memory used in bytes (tracked allocations) |
| `vram_total` | integer | Total GPU device-local memory in bytes |

---

## CORS Support

All responses include:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: POST, GET, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
```

The `OPTIONS` method returns `200 OK` with these headers and an empty body, supporting preflight requests from browser-based clients.

---

## Compatibility Notes

The API is designed to be compatible with the OpenAI Chat Completions API format. Clients that work with OpenAI (Python `openai` library, `curl`, etc.) should work with Artifact Engine by changing the `base_url`:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Artifact Engine doesn't require auth
)

response = client.chat.completions.create(
    model="qwen2",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Differences from OpenAI API

| Feature | OpenAI | Artifact Engine |
|---------|--------|-----------------|
| Authentication | API key required | None (local-only) |
| Model selection | Routes to different models | Single loaded model (model field is informational) |
| Token counting | Accurate BPE counts | Byte-level approximation (MVP tokenizer) |
| Function calling | Supported | Not implemented |
| Response format | Supported | Not implemented |
| Logprobs | Supported | Not implemented |
| N (multiple choices) | Supported | Not implemented |
| Presence/frequency penalty | Supported | Not implemented (repeat_penalty only, hardcoded) |

---

## Request Size Limits

The HTTP server reads up to **64 KB** per request in a single `recv()` call. Prompts that produce request bodies larger than 64 KB may be truncated. This will be addressed with Content-Length-based multi-read in a future version.

---

## Concurrency

The current server handles requests **synchronously** — one request at a time. Concurrent requests are queued by the TCP listen backlog (16). A thread pool for concurrent request handling is planned for a future version.

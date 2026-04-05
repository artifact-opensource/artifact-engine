# HTTP API

Artifact Engine exposes an OpenAI-compatible HTTP API for inference. This document covers all endpoints, request/response formats, and behavior details.

## Base URL

```
http://{host}:{port}
```

Default: `http://localhost:8080`

Configure with `--host` and `--port` flags.

## Authentication

None. Artifact Engine does not implement authentication. If you need access control, place it behind a reverse proxy (nginx, Caddy, etc.) with authentication middleware.

## Endpoints

### POST /v1/chat/completions

Generate a chat completion. This is the primary inference endpoint.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | loaded model | Model identifier (informational only — Artifact Engine serves one model at a time) |
| `messages` | array | Yes | — | Array of message objects |
| `max_tokens` | integer | No | 512 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0 = greedy, higher = more random) |
| `stream` | boolean | No | false | Enable streaming (SSE) — *in progress* |

**Message Object:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | `"system"`, `"user"`, or `"assistant"` |
| `content` | string | Yes | Message text content |

**Example Request:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 128,
    "temperature": 0.3
  }'
```

**Response (200 OK):**
```json
{
  "id": "chatcmpl-artifact-1",
  "object": "chat.completion",
  "created": 1743811200,
  "model": "qwen-3.5-9b",
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

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique completion identifier |
| `object` | string | Always `"chat.completion"` |
| `created` | integer | Unix timestamp of generation |
| `model` | string | Model identifier |
| `choices` | array | Array of completion choices (always length 1) |
| `choices[].index` | integer | Always `0` |
| `choices[].message.role` | string | Always `"assistant"` |
| `choices[].message.content` | string | Generated text |
| `choices[].finish_reason` | string | `"stop"` (hit EOS or stop sequence) or `"length"` (hit max_tokens) |
| `usage.prompt_tokens` | integer | Number of tokens in the prompt |
| `usage.completion_tokens` | integer | Number of tokens generated |
| `usage.total_tokens` | integer | Sum of prompt + completion tokens |

**Error Response (400):**
```json
{
  "error": {
    "message": "messages array is required",
    "type": "invalid_request_error"
  }
}
```

### GET /v1/models

List available models.

**Example Request:**
```bash
curl http://localhost:8080/v1/models
```

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-3.5-9b",
      "object": "model",
      "created": 1743811200,
      "owned_by": "artifact-engine"
    }
  ]
}
```

Artifact Engine loads one model at startup. This endpoint always returns a list with exactly one entry.

### GET /health

Health check endpoint.

**Example Request:**
```bash
curl http://localhost:8080/health
```

**Response (200 OK):**
```
OK
```

Returns `200 OK` with body `OK` when the server is running and the model is loaded. Use this for load balancer health checks and monitoring.

## Chat Template Processing

When the API receives a `messages` array, it formats the conversation using the model's chat template before tokenization.

**Chatml format (Qwen models):**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

**Llama format:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

The template is selected automatically based on the `general.architecture` field in the model's GGUF metadata.

## Sampling Behavior

**Temperature = 0.0 (greedy):**
- Always selects the token with the highest logit (argmax)
- Deterministic — same input always produces the same output
- Best for factual tasks, code generation, structured output

**Temperature > 0.0 (probabilistic):**
- Logits are divided by temperature before softmax
- Lower temperature → more focused (peaked distribution)
- Higher temperature → more creative (flatter distribution)
- Temperature = 1.0 → model's natural distribution
- Temperature > 1.5 → increasingly random, may produce incoherent output

**Stop conditions:**
1. Generated the EOS (end-of-sequence) token → `finish_reason: "stop"`
2. Reached `max_tokens` → `finish_reason: "length"`

## Context Window

The total number of tokens (prompt + completion) cannot exceed `--ctx-len` (default 2048). If the prompt alone exceeds the context length, the request is rejected with an error:

```json
{
  "error": {
    "message": "prompt exceeds context length (2048 tokens)",
    "type": "invalid_request_error"
  }
}
```

## Connection Handling

- HTTP/1.1, keep-alive not supported (connection closed after each response)
- Single-threaded request processing — requests are served sequentially
- Request body size limit: 1MB
- No CORS headers by default (add via reverse proxy if needed)

## Integration Examples

**Python (requests):**
```python
import requests

response = requests.post("http://localhost:8080/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7
})

print(response.json()["choices"][0]["message"]["content"])
```

**Python (openai SDK):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen-3.5-9b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256
)

print(response.choices[0].message.content)
```

**JavaScript (fetch):**
```javascript
const response = await fetch("http://localhost:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Hello!" }],
    max_tokens: 256
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

**cURL (one-liner):**
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}' | jq -r '.choices[0].message.content'
```

## Differences from OpenAI API

| Feature | OpenAI | Artifact Engine |
|---------|--------|----------------|
| Authentication | API key required | None |
| Multiple models | Yes | One model at a time |
| Streaming | SSE | In progress |
| Function calling | Yes | Not supported |
| Logprobs | Yes | Not supported |
| N (multiple completions) | Yes | Always 1 |
| Stop sequences | Yes | Not yet supported |
| Seed (deterministic) | Yes | Use temperature=0 |
| Vision (images) | Yes | Not supported |

The API is compatible enough that the OpenAI Python SDK works out of the box (set `base_url` and any `api_key`).

# GGUF Format

How Artifact Engine parses GGUF model files. This document covers the binary format, metadata extraction, tensor loading, and supported quantization types.

## Overview

GGUF (GPT-Generated Unified Format) is a binary file format for storing quantized LLM weights and metadata. It was created by the ggml/llama.cpp project as a successor to GGML and GGJT formats. Artifact Engine implements a from-scratch GGUF parser in `src/gguf.c` — no dependency on ggml or any external library.

## File Structure

A GGUF file has four sections, read sequentially:

```
┌─────────────────────────┐
│        Header           │  Fixed-size: magic + version + counts
├─────────────────────────┤
│     Metadata KV Pairs   │  Variable: architecture, vocab, config
├─────────────────────────┤
│     Tensor Descriptors  │  Variable: name, shape, type, offset
├─────────────────────────┤
│      Tensor Data        │  Bulk: raw weight data (aligned)
└─────────────────────────┘
```

### Header

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Magic | uint32 | 4 B | `0x46475547` ("GGUF" in little-endian) |
| Version | uint32 | 4 B | Format version (2 or 3) |
| Tensor count | uint64 | 8 B | Number of tensors in the file |
| Metadata KV count | uint64 | 8 B | Number of metadata key-value pairs |

### Metadata Key-Value Pairs

Each KV pair:

```
[key_length: uint64] [key_string: utf8] [value_type: uint32] [value_data: variable]
```

**Value types:**

| Type ID | Type | Size |
|---------|------|------|
| 0 | uint8 | 1 B |
| 1 | int8 | 1 B |
| 2 | uint16 | 2 B |
| 3 | int16 | 2 B |
| 4 | uint32 | 4 B |
| 5 | int32 | 4 B |
| 6 | float32 | 4 B |
| 7 | bool | 1 B |
| 8 | string | len + data |
| 9 | array | type + count + data |
| 10 | uint64 | 8 B |
| 11 | int64 | 8 B |
| 12 | float64 | 8 B |

Strings are stored as `[length: uint64] [utf8_data: length bytes]` (not null-terminated in the file).

Arrays are stored as `[element_type: uint32] [count: uint64] [elements...]`.

### Tensor Descriptors

Each tensor descriptor:

```
[name_length: uint64] [name: utf8] [n_dims: uint32] [dims: uint64 × n_dims] [type: uint32] [offset: uint64]
```

The offset is relative to the start of the tensor data section (after all descriptors), aligned to the tensor type's alignment requirement.

### Tensor Data

Raw tensor data packed sequentially with alignment padding. Each tensor starts at an offset aligned to its type's natural alignment (typically 32 bytes for quantized types).

## Parsing Implementation

The parser in `src/gguf.c` processes the file in a single forward pass:

```c
// Simplified flow
FILE *f = fopen(path, "rb");

// 1. Read and validate header
read_header(f, &magic, &version, &n_tensors, &n_kv);
assert(magic == 0x46475547);
assert(version == 2 || version == 3);

// 2. Read all metadata KV pairs
for (i = 0; i < n_kv; i++) {
    read_kv_pair(f, &metadata[i]);
}

// 3. Read tensor descriptors
for (i = 0; i < n_tensors; i++) {
    read_tensor_info(f, &tensors[i]);
}

// 4. Calculate tensor data base offset (with alignment)
data_offset = align_to(ftell(f), GGUF_DEFAULT_ALIGNMENT);

// 5. Load tensor data (mmap or read)
for (i = 0; i < n_tensors; i++) {
    tensors[i].data = base + data_offset + tensors[i].offset;
}
```

## Key Metadata Fields

The parser extracts these fields to configure the engine:

**Architecture:**
| Key | Example | Purpose |
|-----|---------|---------|
| `general.architecture` | `"qwen2"` | Determines attention layout, normalization type |
| `general.name` | `"Qwen2.5-9B-Instruct"` | Model display name |
| `general.file_type` | `15` | Quantization type indicator |

**Model dimensions:**
| Key | Example | Purpose |
|-----|---------|---------|
| `qwen2.embedding_length` | `4096` | Hidden dimension |
| `qwen2.block_count` | `32` | Number of transformer layers |
| `qwen2.attention.head_count` | `32` | Query attention heads |
| `qwen2.attention.head_count_kv` | `8` | Key/value heads (GQA ratio = 32/8 = 4) |
| `qwen2.feed_forward_length` | `11008` | FFN intermediate dimension |
| `qwen2.rope.freq_base` | `10000.0` | RoPE base frequency |
| `qwen2.context_length` | `32768` | Max context the model was trained for |

**Tokenizer:**
| Key | Type | Purpose |
|-----|------|---------|
| `tokenizer.ggml.model` | string | Tokenizer type (`"gpt2"`, `"llama"`) |
| `tokenizer.ggml.tokens` | string[] | Vocabulary (array of token strings) |
| `tokenizer.ggml.token_type` | uint32[] | Token types (normal, special, control, etc.) |
| `tokenizer.ggml.merges` | string[] | BPE merge rules in order of priority |
| `tokenizer.ggml.bos_token_id` | uint32 | Beginning-of-sequence token ID |
| `tokenizer.ggml.eos_token_id` | uint32 | End-of-sequence token ID |
| `tokenizer.chat_template` | string | Jinja2-style chat template |

## Tensor Naming Convention

Tensors follow a predictable naming scheme:

```
token_embd.weight                    # Token embedding table
blk.{N}.attn_q.weight               # Query projection, layer N
blk.{N}.attn_k.weight               # Key projection, layer N
blk.{N}.attn_v.weight               # Value projection, layer N
blk.{N}.attn_output.weight          # Attention output projection, layer N
blk.{N}.ffn_gate.weight             # FFN gate projection (SwiGLU)
blk.{N}.ffn_up.weight               # FFN up projection
blk.{N}.ffn_down.weight             # FFN down projection
blk.{N}.attn_norm.weight            # Pre-attention RMSNorm
blk.{N}.ffn_norm.weight             # Pre-FFN RMSNorm
output_norm.weight                   # Final RMSNorm
output.weight                       # Language model head
```

Where `{N}` is the layer index (0-based).

## Supported Quantization Types

| Type | Bits/Weight | Block Size | Description |
|------|-----------|-----------|-------------|
| F32 | 32 | 1 | Full precision float |
| F16 | 16 | 1 | Half precision float |
| Q8_0 | 8 | 32 | 8-bit quantization, 1 scale per block |
| Q4_0 | 4 | 32 | 4-bit quantization, 1 scale per block |
| Q4_1 | 4 | 32 | 4-bit with min value per block |
| Q4_K_M | 4.5 | 256 | K-quant medium (recommended) |
| Q5_K_M | 5.5 | 256 | K-quant medium, higher quality |
| Q6_K | 6.5 | 256 | K-quant, near-F16 quality |

**Q4_K_M** is the recommended quantization for Artifact Engine. It provides the best quality-to-size ratio for inference on consumer GPUs with 4-8GB VRAM.

## Dequantization

Quantized tensors are dequantized on-the-fly during computation. For Q4_K_M, each 256-element block stores:

- A primary scale factor (F16)
- A secondary scale factor (F16)
- 256 weight values packed as 4-bit integers

Dequantization: `weight = (quant_value - 8) * scale + min`

The Vulkan backend performs dequantization in the compute shader. The CPU backend dequantizes in C before matrix operations.

## Error Handling

The GGUF parser validates at every step:

- **Magic mismatch** → "Not a GGUF file" (fatal)
- **Unsupported version** → "GGUF version N not supported" (fatal, only v2/v3 supported)
- **Truncated file** → "Unexpected EOF reading tensor data" (fatal)
- **Unknown metadata type** → Skipped with warning, does not abort
- **Missing required metadata** → "Missing key: qwen2.embedding_length" (fatal)
- **Unsupported tensor type** → "Tensor type Q2_K not supported" (fatal)

All errors print a diagnostic message to stderr with the file path, byte offset, and expected vs. actual values.

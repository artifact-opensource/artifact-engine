# Tokenizer

The BPE (Byte-Pair Encoding) tokenizer implementation in Artifact Engine. Covers encoding, decoding, merge tables, special tokens, and Unicode handling.

## Overview

Artifact Engine implements a full BPE tokenizer in `src/tokenizer.c`. No external tokenizer libraries (SentencePiece, tiktoken, HuggingFace Tokenizers) are used. The tokenizer loads its vocabulary and merge rules directly from GGUF metadata, making it self-contained — the model file is the only input needed.

## Vocabulary

The vocabulary is an ordered array of token strings loaded from the `tokenizer.ggml.tokens` metadata field. Each token has:

- **ID** — integer index (position in the array)
- **String** — the text this token represents (UTF-8 encoded)
- **Type** — classification (normal, special, control, byte-fallback, unused)

Typical vocabulary sizes:

| Model Family | Vocabulary Size |
|-------------|----------------|
| Llama 3.x | 128,256 |
| Qwen 2.x | 151,936 |
| Mistral | 32,000 |

## Merge Table

BPE merge rules define how tokens are combined. Loaded from `tokenizer.ggml.merges`, each rule is a pair of tokens that merge into a new token:

```
"t h" → "th"     (merge priority 0 — highest)
"th e" → "the"   (merge priority 1)
"i n" → "in"     (merge priority 2)
...
```

Lower priority index = applied first. The merge table typically contains 50,000–150,000 rules.

### Merge Table Construction

At tokenizer initialization:

1. Parse each merge string (space-separated pair)
2. Look up both tokens in the vocabulary
3. Look up the merged result in the vocabulary
4. Store in a hash map: `(token_a, token_b) → (merged_token, priority)`

The hash map uses a simple open-addressing scheme for O(1) merge lookups during encoding.

## Encoding (Text → Token IDs)

The encoding pipeline:

```
Input text
    │
    ▼
Pre-tokenization (split on whitespace/punctuation boundaries)
    │
    ▼
UTF-8 byte decomposition
    │
    ▼
Initialize: each byte → byte-level token
    │
    ▼
Iterative BPE merging (highest priority first)
    │
    ▼
Special token injection (BOS, EOS, chat markers)
    │
    ▼
Token ID sequence
```

### Step 1: Pre-tokenization

Input text is split into chunks at natural boundaries:
- Whitespace (space, tab, newline)
- Punctuation transitions (letter→punctuation, punctuation→letter)
- Leading spaces are preserved as part of the following chunk (e.g., `" Hello"`)

Each chunk is processed independently through BPE — merges never cross chunk boundaries.

### Step 2: Byte Decomposition

Each chunk is decomposed into individual bytes. Each byte maps to a byte-level token:
- Printable ASCII bytes map to their character token (e.g., byte `0x48` → token `"H"`)
- Non-printable or high bytes map to byte-fallback tokens (e.g., byte `0xC3` → token `"<0xC3>"`)

### Step 3: BPE Merging

The core BPE algorithm runs iteratively on the byte-token sequence:

```
tokens = [byte_tokens...]

while true:
    best_pair = None
    best_priority = MAX

    // Find the highest-priority merge in the current sequence
    for i in 0..len(tokens)-1:
        pair = (tokens[i], tokens[i+1])
        if pair in merge_table and merge_table[pair].priority < best_priority:
            best_pair = (i, pair)
            best_priority = merge_table[pair].priority

    if best_pair is None:
        break  // No more merges possible

    // Apply the merge: replace all occurrences of this pair
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best_pair.pair:
            new_tokens.append(merge_table[best_pair.pair].merged)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    tokens = new_tokens
```

This runs until no more merges are possible. The result is the final token sequence for this chunk.

### Step 4: Special Tokens

After BPE encoding, special tokens are inserted:
- **BOS** (beginning of sequence) at the start
- **EOS** (end of sequence) at the end
- Chat template tokens (e.g., `<|im_start|>`, `<|im_end|>` for Chatml) at appropriate positions

Special tokens are never split by BPE — they are matched and replaced as whole units before BPE runs on the surrounding text.

## Decoding (Token IDs → Text)

Decoding is simpler than encoding:

```
Token ID sequence
    │
    ▼
Look up each ID in vocabulary → token string
    │
    ▼
Concatenate all token strings
    │
    ▼
Reassemble byte-fallback tokens into bytes
    │
    ▼
Validate UTF-8
    │
    ▼
Output text
```

### Byte-Fallback Reassembly

Some tokens represent raw bytes (e.g., `<0xC3>`, `<0xA9>`). During decoding, consecutive byte-fallback tokens are collected and reassembled into the UTF-8 bytes they represent:

```
<0xC3> <0xA9> → bytes [0xC3, 0xA9] → "é"
```

### Handling Edge Cases

- **Unknown token IDs** — mapped to a replacement string (`"<unk>"`) rather than crashing
- **Incomplete UTF-8 sequences** — partial byte-fallback sequences at the end of generation are buffered and emitted when complete, or replaced with U+FFFD
- **Control tokens in output** — special tokens (BOS, EOS, etc.) are stripped from decoded output by default

## Chat Templates

The tokenizer applies chat templates when encoding multi-turn conversations. Template format is determined by model metadata:

**Chatml (Qwen, many others):**
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

**Llama-style:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

The HTTP server's `/v1/chat/completions` endpoint automatically applies the correct template based on the model's `general.architecture` metadata.

## Performance

The tokenizer is optimized for throughput:

- **Hash table merges** — O(1) merge pair lookup instead of linear scan
- **Single-pass pre-tokenization** — no regex, no backtracking
- **Minimal allocation** — token buffers are pre-allocated and reused
- **Vocabulary loaded once** — shared across all requests

Typical encoding speed: ~500K tokens/second on a single core (measuring the tokenizer alone, not inference).

## Data Structures

```c
typedef struct {
    char **tokens;          // Vocabulary: token ID → string
    int *token_types;       // Token type per ID
    int vocab_size;         // Total vocabulary size

    MergePair *merges;      // Merge rules array (priority-ordered)
    int n_merges;           // Number of merge rules
    MergeMap merge_map;     // Hash map: (id_a, id_b) → merged_id

    int bos_token;          // Beginning-of-sequence token ID
    int eos_token;          // End-of-sequence token ID
    int pad_token;          // Padding token ID
    int unk_token;          // Unknown token ID
} Tokenizer;
```

## Limitations

- No SentencePiece unigram model support (BPE only)
- No WordPiece support (BERT-style)
- Chat template application is hardcoded per architecture rather than evaluating Jinja2
- Maximum single-encode input: 1MB of text (guard against runaway encoding on huge inputs)

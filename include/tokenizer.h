/*
 * Artifact Engine — BPE Tokenizer
 *
 * Loads tokenizer data from GGUF metadata:
 *   - tokenizer.ggml.tokens (string array — vocabulary)
 *   - tokenizer.ggml.scores (float array — merge priorities)
 *   - tokenizer.ggml.merges (string array — BPE merge rules, optional)
 *   - tokenizer.ggml.token_type (int array — normal/special/control)
 *
 * Supports both SentencePiece-style (scores) and GPT-style (merges) BPE.
 * Qwen 3.5 uses GPT-style BPE with ~152K vocabulary.
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include "gguf.h"

/* Token types */
typedef enum {
    GGUF_TOKEN_NORMAL   = 1,
    GGUF_TOKEN_UNKNOWN  = 2,
    GGUF_TOKEN_CONTROL  = 3,
    GGUF_TOKEN_USER     = 4,
    GGUF_TOKEN_UNUSED   = 5,
    GGUF_TOKEN_BYTE     = 6,
} token_type;

/* BPE merge rule */
typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t result;
    float    score;     /* priority (lower = merge first) */
} bpe_merge;

/* Tokenizer */
typedef struct {
    /* Vocabulary */
    char**      tokens;        /* [vocab_size] — token strings (owned) */
    float*      scores;        /* [vocab_size] — merge scores */
    int32_t*    types;         /* [vocab_size] — token types */
    uint32_t    vocab_size;

    /* BPE merge table */
    bpe_merge*  merges;
    uint32_t    n_merges;

    /* Special tokens */
    uint32_t    bos_id;
    uint32_t    eos_id;
    uint32_t    unk_id;
    uint32_t    pad_id;

    /* Token-to-ID hash map (for fast encoding) */
    struct {
        char*    key;
        uint32_t id;
    } *token_map;
    uint32_t    map_capacity;

    bool        loaded;
} tokenizer;

/* ───── API ───── */

/* Load tokenizer from GGUF metadata */
bool tokenizer_load(tokenizer* tok, const gguf_file* gf);

/* Encode text to token IDs. Returns allocated array (caller frees). */
uint32_t* tokenizer_encode(const tokenizer* tok, const char* text,
                           uint32_t* n_tokens, bool add_bos);

/* Decode a single token ID to string */
const char* tokenizer_decode(const tokenizer* tok, uint32_t token_id);

/* Decode array of token IDs to string. Returns allocated string (caller frees). */
char* tokenizer_decode_batch(const tokenizer* tok, const uint32_t* ids,
                             uint32_t n_tokens);

/* Free tokenizer */
void tokenizer_free(tokenizer* tok);

#endif /* TOKENIZER_H */

/*
 * Artifact Engine — BPE Tokenizer Implementation
 *
 * Two encoding modes:
 * 1. Merge-based (GPT/Qwen): Apply merge rules iteratively
 * 2. Score-based (SentencePiece/Llama): Greedy merge by score
 *
 * Both produce the same result — just different priority sources.
 */

#define _GNU_SOURCE
#include "../include/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ───── Hash Map for token→ID lookup ───── */

static uint32_t hash_str(const char* s) {
    uint32_t h = 5381;
    while (*s) h = ((h << 5) + h) + (unsigned char)*s++;
    return h;
}

static bool map_init(tokenizer* tok, uint32_t capacity) {
    tok->map_capacity = capacity;
    tok->token_map = calloc(capacity, sizeof(*tok->token_map));
    return tok->token_map != NULL;
}

static void map_set(tokenizer* tok, const char* key, uint32_t id) {
    uint32_t idx = hash_str(key) % tok->map_capacity;
    /* Linear probing */
    while (tok->token_map[idx].key != NULL) {
        if (strcmp(tok->token_map[idx].key, key) == 0) {
            tok->token_map[idx].id = id;
            return;
        }
        idx = (idx + 1) % tok->map_capacity;
    }
    tok->token_map[idx].key = strdup(key);
    tok->token_map[idx].id = id;
}

static bool map_get(const tokenizer* tok, const char* key, uint32_t* id) {
    uint32_t idx = hash_str(key) % tok->map_capacity;
    while (tok->token_map[idx].key != NULL) {
        if (strcmp(tok->token_map[idx].key, key) == 0) {
            *id = tok->token_map[idx].id;
            return true;
        }
        idx = (idx + 1) % tok->map_capacity;
    }
    return false;
}

/* ───── Load from GGUF ───── */

bool tokenizer_load(tokenizer* tok, const gguf_file* gf) {
    memset(tok, 0, sizeof(*tok));

    /* Get vocabulary tokens (string array) */
    const gguf_kv* kv_tokens = gguf_find_kv(gf, "tokenizer.ggml.tokens");
    if (!kv_tokens || kv_tokens->type != GGUF_TYPE_ARRAY) {
        fprintf(stderr, "tokenizer: missing tokenizer.ggml.tokens\n");
        return false;
    }

    tok->vocab_size = (uint32_t)kv_tokens->value.arr.count;
    printf("tokenizer: loading %u tokens\n", tok->vocab_size);

    /* Allocate vocabulary */
    tok->tokens = calloc(tok->vocab_size, sizeof(char*));
    if (!tok->tokens) return false;

    /* Extract token strings from array data */
    /* The array data for string arrays is: [uint64 len, bytes...]* */
    const uint8_t* ptr = (const uint8_t*)kv_tokens->value.arr.data;
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        uint64_t slen;
        memcpy(&slen, ptr, 8);
        ptr += 8;
        tok->tokens[i] = malloc(slen + 1);
        memcpy(tok->tokens[i], ptr, slen);
        tok->tokens[i][slen] = '\0';
        ptr += slen;
    }

    /* Get scores (float array) */
    const gguf_kv* kv_scores = gguf_find_kv(gf, "tokenizer.ggml.scores");
    if (kv_scores && kv_scores->type == GGUF_TYPE_ARRAY) {
        tok->scores = malloc(tok->vocab_size * sizeof(float));
        memcpy(tok->scores, kv_scores->value.arr.data,
               tok->vocab_size * sizeof(float));
    }

    /* Get token types (int32 array) */
    const gguf_kv* kv_types = gguf_find_kv(gf, "tokenizer.ggml.token_type");
    if (kv_types && kv_types->type == GGUF_TYPE_ARRAY) {
        tok->types = malloc(tok->vocab_size * sizeof(int32_t));
        memcpy(tok->types, kv_types->value.arr.data,
               tok->vocab_size * sizeof(int32_t));
    }

    /* Get merge rules (string array — GPT-style BPE) */
    const gguf_kv* kv_merges = gguf_find_kv(gf, "tokenizer.ggml.merges");
    if (kv_merges && kv_merges->type == GGUF_TYPE_ARRAY) {
        tok->n_merges = (uint32_t)kv_merges->value.arr.count;
        printf("tokenizer: loading %u merge rules\n", tok->n_merges);
        tok->merges = calloc(tok->n_merges, sizeof(bpe_merge));

        /* Build hash map first (need it to look up token IDs for merges) */
        if (!map_init(tok, tok->vocab_size * 3)) return false;
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            map_set(tok, tok->tokens[i], i);
        }

        /* Parse merges: each is "tokenA tokenB" */
        const uint8_t* mptr = (const uint8_t*)kv_merges->value.arr.data;
        for (uint32_t i = 0; i < tok->n_merges; i++) {
            uint64_t mlen;
            memcpy(&mlen, mptr, 8);
            mptr += 8;

            /* Temporary copy for parsing */
            char* merge_str = malloc(mlen + 1);
            memcpy(merge_str, mptr, mlen);
            merge_str[mlen] = '\0';
            mptr += mlen;

            /* Split on first space */
            char* space = strchr(merge_str, ' ');
            if (space) {
                *space = '\0';
                const char* left_str = merge_str;
                const char* right_str = space + 1;

                /* Concatenate to find result token */
                char* result_str = malloc(strlen(left_str) + strlen(right_str) + 1);
                sprintf(result_str, "%s%s", left_str, right_str);

                uint32_t left_id, right_id, result_id;
                if (map_get(tok, left_str, &left_id) &&
                    map_get(tok, right_str, &right_id) &&
                    map_get(tok, result_str, &result_id)) {
                    tok->merges[i].left = left_id;
                    tok->merges[i].right = right_id;
                    tok->merges[i].result = result_id;
                    tok->merges[i].score = (float)i;  /* priority = order */
                }
                free(result_str);
            }
            free(merge_str);
        }
        tok->loaded = true;
    } else {
        /* Score-based BPE (SentencePiece) — no merge rules needed */
        if (!map_init(tok, tok->vocab_size * 3)) return false;
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            map_set(tok, tok->tokens[i], i);
        }
        tok->loaded = true;
    }

    /* Special token IDs */
    const gguf_kv* kv;
    kv = gguf_find_kv(gf, "tokenizer.ggml.bos_token_id");
    tok->bos_id = kv ? kv->value.u32 : 1;
    kv = gguf_find_kv(gf, "tokenizer.ggml.eos_token_id");
    tok->eos_id = kv ? kv->value.u32 : 2;
    kv = gguf_find_kv(gf, "tokenizer.ggml.unknown_token_id");
    tok->unk_id = kv ? kv->value.u32 : 0;
    kv = gguf_find_kv(gf, "tokenizer.ggml.padding_token_id");
    tok->pad_id = kv ? kv->value.u32 : 0;

    printf("tokenizer: loaded (vocab=%u, merges=%u, bos=%u, eos=%u)\n",
           tok->vocab_size, tok->n_merges, tok->bos_id, tok->eos_id);
    return true;
}

/* ───── BPE Encoding ───── */

/* Linked list node for BPE encoding */
typedef struct bpe_node {
    uint32_t token_id;
    struct bpe_node* next;
    struct bpe_node* prev;
} bpe_node;

/* Find the best merge pair in the current sequence */
static bool find_best_merge(const tokenizer* tok, bpe_node* head,
                            bpe_node** best_node, uint32_t* best_result) {
    float best_score = 1e30f;
    *best_node = NULL;

    for (bpe_node* n = head; n && n->next; n = n->next) {
        /* Check if this pair has a merge rule */
        if (tok->n_merges > 0) {
            /* GPT-style: search merges by pair */
            for (uint32_t m = 0; m < tok->n_merges; m++) {
                if (tok->merges[m].left == n->token_id &&
                    tok->merges[m].right == n->next->token_id) {
                    if (tok->merges[m].score < best_score) {
                        best_score = tok->merges[m].score;
                        *best_node = n;
                        *best_result = tok->merges[m].result;
                    }
                    break;  /* First match wins for this pair */
                }
            }
        } else if (tok->scores) {
            /* SentencePiece-style: merge by concatenation lookup */
            const char* left_str = tok->tokens[n->token_id];
            const char* right_str = tok->tokens[n->next->token_id];
            size_t llen = strlen(left_str);
            size_t rlen = strlen(right_str);
            char* merged = malloc(llen + rlen + 1);
            memcpy(merged, left_str, llen);
            memcpy(merged + llen, right_str, rlen);
            merged[llen + rlen] = '\0';

            uint32_t merged_id;
            if (map_get(tok, merged, &merged_id)) {
                float score = tok->scores[merged_id];
                if (score < best_score) {
                    best_score = score;
                    *best_node = n;
                    *best_result = merged_id;
                }
            }
            free(merged);
        }
    }

    return *best_node != NULL;
}

uint32_t* tokenizer_encode(const tokenizer* tok, const char* text,
                           uint32_t* n_tokens, bool add_bos) {
    if (!tok->loaded) return NULL;

    /* Step 1: Convert text to initial token sequence (one token per byte/char) */
    size_t text_len = strlen(text);
    bpe_node* head = NULL;
    bpe_node* tail = NULL;
    uint32_t seq_len = 0;

    /* For BPE models with byte fallback, start with individual UTF-8 bytes
     * mapped to their byte tokens. For models without byte tokens, start
     * with the longest token match (greedy). */

    /* Greedy longest-match initial tokenization */
    size_t pos = 0;
    while (pos < text_len) {
        uint32_t best_id = tok->unk_id;
        size_t best_len = 1;

        /* Try matching progressively longer substrings */
        for (size_t end = pos + 1; end <= text_len && end - pos <= 64; end++) {
            char buf[65];
            size_t len = end - pos;
            memcpy(buf, text + pos, len);
            buf[len] = '\0';

            uint32_t id;
            if (map_get(tok, buf, &id)) {
                best_id = id;
                best_len = len;
            }
        }

        /* If no match found, try single byte as <0xNN> token */
        if (best_id == tok->unk_id && best_len == 1) {
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>",
                     (unsigned char)text[pos]);
            uint32_t byte_id;
            if (map_get(tok, byte_tok, &byte_id)) {
                best_id = byte_id;
            }
        }

        /* Add to linked list */
        bpe_node* node = malloc(sizeof(bpe_node));
        node->token_id = best_id;
        node->next = NULL;
        node->prev = tail;
        if (tail) tail->next = node;
        else head = node;
        tail = node;
        seq_len++;

        pos += best_len;
    }

    /* Step 2: Iteratively apply BPE merges */
    bool changed = true;
    while (changed) {
        changed = false;
        bpe_node* best_node;
        uint32_t best_result;

        if (find_best_merge(tok, head, &best_node, &best_result)) {
            /* Apply merge: replace pair with merged token */
            best_node->token_id = best_result;
            bpe_node* to_remove = best_node->next;
            best_node->next = to_remove->next;
            if (to_remove->next) to_remove->next->prev = best_node;
            else tail = best_node;
            free(to_remove);
            seq_len--;
            changed = true;
        }
    }

    /* Step 3: Convert linked list to array */
    uint32_t extra = add_bos ? 1 : 0;
    uint32_t* ids = malloc((seq_len + extra) * sizeof(uint32_t));
    uint32_t idx = 0;

    if (add_bos) ids[idx++] = tok->bos_id;

    bpe_node* n = head;
    while (n) {
        ids[idx++] = n->token_id;
        bpe_node* next = n->next;
        free(n);
        n = next;
    }

    *n_tokens = seq_len + extra;
    return ids;
}

/* ───── Decoding ───── */

const char* tokenizer_decode(const tokenizer* tok, uint32_t token_id) {
    if (token_id >= tok->vocab_size) return "?";

    /* Handle byte tokens like <0xNN> */
    const char* s = tok->tokens[token_id];
    if (s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
        static char byte_buf[2];
        unsigned val;
        if (sscanf(s, "<0x%02X>", &val) == 1) {
            byte_buf[0] = (char)val;
            byte_buf[1] = '\0';
            return byte_buf;
        }
    }

    return s;
}

char* tokenizer_decode_batch(const tokenizer* tok, const uint32_t* ids,
                             uint32_t n_tokens) {
    /* First pass: calculate total length */
    size_t total = 0;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (ids[i] == tok->bos_id || ids[i] == tok->eos_id) continue;
        const char* s = tokenizer_decode(tok, ids[i]);
        total += strlen(s);
    }

    /* Second pass: build string */
    char* result = malloc(total + 1);
    char* p = result;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (ids[i] == tok->bos_id || ids[i] == tok->eos_id) continue;
        const char* s = tokenizer_decode(tok, ids[i]);
        size_t len = strlen(s);
        memcpy(p, s, len);
        p += len;
    }
    *p = '\0';

    return result;
}

/* ───── Cleanup ───── */

void tokenizer_free(tokenizer* tok) {
    if (tok->tokens) {
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->tokens[i]);
        }
        free(tok->tokens);
    }
    free(tok->scores);
    free(tok->types);
    free(tok->merges);

    if (tok->token_map) {
        for (uint32_t i = 0; i < tok->map_capacity; i++) {
            free(tok->token_map[i].key);
        }
        free(tok->token_map);
    }

    memset(tok, 0, sizeof(*tok));
}

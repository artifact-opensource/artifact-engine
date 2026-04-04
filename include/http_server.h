/*
 * Artifact Engine — HTTP Server Interface
 * OpenAI-compatible API
 */

#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include "engine.h"
#include <stdbool.h>
#include <stdint.h>

/* ───── Chat Message ───── */
typedef struct {
    char* role;     /* "system", "user", "assistant" */
    char* content;
} chat_message;

/* ───── Chat Completion Request ───── */
typedef struct {
    char*          model;
    chat_message*  messages;
    uint32_t       n_messages;
    float          temperature;
    float          top_p;
    uint32_t       max_tokens;
    bool           stream;
} chat_request;

/* ───── Server Config ───── */
typedef struct {
    uint16_t port;
    char*    host;
    char*    model_path;
    char*    shader_dir;
    uint32_t max_context;
    uint32_t n_threads;   /* for tokenization, not GPU */
} server_config;

/* ───── API ───── */

/* Start HTTP server (blocks) */
bool server_start(engine* eng, const server_config* cfg);

/* Stop server gracefully */
void server_stop(void);

#endif /* HTTP_SERVER_H */

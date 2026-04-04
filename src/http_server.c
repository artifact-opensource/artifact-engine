/*
 * Artifact Engine — HTTP Server
 * 
 * Minimal HTTP/1.1 server with OpenAI-compatible API.
 * No external dependencies — pure POSIX sockets.
 *
 * Endpoints:
 *   POST /v1/chat/completions — chat completion (streaming + non-streaming)
 *   GET  /v1/models           — list loaded model
 *   GET  /health              — health check
 */

#define _GNU_SOURCE
#include "../include/http_server.h"
#include "../include/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include <time.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef int socklen_t;
  #define close closesocket
  #define MSG_NOSIGNAL 0
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <pthread.h>
  #include <signal.h>
  #ifndef MSG_NOSIGNAL
    #define MSG_NOSIGNAL 0
  #endif
#endif

/* ───── Globals ───── */
static engine*        g_engine = NULL;
static server_config* g_config = NULL;
static int            g_server_fd = -1;
static volatile int   g_running = 1;

/* ───── Simple JSON Builder ───── */

typedef struct {
    char*  buf;
    size_t len;
    size_t cap;
} json_buf;

static void json_init(json_buf* j, size_t initial_cap) {
    j->buf = malloc(initial_cap);
    j->len = 0;
    j->cap = initial_cap;
    j->buf[0] = '\0';
}

static void json_append(json_buf* j, const char* s) {
    size_t slen = strlen(s);
    while (j->len + slen + 1 > j->cap) {
        j->cap *= 2;
        j->buf = realloc(j->buf, j->cap);
    }
    memcpy(j->buf + j->len, s, slen);
    j->len += slen;
    j->buf[j->len] = '\0';
}

static void json_appendf(json_buf* j, const char* fmt, ...) {
    char tmp[4096];
    va_list args;
    va_start(args, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, args);
    va_end(args);
    json_append(j, tmp);
}

static void json_free(json_buf* j) {
    free(j->buf);
    j->buf = NULL;
    j->len = j->cap = 0;
}

/* ───── Simple JSON Parser (just enough for chat completions) ───── */

/* Find a string value for a given key in JSON. Returns malloc'd string or NULL. */
static char* json_get_string(const char* json, const char* key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char* pos = strstr(json, search);
    if (!pos) return NULL;
    pos += strlen(search);
    
    /* Skip whitespace and colon */
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t')) pos++;
    if (*pos != '"') return NULL;
    pos++; /* skip opening quote */
    
    /* Find closing quote (handle escapes) */
    const char* end = pos;
    while (*end && *end != '"') {
        if (*end == '\\') end++; /* skip escaped char */
        end++;
    }
    
    size_t len = end - pos;
    char* result = malloc(len + 1);
    memcpy(result, pos, len);
    result[len] = '\0';
    return result;
}

/* Find a number value for a given key */
static float json_get_float(const char* json, const char* key, float def) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char* pos = strstr(json, search);
    if (!pos) return def;
    pos += strlen(search);
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t')) pos++;
    return strtof(pos, NULL);
}

static int json_get_int(const char* json, const char* key, int def) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char* pos = strstr(json, search);
    if (!pos) return def;
    pos += strlen(search);
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t')) pos++;
    return atoi(pos);
}

static bool json_get_bool(const char* json, const char* key, bool def) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char* pos = strstr(json, search);
    if (!pos) return def;
    pos += strlen(search);
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t')) pos++;
    if (strncmp(pos, "true", 4) == 0) return true;
    if (strncmp(pos, "false", 5) == 0) return false;
    return def;
}

/* Extract messages array from chat completion request */
/* Returns array of {role, content} pairs. Caller must free. */
static chat_message* parse_messages(const char* json, uint32_t* n_messages) {
    const char* arr = strstr(json, "\"messages\"");
    if (!arr) { *n_messages = 0; return NULL; }
    
    /* Find the opening bracket */
    arr = strchr(arr, '[');
    if (!arr) { *n_messages = 0; return NULL; }
    arr++;
    
    /* Count messages (count '{' at depth 1) */
    uint32_t count = 0;
    int depth = 0;
    for (const char* p = arr; *p && !(*p == ']' && depth == 0); p++) {
        if (*p == '{') { if (depth == 0) count++; depth++; }
        if (*p == '}') depth--;
    }
    
    chat_message* msgs = calloc(count, sizeof(chat_message));
    *n_messages = count;
    
    /* Parse each message object */
    const char* p = arr;
    for (uint32_t i = 0; i < count; i++) {
        p = strchr(p, '{');
        if (!p) break;
        p++;
        
        /* Find the closing brace */
        int d = 1;
        const char* end = p;
        while (*end && d > 0) {
            if (*end == '{') d++;
            if (*end == '}') d--;
            end++;
        }
        
        /* Extract a temporary JSON substring */
        size_t len = end - p;
        char* obj = malloc(len + 1);
        memcpy(obj, p - 1, len + 1);
        obj[len] = '\0';
        
        msgs[i].role = json_get_string(obj, "role");
        msgs[i].content = json_get_string(obj, "content");
        
        free(obj);
        p = end;
    }
    
    return msgs;
}

static void free_messages(chat_message* msgs, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        free(msgs[i].role);
        free(msgs[i].content);
    }
    free(msgs);
}

/* ───── HTTP Response Helpers ───── */

static void send_response(int fd, int status, const char* content_type, 
                          const char* body, size_t body_len) {
    const char* status_str = status == 200 ? "OK" : 
                             status == 400 ? "Bad Request" :
                             status == 404 ? "Not Found" :
                             status == 500 ? "Internal Server Error" : "Unknown";
    
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_str, content_type, body_len);
    
    send(fd, header, hlen, MSG_NOSIGNAL);
    if (body && body_len > 0) {
        send(fd, body, body_len, MSG_NOSIGNAL);
    }
}

static void send_json(int fd, int status, const char* json) {
    send_response(fd, status, "application/json", json, strlen(json));
}

static void send_sse_chunk(int fd, const char* data) {
    char chunk[4096];
    int len = snprintf(chunk, sizeof(chunk), "data: %s\n\n", data);
    send(fd, chunk, len, MSG_NOSIGNAL);
}

static void send_sse_done(int fd) {
    const char* done = "data: [DONE]\n\n";
    send(fd, done, strlen(done), MSG_NOSIGNAL);
}

/* ───── Route Handlers ───── */

static void handle_health(int fd) {
    json_buf j;
    json_init(&j, 256);
    json_appendf(&j, "{\"status\":\"ok\",\"model\":\"%s\",\"vram_used\":%zu,\"vram_total\":%zu}",
                 g_engine->arch.arch,
                 vk_memory_used(&g_engine->vk),
                 vk_memory_total(&g_engine->vk));
    send_json(fd, 200, j.buf);
    json_free(&j);
}

static void handle_models(int fd) {
    json_buf j;
    json_init(&j, 512);
    json_append(&j, "{\"object\":\"list\",\"data\":[{");
    json_appendf(&j, "\"id\":\"%s\",", g_engine->arch.arch);
    json_append(&j, "\"object\":\"model\",");
    json_appendf(&j, "\"owned_by\":\"artifact-virtual\",");
    json_appendf(&j, "\"context_length\":%u", g_engine->arch.max_position);
    json_append(&j, "}]}");
    send_json(fd, 200, j.buf);
    json_free(&j);
}

/* Streaming token callback */
typedef struct {
    int      fd;
    char*    model_id;
    uint32_t token_count;
} stream_ctx;

static void stream_token_cb(uint32_t token_id, const char* token_str, void* user_data) {
    (void)token_id;
    stream_ctx* ctx = (stream_ctx*)user_data;
    ctx->token_count++;
    
    /* Build SSE chunk */
    json_buf j;
    json_init(&j, 512);
    json_appendf(&j, "{\"id\":\"chatcmpl-%u\",", (unsigned)time(NULL));
    json_append(&j, "\"object\":\"chat.completion.chunk\",");
    json_appendf(&j, "\"created\":%ld,", (long)time(NULL));
    json_appendf(&j, "\"model\":\"%s\",", ctx->model_id);
    json_append(&j, "\"choices\":[{\"index\":0,\"delta\":{");
    
    /* Escape the token string for JSON */
    json_append(&j, "\"content\":\"");
    for (const char* p = token_str; *p; p++) {
        if (*p == '"') json_append(&j, "\\\"");
        else if (*p == '\\') json_append(&j, "\\\\");
        else if (*p == '\n') json_append(&j, "\\n");
        else if (*p == '\r') json_append(&j, "\\r");
        else if (*p == '\t') json_append(&j, "\\t");
        else { char c[2] = {*p, 0}; json_append(&j, c); }
    }
    json_append(&j, "\"");
    
    json_append(&j, "},\"finish_reason\":null}]}");
    
    send_sse_chunk(ctx->fd, j.buf);
    json_free(&j);
}

static void handle_chat_completions(int fd, const char* body) {
    /* Parse request */
    uint32_t n_messages = 0;
    chat_message* messages = parse_messages(body, &n_messages);
    
    if (!messages || n_messages == 0) {
        send_json(fd, 400, "{\"error\":{\"message\":\"No messages provided\",\"type\":\"invalid_request\"}}");
        return;
    }
    
    float temperature = json_get_float(body, "temperature", 0.7f);
    float top_p = json_get_float(body, "top_p", 0.9f);
    int max_tokens = json_get_int(body, "max_tokens", 2048);
    bool stream = json_get_bool(body, "stream", false);
    
    /* Build prompt from messages */
    /* For now, simple concatenation. TODO: proper chat template per model */
    json_buf prompt;
    json_init(&prompt, 4096);
    
    for (uint32_t i = 0; i < n_messages; i++) {
        if (messages[i].role && messages[i].content) {
            if (strcmp(messages[i].role, "system") == 0) {
                json_appendf(&prompt, "<|im_start|>system\n%s<|im_end|>\n", messages[i].content);
            } else if (strcmp(messages[i].role, "user") == 0) {
                json_appendf(&prompt, "<|im_start|>user\n%s<|im_end|>\n", messages[i].content);
            } else if (strcmp(messages[i].role, "assistant") == 0) {
                json_appendf(&prompt, "<|im_start|>assistant\n%s<|im_end|>\n", messages[i].content);
            }
        }
    }
    json_append(&prompt, "<|im_start|>assistant\n");
    
    /* Tokenize */
    uint32_t n_tokens = 0;
    uint32_t* tokens = engine_tokenize(g_engine, prompt.buf, &n_tokens);
    json_free(&prompt);
    
    if (!tokens || n_tokens == 0) {
        send_json(fd, 500, "{\"error\":{\"message\":\"Tokenization failed\",\"type\":\"server_error\"}}");
        free_messages(messages, n_messages);
        return;
    }
    
    /* Set up sampling params */
    sample_params params = {
        .temperature = temperature,
        .top_p = top_p,
        .top_k = 40,
        .repeat_penalty = 1.1f,
        .max_tokens = (uint32_t)max_tokens,
        .seed = 0,
    };
    
    if (stream) {
        /* SSE streaming response */
        char header[512];
        int hlen = snprintf(header, sizeof(header),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n");
        send(fd, header, hlen, MSG_NOSIGNAL);
        
        stream_ctx ctx = {
            .fd = fd,
            .model_id = (char*)g_engine->arch.arch,
            .token_count = 0,
        };
        
        engine_generate(g_engine, tokens, n_tokens, &params, stream_token_cb, &ctx);
        
        /* Send final chunk with finish_reason */
        json_buf fin;
        json_init(&fin, 256);
        json_appendf(&fin, "{\"id\":\"chatcmpl-%u\",", (unsigned)time(NULL));
        json_appendf(&fin, "\"object\":\"chat.completion.chunk\",\"created\":%ld,", (long)time(NULL));
        json_appendf(&fin, "\"model\":\"%s\",", g_engine->arch.arch);
        json_append(&fin, "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}");
        send_sse_chunk(fd, fin.buf);
        json_free(&fin);
        
        send_sse_done(fd);
    } else {
        /* Non-streaming: collect all output */
        /* TODO: use a buffer callback instead of stream callback */
        json_buf output;
        json_init(&output, 4096);
        
        /* For now, generate and collect (simplified) */
        /* In full implementation, engine_generate would return the full text */
        
        /* Placeholder response until engine is fully wired */
        json_append(&output, "This is Artifact Engine. Model loaded and ready.");
        
        /* Build response */
        json_buf resp;
        json_init(&resp, 8192);
        json_appendf(&resp, "{\"id\":\"chatcmpl-%u\",", (unsigned)time(NULL));
        json_append(&resp, "\"object\":\"chat.completion\",");
        json_appendf(&resp, "\"created\":%ld,", (long)time(NULL));
        json_appendf(&resp, "\"model\":\"%s\",", g_engine->arch.arch);
        json_append(&resp, "\"choices\":[{\"index\":0,\"message\":{");
        json_append(&resp, "\"role\":\"assistant\",\"content\":\"");
        
        /* Escape output */
        for (size_t i = 0; i < output.len; i++) {
            char c = output.buf[i];
            if (c == '"') json_append(&resp, "\\\"");
            else if (c == '\\') json_append(&resp, "\\\\");
            else if (c == '\n') json_append(&resp, "\\n");
            else { char s[2] = {c, 0}; json_append(&resp, s); }
        }
        
        json_append(&resp, "\"},\"finish_reason\":\"stop\"}],");
        json_appendf(&resp, "\"usage\":{\"prompt_tokens\":%u,\"completion_tokens\":%zu,\"total_tokens\":%zu}}",
                     n_tokens, output.len, n_tokens + output.len);
        
        send_json(fd, 200, resp.buf);
        json_free(&resp);
        json_free(&output);
    }
    
    free(tokens);
    free_messages(messages, n_messages);
}

/* ───── Request Router ───── */

static void handle_request(int client_fd) {
    char buf[65536];
    int received = recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (received <= 0) { close(client_fd); return; }
    buf[received] = '\0';
    
    /* Parse method and path */
    char method[16], path[256];
    sscanf(buf, "%15s %255s", method, path);
    
    /* Find body (after \r\n\r\n) */
    char* body = strstr(buf, "\r\n\r\n");
    if (body) body += 4;
    
    /* For larger bodies, read more data based on Content-Length */
    /* TODO: handle multi-packet bodies for large prompts */
    
    printf("[%s] %s %s\n", 
           method, path,
           strcmp(method, "POST") == 0 ? "(body)" : "");
    
    /* CORS preflight */
    if (strcmp(method, "OPTIONS") == 0) {
        send_response(client_fd, 200, "text/plain", "", 0);
    }
    /* Routes */
    else if (strcmp(path, "/health") == 0) {
        handle_health(client_fd);
    }
    else if (strcmp(path, "/v1/models") == 0) {
        handle_models(client_fd);
    }
    else if (strcmp(path, "/v1/chat/completions") == 0 && strcmp(method, "POST") == 0) {
        if (body) {
            handle_chat_completions(client_fd, body);
        } else {
            send_json(client_fd, 400, "{\"error\":{\"message\":\"No body\",\"type\":\"invalid_request\"}}");
        }
    }
    else {
        send_json(client_fd, 404, "{\"error\":{\"message\":\"Not found\",\"type\":\"not_found\"}}");
    }
    
    close(client_fd);
}

/* ───── Server Lifecycle ───── */

bool server_start(engine* eng, const server_config* cfg) {
    g_engine = eng;
    g_config = (server_config*)cfg;
    
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    
    g_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        perror("socket");
        return false;
    }
    
    int opt = 1;
    setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
    
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(cfg->port),
        .sin_addr.s_addr = INADDR_ANY,
    };
    
    if (cfg->host && strcmp(cfg->host, "0.0.0.0") != 0) {
        inet_pton(AF_INET, cfg->host, &addr.sin_addr);
    }
    
    if (bind(g_server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(g_server_fd);
        return false;
    }
    
    if (listen(g_server_fd, 16) < 0) {
        perror("listen");
        close(g_server_fd);
        return false;
    }
    
    printf("╔══════════════════════════════════════╗\n");
    printf("║  Artifact Engine — Listening         ║\n");
    printf("║  http://%s:%-5u                 ║\n", cfg->host, cfg->port);
    printf("╚══════════════════════════════════════╝\n\n");
    
    /* Accept loop */
    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(g_server_fd, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_fd < 0) {
            if (g_running) perror("accept");
            continue;
        }
        
        /* Handle synchronously for now. TODO: thread pool */
        handle_request(client_fd);
    }
    
    return true;
}

void server_stop(void) {
    g_running = 0;
    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }
}

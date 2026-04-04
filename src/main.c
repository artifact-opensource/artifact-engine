/*
 * Artifact Engine — Main Entry Point
 *
 * Usage: artifact-engine --model <path.gguf> [--port 8080] [--ctx 4096]
 *
 * Modes:
 *   --info     Just print model info and exit
 *   --server   Run HTTP API server (default)
 *   --bench    Run a simple benchmark
 */

#include "../include/gguf.h"
#include "../include/engine.h"
#include "../include/http_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#define VERSION "0.1.0"

static volatile int running = 1;

static void signal_handler(int sig) {
    (void)sig;
    running = 0;
    server_stop();
}

static void print_usage(const char* argv0) {
#ifdef CPU_ONLY
    printf("Artifact Engine v%s — CPU LLM Inference\n", VERSION);
#else
    printf("Artifact Engine v%s — GPU LLM Inference via Vulkan\n", VERSION);
#endif
    printf("Artifact Virtual — artifact.cloud\n\n");
    printf("Usage: %s --model <path.gguf> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --model <path>    Path to GGUF model file (required)\n");
    printf("  --port <port>     HTTP server port (default: 8080)\n");
    printf("  --host <host>     HTTP server host (default: 0.0.0.0)\n");
    printf("  --ctx <length>    Max context length (default: 4096)\n");
    printf("  --shaders <dir>   Path to compiled shader directory (default: ./shaders)\n");
    printf("  --info            Print model info and exit\n");
    printf("  --bench           Run inference benchmark\n");
    printf("  --help            Show this help\n");
    printf("\nExamples:\n");
    printf("  %s --model qwen3.5-9b-q4_k_m.gguf\n", argv0);
    printf("  %s --model qwen3.5-9b-q4_k_m.gguf --port 8080 --ctx 8192\n", argv0);
    printf("\nAPI Endpoints:\n");
    printf("  POST /v1/chat/completions    OpenAI-compatible chat\n");
    printf("  GET  /v1/models              List loaded model\n");
    printf("  GET  /health                 Health check\n");
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* shader_dir = "./shaders";
    const char* host = "0.0.0.0";
    uint16_t port = 8080;
    uint32_t ctx_len = 4096;
    int mode_info = 0;
    int mode_bench = 0;
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            ctx_len = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--shaders") == 0 && i + 1 < argc) {
            shader_dir = argv[++i];
        } else if (strcmp(argv[i], "--info") == 0) {
            mode_info = 1;
        } else if (strcmp(argv[i], "--bench") == 0) {
            mode_bench = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!model_path) {
        fprintf(stderr, "Error: --model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    /* ─── Info Mode: Just load and print ─── */
    if (mode_info) {
        printf("Loading model: %s\n", model_path);
        gguf_file* gf = gguf_load(model_path);
        if (!gf) {
            fprintf(stderr, "Failed to load GGUF file\n");
            return 1;
        }
        gguf_print_info(gf);
        gguf_free(gf);
        return 0;
    }
    
    /* ─── Signal handling ─── */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* ─── Initialize Engine ─── */
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║       ARTIFACT ENGINE v%s         ║\n", VERSION);
    printf("║       Artifact Virtual               ║\n");
    printf("╚══════════════════════════════════════╝\n\n");
    
    engine eng = {0};
    
#ifdef CPU_ONLY
    printf("[1/4] Initializing CPU compute...\n");
#else
    printf("[1/4] Initializing Vulkan...\n");
#endif
    if (!engine_init(&eng, shader_dir)) {
        fprintf(stderr, "Failed to initialize compute backend\n");
        return 1;
    }
    
    printf("[2/4] Loading model: %s\n", model_path);
    if (!engine_load_model(&eng, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        engine_destroy(&eng);
        return 1;
    }
    
    printf("[3/4] Allocating KV cache (ctx=%u)...\n", ctx_len);
    if (!engine_alloc_cache(&eng, ctx_len)) {
        fprintf(stderr, "Failed to allocate KV cache (out of VRAM?)\n");
        engine_destroy(&eng);
        return 1;
    }
    
    engine_print_status(&eng);
    
    /* ─── Bench Mode ─── */
    if (mode_bench) {
        printf("\n[BENCH] Running benchmark...\n");
        /* Simple benchmark: encode a short prompt, measure tokens/sec */
        const char* prompt = "The capital of France is";
        uint32_t n_tokens = 0;
        uint32_t* tokens = engine_tokenize(&eng, prompt, &n_tokens);
        
        if (tokens && n_tokens > 0) {
            printf("[BENCH] Prompt: \"%s\" (%u tokens)\n", prompt, n_tokens);
            
            /* TODO: timing code */
            sample_params params = {
                .temperature = 0.0f, /* greedy for benchmark */
                .top_p = 1.0f,
                .top_k = 1,
                .max_tokens = 32,
                .seed = 42,
            };
            
            engine_generate(&eng, tokens, n_tokens, &params, NULL, NULL);
            free(tokens);
        }
        
        engine_destroy(&eng);
        return 0;
    }
    
    /* ─── Server Mode ─── */
    printf("[4/4] Starting HTTP server on %s:%u...\n\n", host, port);
    
    server_config cfg = {
        .port = port,
        .host = (char*)host,
        .model_path = (char*)model_path,
        .shader_dir = (char*)shader_dir,
        .max_context = ctx_len,
        .n_threads = 4,
    };
    
    if (!server_start(&eng, &cfg)) {
        fprintf(stderr, "Failed to start HTTP server\n");
        engine_destroy(&eng);
        return 1;
    }
    
    /* server_start blocks until stopped */
    
    printf("\nShutting down...\n");
    engine_destroy(&eng);
    printf("Done.\n");
    
    return 0;
}

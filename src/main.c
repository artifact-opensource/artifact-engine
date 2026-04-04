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
#include "../include/model_fetch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef _WIN32
  #include <windows.h>
#endif

#define VERSION "0.4.0"

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
    printf("  --pull <url>      Download model from HTTP URL before starting\n");
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

#ifdef _WIN32
/* Xbox/UWP needs a message loop to stay alive. Run engine in a thread. */
static DWORD WINAPI engine_thread(LPVOID param);
static int g_argc;
static char** g_argv;

/* Minimal hidden window + message pump for Xbox process lifetime */
static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

static void run_message_pump(void) {
    WNDCLASSA wc = {0};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "ArtifactEngine";
    RegisterClassA(&wc);
    
    /* Create a hidden window — Xbox needs this to keep the process alive */
    HWND hwnd = CreateWindowA("ArtifactEngine", "Artifact Engine", 
                               0, 0, 0, 1, 1, NULL, NULL, wc.hInstance, NULL);
    (void)hwnd;
    
    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
}
#endif

int engine_main(int argc, char** argv);

int main(int argc, char** argv) {
#ifdef _WIN32
    /* Redirect stdout/stderr to a log file for debugging on Xbox */
    const char* appdata = getenv("LOCALAPPDATA");
    if (appdata) {
        char logpath[512];
        snprintf(logpath, sizeof(logpath), "%s\\..\\LocalState\\engine.log", appdata);
        FILE* lf = fopen(logpath, "w");
        if (lf) { fclose(lf); freopen(logpath, "w", stdout); freopen(logpath, "a", stderr); }
    }
    
    /* Start engine in a background thread */
    g_argc = argc;
    g_argv = argv;
    CreateThread(NULL, 0, engine_thread, NULL, 0, NULL);
    
    /* Run message pump on main thread (keeps Xbox from killing us) */
    run_message_pump();
    return 0;
#else
    return engine_main(argc, argv);
#endif
}

#ifdef _WIN32
static DWORD WINAPI engine_thread(LPVOID param) {
    (void)param;
    int ret = engine_main(g_argc, g_argv);
    if (ret != 0) {
        fprintf(stderr, "Engine exited with code %d\n", ret);
        /* Don't exit — keep the process alive for debugging via log file */
        Sleep(INFINITE);
    }
    return ret;
}
#endif

int engine_main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* pull_url = NULL;
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
        } else if (strcmp(argv[i], "--pull") == 0 && i + 1 < argc) {
            pull_url = argv[++i];
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
    
    /* ─── Pull model from URL if requested ─── */
    if (pull_url && !model_path) {
        const char* fname = model_fetch_filename(pull_url);
        static char pull_path[512];
#ifdef _WIN32
        const char* appdata = getenv("LOCALAPPDATA");
        if (appdata) {
            snprintf(pull_path, sizeof(pull_path), "%s\\..\\LocalState\\%s", appdata, fname);
        } else {
            snprintf(pull_path, sizeof(pull_path), "models\\%s", fname);
        }
#else
        snprintf(pull_path, sizeof(pull_path), "models/%s", fname);
#endif
        /* Check if already downloaded */
        FILE* check = fopen(pull_path, "rb");
        if (check) {
            fseek(check, 0, SEEK_END);
            long fsize = ftell(check);
            fclose(check);
            if (fsize > 1024*1024) {
                printf("Model already exists: %s (%.2f GB)\n", pull_path, 
                       fsize / (1024.0*1024*1024));
                model_path = pull_path;
            }
        }
        
        if (!model_path) {
            printf("\nDownloading model from %s...\n\n", pull_url);
            if (model_fetch(pull_url, pull_path)) {
                model_path = pull_path;
            } else {
                fprintf(stderr, "Failed to download model\n");
                return 1;
            }
        }
    }
    
    if (!model_path) {
        static char auto_path[512];
#ifdef _WIN32
        /* Check LocalState (UWP app data) */
        const char* local_appdata = getenv("LOCALAPPDATA");
        const char* scan_dirs[] = {
            ".",
            "models",
            "D:\\DevelopmentFiles",
            "D:\\DevelopmentFiles\\models",
            "D:\\DevelopmentFiles\\WdpTempWebFolder",
            "C:\\Users\\ali\\models",
            /* Edge download locations (Xbox user profiles) */
            "Q:\\Users\\UserMgr0\\Downloads",
            "Q:\\Users\\DefaultAccount\\Downloads",
            "Q:\\Users\\UserMgr0\\Desktop",
            /* Edge app container download paths */
            "Q:\\Users\\UserMgr0\\AppData\\Local\\Packages\\Microsoft.MicrosoftEdge.Stable_8wekyb3d8bbwe\\LocalCache\\Local\\Microsoft\\Edge\\User Data\\Default\\Downloads",
            NULL
        };
        
        /* Recursive scan helper: check dir and one level of subdirs */
        #define SCAN_GGUF(dir) do { \
            WIN32_FIND_DATAA _fd; \
            char _search[512]; \
            snprintf(_search, sizeof(_search), "%s\\*.gguf", dir); \
            HANDLE _h = FindFirstFileA(_search, &_fd); \
            if (_h != INVALID_HANDLE_VALUE) { \
                snprintf(auto_path, sizeof(auto_path), "%s\\%s", dir, _fd.cFileName); \
                model_path = auto_path; \
                FindClose(_h); \
            } \
        } while(0)
        
        /* Try LocalState first (Xbox UWP) */
        if (local_appdata) {
            WIN32_FIND_DATAA fd;
            char search[512];
            snprintf(search, sizeof(search), "%s\\..\\LocalState\\*.gguf", local_appdata);
            HANDLE h = FindFirstFileA(search, &fd);
            if (h != INVALID_HANDLE_VALUE) {
                snprintf(auto_path, sizeof(auto_path), "%s\\..\\LocalState\\%s", 
                         local_appdata, fd.cFileName);
                model_path = auto_path;
                FindClose(h);
            }
        }
        
        /* Then try each scan dir */
        if (!model_path) {
            for (int d = 0; scan_dirs[d] && !model_path; d++) {
                SCAN_GGUF(scan_dirs[d]);
            }
        }
        
        /* Deep scan WdpTempWebFolder subdirs (Device Portal uploads land here) */
        if (!model_path) {
            WIN32_FIND_DATAA subdir;
            HANDLE hdir = FindFirstFileA("D:\\DevelopmentFiles\\WdpTempWebFolder\\*", &subdir);
            if (hdir != INVALID_HANDLE_VALUE) {
                do {
                    if ((subdir.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && 
                        subdir.cFileName[0] != '.') {
                        char subpath[512];
                        snprintf(subpath, sizeof(subpath), 
                                 "D:\\DevelopmentFiles\\WdpTempWebFolder\\%s", subdir.cFileName);
                        SCAN_GGUF(subpath);
                        if (model_path) break;
                    }
                } while (FindNextFileA(hdir, &subdir));
                FindClose(hdir);
            }
        }
#else
        /* Linux: scan current dir and models/ */
        const char* scan_dirs[] = {".", "models", NULL};
        /* Simple approach: just check if models/*.gguf exists */
        FILE* fp;
        for (int d = 0; scan_dirs[d] && !model_path; d++) {
            char cmd[256];
            snprintf(cmd, sizeof(cmd), "ls %s/*.gguf 2>/dev/null | head -1", scan_dirs[d]);
            fp = popen(cmd, "r");
            if (fp) {
                if (fgets(auto_path, sizeof(auto_path), fp)) {
                    auto_path[strcspn(auto_path, "\n")] = 0;
                    if (auto_path[0]) model_path = auto_path;
                }
                pclose(fp);
            }
        }
#endif
        if (!model_path) {
            /* No model found locally — try auto-pull from LAN */
            const char* lan_url = pull_url ? pull_url : 
                "http://192.168.1.8:9090/Qwen3.5-9B-Q4_K_M.gguf";
            const char* fname = model_fetch_filename(lan_url);
            static char fetch_path[512];
#ifdef _WIN32
            const char* ad = getenv("LOCALAPPDATA");
            if (ad) {
                /* UWP LocalState — writable by the app */
                snprintf(fetch_path, sizeof(fetch_path), "%s\\..\\LocalState\\%s", ad, fname);
            } else {
                snprintf(fetch_path, sizeof(fetch_path), "models\\%s", fname);
            }
#else
            snprintf(fetch_path, sizeof(fetch_path), "models/%s", fname);
#endif
            printf("\nNo local model found. Fetching from LAN...\n");
            printf("URL: %s\n", lan_url);
            printf("Destination: %s\n\n", fetch_path);
            
            if (model_fetch(lan_url, fetch_path)) {
                model_path = fetch_path;
            } else {
                fprintf(stderr, "\nError: No model available.\n");
                fprintf(stderr, "Provide --model <path.gguf> or --pull <url>\n");
                fprintf(stderr, "Or start HTTP server on Victus: python3 -m http.server 9090\n\n");
                return 1;
            }
        }
        printf("Auto-detected model: %s\n", model_path);
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

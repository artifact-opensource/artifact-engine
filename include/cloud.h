/*
 * Artifact Engine — Cloud Model Integration
 * 
 * Provides cloud model access through multiple providers:
 * - HuggingFace Inference API
 * - OpenRouter
 * - Google AI Studio (Gemini)
 * - Anthropic (Claude)
 * - OpenAI (GPT)
 * - Azure OpenAI
 * - AWS Bedrock
 * - Ollama
 * - LM Studio
 */

#ifndef CLOUD_H
#define CLOUD_H

#include <stdbool.h>
#include <stdint.h>

/* Cloud provider types */
typedef enum {
    CLOUD_PROVIDER_NONE = 0,
    CLOUD_PROVIDER_HUGGINGFACE,
    CLOUD_PROVIDER_OPENROUTER,
    CLOUD_PROVIDER_GOOGLE_AI_STUDIO,
    CLOUD_PROVIDER_ANTHROPIC,
    CLOUD_PROVIDER_OPENAI,
    CLOUD_PROVIDER_AZURE_OPENAI,
    CLOUD_PROVIDER_AWS_BEDROCK,
    CLOUD_PROVIDER_OLLAMA,
    CLOUD_PROVIDER_LM_STUDIO
} cloud_provider;

/* Cloud model configuration */
typedef struct {
    char* name;              /* Model name */
    char* id;                /* Provider model ID */
    cloud_provider provider; /* Provider type */
    bool available;          /* Is provider configured? */
    char* endpoint;          /* API endpoint URL */
    char* api_key;           /* API key (from env) */
    uint32_t max_tokens;     /* Max tokens supported */
    bool supports_vision;    /* Vision capabilities */
    bool supports_tools;     /* Tool use capabilities */
} cloud_model_config;

/* Cloud inference request */
typedef struct {
    char* model_id;          /* Model to use */
    char* prompt;            /* User prompt */
    float temperature;       /* Sampling temperature */
    float top_p;            /* Nucleus sampling */
    uint32_t max_tokens;     /* Max tokens to generate */
    bool stream;            /* Stream response? */
    char* system_prompt;    /* Optional system prompt */
} cloud_request;

/* Cloud inference response */
typedef struct {
    char* text;              /* Generated text */
    uint32_t prompt_tokens;  /* Tokens in prompt */
    uint32_t completion_tokens; /* Tokens generated */
    uint32_t total_tokens;   /* Total tokens */
    bool success;            /* Success flag */
    char* error_message;     /* Error if failed */
} cloud_response;

/* Tool definitions for computer_use */
typedef struct {
    char* type;              /* "computer", "browse", "python", etc. */
    char* name;              /* Tool name */
    char* description;       /* Tool description */
    char* parameters;        /* JSON schema for parameters */
} cloud_tool;

/* Tool call */
typedef struct {
    char* name;              /* Tool name */
    char* arguments;         /* JSON arguments */
} tool_call;

/* Computer use tool configuration */
typedef struct {
    char* display;           /* Display to control (e.g., "screenshot", "terminal") */
    char* actions;           /* Available actions */
    bool enabled;            /* Is computer use enabled? */
} computer_tool_config;

/* Initialize cloud provider from environment variables */
bool cloud_init_from_env(void);

/* List available cloud models */
uint32_t cloud_list_models(cloud_model_config** models);

/* Get model by ID */
cloud_model_config* cloud_get_model(const char* model_id);

/* Check if provider is configured */
bool cloud_is_provider_available(cloud_provider provider);

/* Generate text using cloud model */
cloud_response* cloud_generate(const cloud_request* req);

/* Generate with tool support */
cloud_response* cloud_generate_with_tools(
    const cloud_request* req,
    cloud_tool* tools,
    uint32_t n_tools
);

/* Handle tool call */
cloud_response* cloud_handle_tool_call(const tool_call* call);

/* Health check for cloud provider */
bool cloud_health_check(cloud_provider provider);

/* Free cloud response */
void cloud_free_response(cloud_response* resp);

/* Computer use tool functions */
bool computer_init(void);
void computer_cleanup(void);
cloud_response* computer_take_screenshot(void);
cloud_response* computer_type_text(const char* text);
cloud_response* computer_click(int x, int y);
cloud_response* computer_scroll(int direction);
cloud_response* computer_wait(uint32_t milliseconds);

/* Environment variable names */
#define ENV_HUGGINGFACE_API_KEY    "HUGGING_FACE_API_KEY"
#define ENV_OPENROUTER_API_KEY     "OPENROUTER_API_KEY"
#define ENV_GOOGLE_AI_API_KEY      "GOOGLE_AI_STUDIO_API_KEY"
#define ENV_ANTHROPIC_API_KEY      "ANTHROPIC_API_KEY"
#define ENV_OPENAI_API_KEY         "OPENAI_API_KEY"
#define ENV_AZURE_OPENAI_KEY       "AZURE_OPENAI_API_KEY"
#define ENV_AZURE_OPENAI_ENDPOINT  "AZURE_OPENAI_ENDPOINT"
#define ENV_AWS_ACCESS_KEY         "AWS_ACCESS_KEY_ID"
#define ENV_AWS_SECRET_KEY         "AWS_SECRET_ACCESS_KEY"
#define ENV_AWS_REGION             "AWS_REGION"
#define ENV_OLLAMA_ENDPOINT        "OLLAMA_API_BASE"
#define ENV_LM_STUDIO_ENDPOINT     "LM_STUDIO_API_BASE"

/* Default timeout for cloud requests (seconds) */
#define CLOUD_DEFAULT_TIMEOUT 300

/* Max response size */
#define CLOUD_MAX_RESPONSE_SIZE (1024 * 1024)  /* 1MB */

#endif /* CLOUD_H */
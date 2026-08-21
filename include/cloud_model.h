/*
 * Artifact Engine — Cloud Model Integration
 * Enables access to cloud AI models alongside local GGUF models
 */

#ifndef CLOUD_MODEL_H
#define CLOUD_MODEL_H

#include <stdbool.h>
#include <stdint.h>

/* ───── Cloud Provider Types ───── */
typedef enum {
    CLOUD_PROVIDER_NONE = 0,
    CLOUD_PROVIDER_OPENAI,
    CLOUD_PROVIDER_ANTHROPIC,
    CLOUD_PROVIDER_GOOGLE,
    CLOUD_PROVIDER_AZURE,
    CLOUD_PROVIDER_AWS_BEDROCK,
    CLOUD_PROVIDER_HUGGINGFACE,
    CLOUD_PROVIDER_OPENROUTER,
    CLOUD_PROVIDER_OLLAMA,
    CLOUD_PROVIDER_LM_STUDIO
} cloud_provider;

/* ───── Cloud Model Config ───── */
typedef struct {
    char* api_key;
    char* api_base;
    char* model_name;
    cloud_provider provider;
    bool enabled;
} cloud_model_config;

/* ───── Chat Message ───── */
typedef struct {
    char* role;     /* "system", "user", "assistant" */
    char* content;
} chat_message;

/* ───── Cloud Chat Request ───── */
typedef struct {
    char*          model;
    chat_message*  messages;
    uint32_t       n_messages;
    float          temperature;
    float          top_p;
    uint32_t       max_tokens;
    bool           stream;
} cloud_chat_request;

/* ───── Cloud Chat Response ───── */
typedef struct {
    char* content;
    char* model_id;
    int   usage_prompt_tokens;
    int   usage_completion_tokens;
    int   usage_total_tokens;
} cloud_chat_response;

/* ───── Cloud Model Registry ───── */
typedef struct {
    cloud_model_config* configs;
    uint32_t n_configs;
    char* default_provider;
    bool enable_fallback;
} cloud_model_registry;

/* ───── API ───── */

/* Initialize cloud model registry */
bool cloud_model_init(cloud_model_registry* registry);

/* Check if model is a cloud model */
bool cloud_model_is_cloud(const char* model_name);

/* Get cloud provider from model name */
cloud_provider cloud_model_get_provider(const char* model_name);

/* Generate chat completion using cloud model */
bool cloud_model_chat(
    const cloud_model_registry* registry,
    const cloud_chat_request* request,
    cloud_chat_response* response
);

/* Add cloud model to registry */
bool cloud_model_add(
    cloud_model_registry* registry,
    const char* api_key,
    const char* api_base,
    const char* model_name,
    cloud_provider provider
);

/* Remove cloud model from registry */
bool cloud_model_remove(cloud_model_registry* registry, const char* model_name);

/* List all cloud models */
bool cloud_model_list(const cloud_model_registry* registry, char*** models, uint32_t* n_models);

/* Health check for cloud provider */
bool cloud_model_health_check(const cloud_model_registry* registry, const char* model_name);

/* Cleanup cloud model resources */
void cloud_model_cleanup(cloud_model_registry* registry);

#endif /* CLOUD_MODEL_H */
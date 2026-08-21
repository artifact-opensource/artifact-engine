/**
 * Cloud Model Integration Header
 * 
 * Provides unified access to cloud-based AI models from multiple providers
 * including HuggingFace, OpenRouter, Google AI Studio, Anthropic, OpenAI,
 * Azure OpenAI, AWS Bedrock, Ollama, and LM Studio.
 */

#ifndef CLOUD_INTEGRATION_H
#define CLOUD_INTEGRATION_H

#include <stdbool.h>
#include <stdint.h>

// Cloud provider types
typedef enum {
    PROVIDER_OLLAMA,
    PROVIDER_HUGGING_FACE,
    PROVIDER_OPENROUTER,
    PROVIDER_GOOGLE_AI_STUDIO,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_AWS_BEDROCK,
    PROVIDER_LM_STUDIO
} cloud_provider_t;

// Model configuration
typedef struct {
    char* provider_name;
    char* model_name;
    char* api_key;
    char* base_url;
    int timeout_ms;
    int max_tokens;
    float temperature;
    bool enable_caching;
    int cache_ttl_seconds;
} cloud_model_config_t;

// Generation result
typedef struct {
    char* response;
    int tokens_used;
    char* model_used;
    char* provider_used;
    int64_t timestamp;
    bool success;
    char* error_message;
} cloud_generation_result_t;

// Chat message
typedef struct {
    char* role;  // "system", "user", "assistant"
    char* content;
} chat_message_t;

// Chat completion result
typedef struct {
    char* response;
    int tokens_used;
    char* model_used;
    char* provider_used;
    int64_t timestamp;
    bool success;
    char* error_message;
    char** usage_details;  // JSON string
} cloud_chat_result_t;

// Vector database integration
typedef struct {
    char* db_path;
    int max_vectors;
    int vector_dimensions;
    bool enable_compression;
} vector_db_config_t;

// Computer use tool configuration
typedef struct {
    bool enable_desktop_control;
    bool enable_file_operations;
    bool enable_network_access;
    char* allowed_directories;
    char* blocked_commands;
    int execution_timeout_ms;
} computer_use_config_t;

// Agentic capabilities configuration
typedef struct {
    bool enable_reasoning_chains;
    bool enable_tool_chaining;
    bool enable_memory_retention;
    int max_conversation_turns;
    bool enable_self_critique;
    bool enable_planning;
} agentic_config_t;

// Cloud integration context
typedef struct {
    cloud_model_config_t* models;
    int model_count;
    vector_db_config_t* vector_db;
    computer_use_config_t* computer_use;
    agentic_config_t* agentic;
    bool initialized;
} cloud_integration_context_t;

// Function prototypes

/**
 * Initialize cloud integration with configuration
 */
int cloud_init(cloud_integration_context_t* ctx, const char* config_json);

/**
 * Cleanup cloud integration resources
 */
void cloud_cleanup(cloud_integration_context_t* ctx);

/**
 * List available models from a specific provider
 */
int cloud_list_models(cloud_provider_t provider, char*** models, int* count);

/**
 * Generate text using a cloud model
 */
cloud_generation_result_t* cloud_generate(
    cloud_integration_context_t* ctx,
    const char* model_id,
    const char* prompt,
    const char* options_json
);

/**
 * Chat completion using a cloud model
 */
cloud_chat_result_t* cloud_chat(
    cloud_integration_context_t* ctx,
    const char* model_id,
    chat_message_t* messages,
    int message_count,
    const char* options_json
);

/**
 * Check health of cloud providers
 */
bool cloud_health_check(cloud_integration_context_t* ctx, cloud_provider_t provider);

/**
 * Get model details
 */
cloud_model_config_t* cloud_get_model_config(cloud_integration_context_t* ctx, const char* model_id);

/**
 * Set timeout for a specific model
 */
void cloud_set_timeout(const char* model_id, int timeout_ms);

/**
 * Enable/disable caching for a model
 */
void cloud_set_caching(const char* model_id, bool enabled);

/**
 * Get provider status
 */
bool cloud_is_provider_available(cloud_integration_context_t* ctx, cloud_provider_t provider);

/**
 * Get default provider
 */
cloud_provider_t cloud_get_default_provider(cloud_integration_context_t* ctx);

/**
 * Set default provider
 */
void cloud_set_default_provider(cloud_integration_context_t* ctx, cloud_provider_t provider);

/**
 * Vector database operations
 */
int vdb_store_vector(const char* collection, const char* id, float* vector, int dimensions);
int vdb_search_vectors(const char* collection, float* query_vector, int dimensions, int top_k, char*** ids, float** scores);
int vdb_delete_vector(const char* collection, const char* id);
int vdb_clear_collection(const char* collection);

/**
 * Computer use tool operations
 */
int computer_use_execute(const char* command, char** output, int* exit_code);
int computer_use_file_read(const char* filepath, char** content);
int computer_use_file_write(const char* filepath, const char* content);
int computer_use_network_request(const char* url, const char* method, const char* data, char** response);

/**
 * Agentic reasoning operations
 */
char* agent_reason(const char* context, const char* goal, const char* constraints);
char* agent_plan(const char* goal, const char* available_tools);
char* agent_critique(const char* output, const char* criteria);

#endif // CLOUD_INTEGRATION_H
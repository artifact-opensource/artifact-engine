/*
 * Artifact Engine — Cloud Model Implementation
 * Provides cloud model access through HTTP API calls
 */

#include "cloud_model.h"
#include "http_client.h"
#include "json_parser.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Default API endpoints */
static const char* OPENAI_ENDPOINTS[] = {
    [CLOUD_PROVIDER_OPENAI] = "https://api.openai.com/v1/chat/completions",
    [CLOUD_PROVIDER_ANTHROPIC] = "https://api.anthropic.com/v1/messages",
    [CLOUD_PROVIDER_GOOGLE] = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    [CLOUD_PROVIDER_AZURE] = NULL, /* Configured per instance */
    [CLOUD_PROVIDER_AWS_BEDROCK] = NULL, /* Configured per instance */
    [CLOUD_PROVIDER_HUGGINGFACE] = "https://api-inference.huggingface.co/models",
    [CLOUD_PROVIDER_OPENROUTER] = "https://openrouter.ai/api/v1/chat/completions",
    [CLOUD_PROVIDER_OLLAMA] = "/api/chat",
    [CLOUD_PROVIDER_LM_STUDIO] = "/v1/chat/completions"
};

/* Initialize cloud model registry */
bool cloud_model_init(cloud_model_registry* registry) {
    if (!registry) return false;
    
    registry->configs = NULL;
    registry->n_configs = 0;
    registry->default_provider = strdup("openai");
    registry->enable_fallback = true;
    
    return true;
}

/* Check if model is a cloud model */
bool cloud_model_is_cloud(const char* model_name) {
    if (!model_name) return false;
    
    /* Cloud models have provider prefix */
    if (strstr(model_name, "openrouter:") == model_name) return true;
    if (strstr(model_name, "openai:") == model_name) return true;
    if (strstr(model_name, "anthropic:") == model_name) return true;
    if (strstr(model_name, "google:") == model_name) return true;
    if (strstr(model_name, "azure:") == model_name) return true;
    if (strstr(model_name, "aws:") == model_name) return true;
    if (strstr(model_name, "huggingface:") == model_name) return true;
    if (strstr(model_name, "ollama:") == model_name) return true;
    if (strstr(model_name, "lmstudio:") == model_name) return true;
    
    return false;
}

/* Get cloud provider from model name */
cloud_provider cloud_model_get_provider(const char* model_name) {
    if (!model_name) return CLOUD_PROVIDER_NONE;
    
    if (strstr(model_name, "openrouter:") == model_name) return CLOUD_PROVIDER_OPENROUTER;
    if (strstr(model_name, "openai:") == model_name) return CLOUD_PROVIDER_OPENAI;
    if (strstr(model_name, "anthropic:") == model_name) return CLOUD_PROVIDER_ANTHROPIC;
    if (strstr(model_name, "google:") == model_name) return CLOUD_PROVIDER_GOOGLE;
    if (strstr(model_name, "azure:") == model_name) return CLOUD_PROVIDER_AZURE;
    if (strstr(model_name, "aws:") == model_name) return CLOUD_PROVIDER_AWS_BEDROCK;
    if (strstr(model_name, "huggingface:") == model_name) return CLOUD_PROVIDER_HUGGINGFACE;
    if (strstr(model_name, "ollama:") == model_name) return CLOUD_PROVIDER_OLLAMA;
    if (strstr(model_name, "lmstudio:") == model_name) return CLOUD_PROVIDER_LM_STUDIO;
    
    return CLOUD_PROVIDER_NONE;
}

/* Add cloud model to registry */
bool cloud_model_add(
    cloud_model_registry* registry,
    const char* api_key,
    const char* api_base,
    const char* model_name,
    cloud_provider provider
) {
    if (!registry || !model_name) return false;
    
    /* Expand registry */
    cloud_model_config* new_configs = realloc(registry->configs, 
        (registry->n_configs + 1) * sizeof(cloud_model_config));
    if (!new_configs) return false;
    
    registry->configs = new_configs;
    
    /* Add new config */
    cloud_model_config* config = &registry->configs[registry->n_configs];
    config->api_key = strdup(api_key ? api_key : "");
    config->api_base = strdup(api_base ? api_base : "");
    config->model_name = strdup(model_name);
    config->provider = provider;
    config->enabled = true;
    
    registry->n_configs++;
    
    return true;
}

/* Remove cloud model from registry */
bool cloud_model_remove(cloud_model_registry* registry, const char* model_name) {
    if (!registry || !model_name) return false;
    
    for (uint32_t i = 0; i < registry->n_configs; i++) {
        if (strcmp(registry->configs[i].model_name, model_name) == 0) {
            free(registry->configs[i].api_key);
            free(registry->configs[i].api_base);
            free(registry->configs[i].model_name);
            
            /* Shift remaining configs */
            for (uint32_t j = i; j < registry->n_configs - 1; j++) {
                registry->configs[j] = registry->configs[j + 1];
            }
            
            registry->n_configs--;
            return true;
        }
    }
    
    return false;
}

/* List all cloud models */
bool cloud_model_list(const cloud_model_registry* registry, char*** models, uint32_t* n_models) {
    if (!registry || !models || !n_models) return false;
    
    *models = malloc(registry->n_configs * sizeof(char*));
    if (!*models) return false;
    
    for (uint32_t i = 0; i < registry->n_configs; i++) {
        (*models)[i] = strdup(registry->configs[i].model_name);
    }
    
    *n_models = registry->n_configs;
    return true;
}

/* Generate chat completion using cloud model */
bool cloud_model_chat(
    const cloud_model_registry* registry,
    const cloud_chat_request* request,
    cloud_chat_response* response
) {
    if (!registry || !request || !response) return false;
    
    /* Find matching config */
    cloud_model_config* config = NULL;
    for (uint32_t i = 0; i < registry->n_configs; i++) {
        if (strcmp(registry->configs[i].model_name, request->model) == 0) {
            config = (cloud_model_config*)&registry->configs[i];
            break;
        }
    }
    
    if (!config || !config->enabled) return false;
    
    /* Build JSON request */
    char* json_request = NULL;
    char* endpoint = NULL;
    
    /* Get endpoint based on provider */
    if (config->provider == CLOUD_PROVIDER_AZURE && config->api_base) {
        endpoint = strdup(config->api_base);
    } else if (OPENAI_ENDPOINTS[config->provider]) {
        endpoint = strdup(OPENAI_ENDPOINTS[config->provider]);
    } else {
        endpoint = config->api_base;
    }
    
    /* Build request body based on provider */
    switch (config->provider) {
        case CLOUD_PROVIDER_ANTHROPIC:
            /* Build Anthropic format */
            json_request = json_build_anthropic_request(request);
            break;
            
        case CLOUD_PROVIDER_GOOGLE:
            /* Build Google format */
            json_request = json_build_google_request(request);
            break;
            
        case CLOUD_PROVIDER_OLLAMA:
        case CLOUD_PROVIDER_LM_STUDIO:
            /* Build OpenAI-compatible format for Ollama/LM Studio */
            json_request = json_build_openai_request(request);
            break;
            
        default:
            /* Build OpenAI format */
            json_request = json_build_openai_request(request);
            break;
    }
    
    if (!json_request) {
        free(endpoint);
        return false;
    }
    
    /* Make HTTP request */
    char* json_response = NULL;
    int http_status = 0;
    
    http_client_post(endpoint, config->api_key, json_request, &json_response, &http_status);
    
    /* Parse response */
    if (json_response && http_status == 200) {
        json_parse_cloud_response(json_response, response, config->provider);
    }
    
    /* Cleanup */
    free(json_request);
    free(endpoint);
    if (json_response) free(json_response);
    
    return (http_status == 200);
}

/* Health check for cloud provider */
bool cloud_model_health_check(const cloud_model_registry* registry, const char* model_name) {
    if (!registry || !model_name) return false;
    
    /* For cloud models, we check if API key is configured */
    for (uint32_t i = 0; i < registry->n_configs; i++) {
        if (strcmp(registry->configs[i].model_name, model_name) == 0) {
            return registry->configs[i].enabled && 
                   registry->configs[i].api_key && 
                   strlen(registry->configs[i].api_key) > 0;
        }
    }
    
    return false;
}

/* Cleanup cloud model resources */
void cloud_model_cleanup(cloud_model_registry* registry) {
    if (!registry) return;
    
    for (uint32_t i = 0; i < registry->n_configs; i++) {
        free(registry->configs[i].api_key);
        free(registry->configs[i].api_base);
        free(registry->configs[i].model_name);
    }
    
    free(registry->configs);
    free(registry->default_provider);
    
    registry->configs = NULL;
    registry->n_configs = 0;
}
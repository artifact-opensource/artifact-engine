/*
 * Artifact Engine — Computer Use Tool Implementation
 * Provides system interaction capabilities for AI models
 */

#include "computer_use.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

/* Default tool definitions */
static const tool_definition DEFAULT_TOOLS[] = {
    {
        .name = "computer",
        .description = "Use the computer to open, close, or interact with applications and files",
        .parameters = "{\"type\": \"object\", \"properties\": {\"action\": {\"enum\": [\"open\", \"close\", \"run\", \"execute\", \"screenshot\"], \"description\": \"Action to perform\"}, \"application\": {\"type\": \"string\", \"description\": \"Application to open/close\"}, \"path\": {\"type\": \"string\", \"description\": \"File path to open\"}, \"command\": {\"type\": \"string\", \"description\": \"Command to execute\"}}}, \"required\": [\"action\"]}",
        .strict = true
    },
    {
        .name = "browser",
        .description = "Interact with web browser to search, navigate, and interact with websites",
        .parameters = "{\"type\": \"object\", \"properties\": {\"action\": {\"enum\": [\"search\", \"navigate\", \"click\", \"type\", \"submit\"], \"description\": \"Browser action\"}, \"url\": {\"type\": \"string\", \"description\": \"URL to navigate to\"}, \"query\": {\"type\": \"string\", \"description\": \"Search query\"}}, \"required\": [\"action\"]}",
        .strict = true
    },
    {
        .name = "files",
        .description = "Read, write, and manipulate files on the system",
        .parameters = "{\"type\": \"object\", \"properties\": {\"action\": {\"enum\": [\"read\", \"write\", \"delete\", \"list\", \"create\", \"move\", \"copy\"], \"description\": \"File action\"}, \"path\": {\"type\": \"string\", \"description\": \"File path\"}, \"content\": {\"type\": \"string\", \"description\": \"Content to write\"}, \"destination\": {\"type\": \"string\", \"description\": \"Destination path\"}}, \"required\": [\"action\", \"path\"]}",
        .strict = true
    },
    {
        .name = "network",
        .description = "Make HTTP requests and interact with network resources",
        .parameters = "{\"type\": \"object\", \"properties\": {\"action\": {\"enum\": [\"get\", \"post\", \"put\", \"delete\"], \"description\": \"HTTP method\"}, \"url\": {\"type\": \"string\", \"description\": \"URL to request\"}, \"headers\": {\"type\": \"object\", \"description\": \"Request headers\"}, \"body\": {\"type\": \"string\", \"description\": \"Request body\"}}, \"required\": [\"action\", \"url\"]}",
        .strict = true
    },
    {
        .name = "system",
        .description = "Execute system commands and interact with the operating system",
        .parameters = "{\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\", \"description\": \"System command to execute\"}, \"shell\": {\"type\": \"boolean\", \"default\": true, \"description\": \"Execute in shell\"}}, \"required\": [\"command\"]}",
        .strict = true
    }
};

/* Initialize tool registry */
bool tool_init(tool_registry* registry) {
    if (!registry) return false;
    
    registry->tools = NULL;
    registry->n_tools = 0;
    registry->context = calloc(1, sizeof(computer_use_context));
    
    if (!registry->context) return false;
    
    /* Set default context */
    registry->context->working_directory = strdup(getenv("PWD") ? getenv("PWD") : "/tmp");
    registry->context->allow_network = true;
    registry->context->allow_file_access = true;
    registry->context->allow_screen_capture = true;
    registry->context->timeout_seconds = 30;
    
    return true;
}

/* Register computer use tool */
bool tool_register_computer(tool_registry* registry) {
    if (!registry) return false;
    
    tool_definition* new_tools = realloc(registry->tools, 
        (registry->n_tools + 1) * sizeof(tool_definition));
    if (!new_tools) return false;
    
    registry->tools = new_tools;
    registry->tools[registry->n_tools] = DEFAULT_TOOLS[0];
    registry->n_tools++;
    
    return true;
}

/* Register browser tool */
bool tool_register_browser(tool_registry* registry) {
    if (!registry) return false;
    
    tool_definition* new_tools = realloc(registry->tools, 
        (registry->n_tools + 1) * sizeof(tool_definition));
    if (!new_tools) return false;
    
    registry->tools = new_tools;
    registry->tools[registry->n_tools] = DEFAULT_TOOLS[1];
    registry->n_tools++;
    
    return true;
}

/* Register file system tool */
bool tool_register_files(tool_registry* registry) {
    if (!registry) return false;
    
    tool_definition* new_tools = realloc(registry->tools, 
        (registry->n_tools + 1) * sizeof(tool_definition));
    if (!new_tools) return false;
    
    registry->tools = new_tools;
    registry->tools[registry->n_tools] = DEFAULT_TOOLS[2];
    registry->n_tools++;
    
    return true;
}

/* Register network tool */
bool tool_register_network(tool_registry* registry) {
    if (!registry) return false;
    
    tool_definition* new_tools = realloc(registry->tools, 
        (registry->n_tools + 1) * sizeof(tool_definition));
    if (!new_tools) return false;
    
    registry->tools = new_tools;
    registry->tools[registry->n_tools] = DEFAULT_TOOLS[3];
    registry->n_tools++;
    
    return true;
}

/* Execute tool call */
bool tool_execute(const tool_registry* registry, const tool_call* call, tool_response* response) {
    if (!registry || !call || !response) return false;
    
    /* Find matching tool */
    for (uint32_t i = 0; i < registry->n_tools; i++) {
        if (strcmp(registry->tools[i].name, call->tool_name) == 0) {
            /* Route to appropriate handler */
            if (strcmp(call->tool_name, "computer") == 0) {
                return tool_execute_computer(registry->context, call, response);
            } else if (strcmp(call->tool_name, "browser") == 0) {
                return tool_execute_browser(registry->context, call, response);
            } else if (strcmp(call->tool_name, "files") == 0) {
                return tool_execute_files(registry->context, call, response);
            } else if (strcmp(call->tool_name, "network") == 0) {
                return tool_execute_network(registry->context, call, response);
            } else if (strcmp(call->tool_name, "system") == 0) {
                return tool_execute_computer(registry->context, call, response);
            }
        }
    }
    
    response->success = false;
    response->error = strdup("Tool not found");
    return false;
}

/* Execute computer action */
bool tool_execute_computer(const computer_use_context* ctx, const tool_call* call, tool_response* response) {
    if (!ctx || !call || !response) return false;
    
    response->success = false;
    response->content = NULL;
    response->error = NULL;
    response->exit_code = 0;
    
    /* Parse action from arguments */
    const char* action = json_get_string(call->arguments, "action");
    if (!action) {
        response->error = strdup("Missing action parameter");
        return false;
    }
    
    /* Execute based on action */
    if (strcmp(action, "open") == 0) {
        const char* application = json_get_string(call->arguments, "application");
        if (!application) {
            response->error = strdup("Missing application parameter");
            return false;
        }
        
        /* Open application */
        pid_t pid = fork();
        if (pid == 0) {
            execl("/usr/bin/open", "open", application, NULL);
            exit(1);
        } else if (pid > 0) {
            waitpid(pid, &response->exit_code, 0);
            response->success = true;
            response->content = strdup("Application opened successfully");
        } else {
            response->error = strdup("Failed to fork process");
        }
    }
    else if (strcmp(action, "run") == 0 || strcmp(action, "execute") == 0) {
        const char* command = json_get_string(call->arguments, "command");
        if (!command) {
            response->error = strdup("Missing command parameter");
            return false;
        }
        
        /* Execute command */
        FILE* fp = popen(command, "r");
        if (fp) {
            char buffer[4096];
            char* output = malloc(1);
            output[0] = '\0';
            
            while (fgets(buffer, sizeof(buffer), fp)) {
                char* new_output = realloc(output, strlen(output) + strlen(buffer) + 1);
                if (!new_output) {
                    free(output);
                    pclose(fp);
                    response->error = strdup("Memory allocation failed");
                    return false;
                }
                output = new_output;
                strcat(output, buffer);
            }
            
            response->exit_code = pclose(fp);
            response->success = true;
            response->content = output;
        } else {
            response->error = strdup("Failed to execute command");
        }
    }
    else if (strcmp(action, "screenshot") == 0) {
        /* Take screenshot */
        const char* output_path = json_get_string(call->arguments, "path");
        if (!output_path) output_path = "/tmp/screenshot.png";
        
        char command[512];
        snprintf(command, sizeof(command), "screencapture -x %s", output_path);
        
        int result = system(command);
        if (result == 0) {
            response->success = true;
            response->content = strdup(output_path);
        } else {
            response->error = strdup("Failed to take screenshot");
            response->exit_code = result;
        }
    }
    else {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Unknown action: %s", action);
        response->error = strdup(error_msg);
    }
    
    return response->success;
}

/* Execute browser action */
bool tool_execute_browser(const computer_use_context* ctx, const tool_call* call, tool_response* response) {
    if (!ctx || !call || !response) return false;
    
    response->success = false;
    response->content = NULL;
    response->error = NULL;
    response->exit_code = 0;
    
    /* Parse action from arguments */
    const char* action = json_get_string(call->arguments, "action");
    if (!action) {
        response->error = strdup("Missing action parameter");
        return false;
    }
    
    /* Execute browser action */
    if (strcmp(action, "search") == 0) {
        const char* query = json_get_string(call->arguments, "query");
        if (!query) {
            response->error = strdup("Missing query parameter");
            return false;
        }
        
        /* Open browser with search */
        char command[1024];
        snprintf(command, sizeof(command), "open \"https://www.google.com/search?q=%s\"", query);
        
        int result = system(command);
        response->success = (result == 0);
        response->exit_code = result;
        if (response->success) {
            response->content = strdup("Browser opened with search results");
        } else {
            response->error = strdup("Failed to open browser");
        }
    }
    else if (strcmp(action, "navigate") == 0) {
        const char* url = json_get_string(call->arguments, "url");
        if (!url) {
            response->error = strdup("Missing URL parameter");
            return false;
        }
        
        /* Open browser with URL */
        char command[1024];
        snprintf(command, sizeof(command), "open \"%s\"", url);
        
        int result = system(command);
        response->success = (result == 0);
        response->exit_code = result;
        if (response->success) {
            response->content = strdup("Browser opened with URL");
        } else {
            response->error = strdup("Failed to open browser");
        }
    }
    else {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Browser action not implemented: %s", action);
        response->error = strdup(error_msg);
    }
    
    return response->success;
}

/* Execute file system action */
bool tool_execute_files(const computer_use_context* ctx, const tool_call* call, tool_response* response) {
    if (!ctx || !call || !response) return false;
    
    response->success = false;
    response->content = NULL;
    response->error = NULL;
    response->exit_code = 0;
    
    /* Parse action from arguments */
    const char* action = json_get_string(call->arguments, "action");
    const char* path = json_get_string(call->arguments, "path");
    
    if (!action || !path) {
        response->error = strdup("Missing action or path parameter");
        return false;
    }
    
    /* Execute file action */
    if (strcmp(action, "read") == 0) {
        FILE* fp = fopen(path, "r");
        if (fp) {
            char buffer[8192];
            char* content = malloc(1);
            content[0] = '\0';
            
            size_t bytes;
            while ((bytes = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
                char* new_content = realloc(content, strlen(content) + bytes + 1);
                if (!new_content) {
                    free(content);
                    fclose(fp);
                    response->error = strdup("Memory allocation failed");
                    return false;
                }
                content = new_content;
                fwrite(buffer, 1, bytes, fp);
                content[strlen(content)] = '\0';
            }
            
            fclose(fp);
            response->success = true;
            response->content = content;
        } else {
            response->error = strdup("Failed to open file");
        }
    }
    else if (strcmp(action, "write") == 0) {
        const char* file_content = json_get_string(call->arguments, "content");
        if (!file_content) {
            response->error = strdup("Missing content parameter");
            return false;
        }
        
        FILE* fp = fopen(path, "w");
        if (fp) {
            fprintf(fp, "%s", file_content);
            fclose(fp);
            response->success = true;
            response->content = strdup("File written successfully");
        } else {
            response->error = strdup("Failed to open file for writing");
        }
    }
    else if (strcmp(action, "list") == 0) {
        /* List directory contents */
        char command[1024];
        snprintf(command, sizeof(command), "ls -la %s", path);
        
        FILE* fp = popen(command, "r");
        if (fp) {
            char buffer[4096];
            char* output = malloc(1);
            output[0] = '\0';
            
            while (fgets(buffer, sizeof(buffer), fp)) {
                char* new_output = realloc(output, strlen(output) + strlen(buffer) + 1);
                if (!new_output) {
                    free(output);
                    pclose(fp);
                    response->error = strdup("Memory allocation failed");
                    return false;
                }
                output = new_output;
                strcat(output, buffer);
            }
            
            pclose(fp);
            response->success = true;
            response->content = output;
        } else {
            response->error = strdup("Failed to list directory");
        }
    }
    else {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "File action not implemented: %s", action);
        response->error = strdup(error_msg);
    }
    
    return response->success;
}

/* Execute network action */
bool tool_execute_network(const computer_use_context* ctx, const tool_call* call, tool_response* response) {
    if (!ctx || !call || !response) return false;
    
    response->success = false;
    response->content = NULL;
    response->error = NULL;
    response->exit_code = 0;
    
    /* Check if network access is allowed */
    if (!ctx->allow_network) {
        response->error = strdup("Network access not allowed");
        return false;
    }
    
    /* Parse action from arguments */
    const char* action = json_get_string(call->arguments, "action");
    const char* url = json_get_string(call->arguments, "url");
    
    if (!action || !url) {
        response->error = strdup("Missing action or URL parameter");
        return false;
    }
    
    /* Execute network action using curl */
    char command[2048];
    if (strcmp(action, "get") == 0) {
        snprintf(command, sizeof(command), "curl -s \"%s\"", url);
    } else if (strcmp(action, "post") == 0) {
        const char* body = json_get_string(call->arguments, "body");
        snprintf(command, sizeof(command), "curl -s -X POST -d \"%s\" \"%s\"", body ? body : "", url);
    } else if (strcmp(action, "put") == 0) {
        const char* body = json_get_string(call->arguments, "body");
        snprintf(command, sizeof(command), "curl -s -X PUT -d \"%s\" \"%s\"", body ? body : "", url);
    } else if (strcmp(action, "delete") == 0) {
        snprintf(command, sizeof(command), "curl -s -X DELETE \"%s\"", url);
    } else {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Network action not implemented: %s", action);
        response->error = strdup(error_msg);
        return false;
    }
    
    /* Execute curl command */
    FILE* fp = popen(command, "r");
    if (fp) {
        char buffer[8192];
        char* output = malloc(1);
        output[0] = '\0';
        
        size_t bytes;
        while ((bytes = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
            char* new_output = realloc(output, strlen(output) + bytes + 1);
            if (!new_output) {
                free(output);
                pclose(fp);
                response->error = strdup("Memory allocation failed");
                return false;
            }
            output = new_output;
            fwrite(buffer, 1, bytes, stdout);
            output[strlen(output)] = '\0';
        }
        
        response->exit_code = pclose(fp);
        response->success = true;
        response->content = output;
    } else {
        response->error = strdup("Failed to execute network request");
    }
    
    return response->success;
}

/* Get available tools as JSON */
bool tool_get_definitions_json(const tool_registry* registry, char** json_output) {
    if (!registry || !json_output) return false;
    
    /* Build JSON array of tools */
    char* json = strdup("{\"type\": \"function\", \"function\": {\"name\": \"tool\", \"description\": \"Computer use tools\", \"parameters\": {\"type\": \"object\", \"properties\": {");
    
    for (uint32_t i = 0; i < registry->n_tools; i++) {
        char tool_json[2048];
        snprintf(tool_json, sizeof(tool_json), 
            "\"%s\": {\"type\": \"object\", \"properties\": %s},",
            registry->tools[i].name,
            registry->tools[i].parameters);
        
        char* new_json = realloc(json, strlen(json) + strlen(tool_json) + 1);
        if (!new_json) {
            free(json);
            return false;
        }
        json = new_json;
        strcat(json, tool_json);
    }
    
    strcat(json, "\"},\"required\":[\"name\",\"parameters\"]}}");
    *json_output = json;
    
    return true;
}

/* Cleanup tool resources */
void tool_cleanup(tool_registry* registry) {
    if (!registry) return;
    
    free(registry->tools);
    free(registry->context->working_directory);
    free(registry->context->environment);
    free(registry->context);
    
    registry->tools = NULL;
    registry->n_tools = 0;
    registry->context = NULL;
}
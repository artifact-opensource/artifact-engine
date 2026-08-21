/*
 * Artifact Engine — Computer Use Tool
 * Enables AI models to interact with the computer system
 */

#ifndef COMPUTER_USE_H
#define COMPUTER_USE_H

#include <stdbool.h>
#include <stdint.h>

/* ───── Tool Types ───── */
typedef enum {
    TOOL_TYPE_COMPUTER = 0,
    TOOL_TYPE_BROWSER,
    TOOL_TYPE_FILES,
    TOOL_TYPE_NETWORK,
    TOOL_TYPE_SYSTEM,
    TOOL_TYPE_SCREENSHOT,
    TOOL_TYPE_KEYBOARD,
    TOOL_TYPE_MOUSE
} tool_type;

/* ───── Computer Action ───── */
typedef enum {
    COMPUTER_ACTION_OPEN = 0,
    COMPUTER_ACTION_CLOSE,
    COMPUTER_ACTION_RUN,
    COMPUTER_ACTION_EXECUTE,
    COMPUTER_ACTION_SCREENSHOT
} computer_action;

/* ───── Tool Call Request ───── */
typedef struct {
    char* name;           /* Tool name */
    char* description;    /* Tool description */
    char* parameters;     /* JSON parameters */
    bool strict;          /* Strict parameter validation */
} tool_definition;

/* ───── Tool Call ───── */
typedef struct {
    char* tool_name;      /* Name of tool to call */
    char* arguments;      /* JSON arguments */
} tool_call;

/* ───── Tool Response ───── */
typedef struct {
    char* content;        /* Response content */
    char* error;          /* Error message if any */
    bool success;         /* Whether tool call succeeded */
    int exit_code;        /* Exit code if applicable */
} tool_response;

/* ───── Computer Use Context ───── */
typedef struct {
    char* working_directory;
    char* environment;
    bool allow_network;
    bool allow_file_access;
    bool allow_screen_capture;
    int timeout_seconds;
} computer_use_context;

/* ───── Tool Registry ───── */
typedef struct {
    tool_definition* tools;
    uint32_t n_tools;
    computer_use_context* context;
} tool_registry;

/* ───── API ───── */

/* Initialize tool registry */
bool tool_init(tool_registry* registry);

/* Register computer use tool */
bool tool_register_computer(tool_registry* registry);

/* Register browser tool */
bool tool_register_browser(tool_registry* registry);

/* Register file system tool */
bool tool_register_files(tool_registry* registry);

/* Register network tool */
bool tool_register_network(tool_registry* registry);

/* Execute tool call */
bool tool_execute(const tool_registry* registry, const tool_call* call, tool_response* response);

/* Execute computer action */
bool tool_execute_computer(const computer_use_context* ctx, const tool_call* call, tool_response* response);

/* Execute browser action */
bool tool_execute_browser(const computer_use_context* ctx, const tool_call* call, tool_response* response);

/* Execute file system action */
bool tool_execute_files(const computer_use_context* ctx, const tool_call* call, tool_response* response);

/* Execute network action */
bool tool_execute_network(const computer_use_context* ctx, const tool_call* call, tool_response* response);

/* Get available tools as JSON */
bool tool_get_definitions_json(const tool_registry* registry, char** json_output);

/* Cleanup tool resources */
void tool_cleanup(tool_registry* registry);

#endif /* COMPUTER_USE_H */
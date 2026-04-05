/*
 * Artifact Engine — Game Companion
 *
 * The AI agent that lives alongside the game. Not an overlay.
 * Not a chatbot. A persistent intelligence that:
 *
 *   1. SEES the game (frame capture → vision encoder → scene understanding)
 *   2. READS game state (memory hooks where available, vision where not)
 *   3. LEARNS the player (behavior embeddings built over sessions)
 *   4. ACTS (voice, text, or in-game if hooks allow)
 *   5. PERSISTS (state survives power cycles, builds over weeks/months)
 *
 * Architecture:
 *
 *   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 *   │ Frame Capture │────→│Vision Encoder│────→│              │
 *   └──────────────┘     └──────────────┘     │              │
 *   ┌──────────────┐                          │  Companion   │
 *   │ Game State   │─────────────────────────→│  LLM Core    │
 *   │ (memory/API) │                          │  (GGUF)      │
 *   └──────────────┘                          │              │
 *   ┌──────────────┐                          │              │
 *   │Player Profile│←────────────────────────→│              │
 *   │ (persistent) │                          └──────┬───────┘
 *   └──────────────┘                                 │
 *                                              ┌─────▼─────┐
 *                                              │  Output    │
 *                                              │ Voice/Text │
 *                                              │ Game Hooks │
 *                                              └───────────┘
 *
 * The companion shares GPU compute with the game via async compute queue.
 * Inference runs on RDNA 2 spare cycles (same GPU rendering the game).
 * This is NOT time-slicing — D3D12 async compute is designed for this.
 *
 * Memory budget: ~2GB VRAM for companion (out of 10GB on Xbox Series X)
 *   - Vision encoder: ~200MB (tiny ViT or MobileNet)
 *   - LLM (Q4_K): ~1.5GB (3B model) or ~500MB (1B model)
 *   - KV cache + working memory: ~300MB
 */

#ifndef COMPANION_H
#define COMPANION_H

#include <stdint.h>
#include <stdbool.h>

/* ───── Player Profile ───── */
typedef struct {
    /* Identity */
    char     gamertag[64];
    char     game_title[128];
    uint32_t session_count;          /* total play sessions observed */
    uint64_t total_playtime_sec;     /* total time observed */

    /* Behavior embeddings (learned representation of playstyle) */
    float*   behavior_embedding;     /* dim-dimensional vector */
    uint32_t embedding_dim;          /* typically 256 or 512 */
    uint64_t embedding_version;      /* incremented on each update */

    /* Play patterns (accumulated observations) */
    float    aggression;             /* 0.0 = passive, 1.0 = aggressive */
    float    exploration;            /* 0.0 = linear, 1.0 = explorer */
    float    caution;               /* 0.0 = reckless, 1.0 = methodical */
    float    social;                /* 0.0 = solo, 1.0 = cooperative */
    float    completionism;         /* 0.0 = mainline, 1.0 = 100% */
    float    skill_estimate;        /* 0.0-1.0 relative to game difficulty */

    /* Recent context */
    char     current_objective[256]; /* best guess at what player is doing */
    char     recent_events[1024];    /* narrative of recent game events */
    uint64_t last_death_time;        /* when player last died/failed */
    uint32_t deaths_this_session;
    uint32_t achievements_this_session;
} player_profile;

/* ───── Scene Understanding ───── */
typedef struct {
    /* Vision encoder output */
    float*   scene_embedding;        /* visual embedding of current frame */
    uint32_t embedding_dim;

    /* Extracted info (model-derived) */
    char     scene_description[512]; /* "player in dark cave, low health, 3 enemies" */
    float    danger_level;           /* 0.0 = safe, 1.0 = imminent death */
    float    novelty;                /* 0.0 = seen before, 1.0 = new area */
    bool     in_menu;                /* true if game is paused/in menu */
    bool     in_cutscene;            /* true if non-interactive */
    bool     loading_screen;         /* true if loading */
} scene_state;

/* ───── Companion Action ───── */
typedef enum {
    COMPANION_ACTION_NONE = 0,       /* Stay silent */
    COMPANION_ACTION_SPEAK,          /* Voice output */
    COMPANION_ACTION_TEXT,            /* Text overlay */
    COMPANION_ACTION_HINT,           /* Subtle hint (visual indicator) */
    COMPANION_ACTION_WARN,           /* Danger warning */
    COMPANION_ACTION_CELEBRATE,      /* Achievement/success reaction */
    COMPANION_ACTION_COMFORT,        /* After death/failure */
    COMPANION_ACTION_SUGGEST,        /* Strategy suggestion */
} companion_action_type;

typedef struct {
    companion_action_type type;
    char                  content[512];   /* what to say/show */
    float                 urgency;        /* 0.0 = can wait, 1.0 = NOW */
    float                 confidence;     /* how sure the companion is */
    uint64_t              cooldown_ms;    /* don't repeat for this long */
} companion_action;

/* ───── Companion Config ───── */
typedef struct {
    /* Model */
    char     model_path[512];         /* GGUF model for reasoning */
    char     vision_model_path[512];  /* GGUF model for vision encoding */
    uint32_t ctx_length;              /* context window */

    /* Personality */
    float    chattiness;              /* 0.0 = silent observer, 1.0 = always talking */
    float    humor;                   /* 0.0 = serious, 1.0 = playful */
    bool     voice_enabled;           /* TTS output */

    /* Frame capture */
    uint32_t capture_fps;             /* how often to capture (1-5 typical) */
    uint32_t vision_interval_ms;      /* how often to run vision encoder */

    /* Persistence */
    char     profile_path[512];       /* where to save player profile */
    bool     auto_save;               /* save profile periodically */
    uint32_t save_interval_sec;       /* auto-save frequency */

    /* Resource limits */
    size_t   max_vram_bytes;          /* hard cap on GPU memory usage */
} companion_config;

/* ───── Companion Context ───── */
typedef struct companion companion;

/* ───── API ───── */

/* Create and initialize the companion
 * Loads models, starts frame capture, loads player profile if exists */
companion* companion_create(const companion_config* config);

/* Process one frame + game state → decide action
 * Called from the main loop. Non-blocking if inference is still running.
 * Returns the companion's decided action (may be NONE). */
companion_action companion_tick(companion* c, uint64_t game_time_ms);

/* Notify the companion of a game event
 * Events the game (or a hook) can report. The companion incorporates
 * these into its understanding even without vision. */
typedef enum {
    GAME_EVENT_DEATH,
    GAME_EVENT_KILL,
    GAME_EVENT_ACHIEVEMENT,
    GAME_EVENT_LEVEL_UP,
    GAME_EVENT_ITEM_FOUND,
    GAME_EVENT_AREA_ENTERED,
    GAME_EVENT_BOSS_ENCOUNTER,
    GAME_EVENT_SAVE_POINT,
    GAME_EVENT_CUSTOM,
} game_event_type;

void companion_notify_event(companion* c, game_event_type event,
                            const char* details);

/* Player asks the companion something (voice or text input) */
companion_action companion_ask(companion* c, const char* question);

/* Get current scene understanding */
const scene_state* companion_get_scene(const companion* c);

/* Get player profile (read-only) */
const player_profile* companion_get_profile(const companion* c);

/* Save player profile to disk */
bool companion_save_profile(companion* c);

/* Load a different player profile */
bool companion_load_profile(companion* c, const char* profile_path);

/* Get companion memory/performance stats */
typedef struct {
    size_t   vram_used;
    double   avg_inference_ms;
    double   avg_vision_ms;
    double   avg_capture_ms;
    uint64_t total_inferences;
    uint64_t total_actions_taken;
    uint64_t session_start_time;
} companion_stats;

void companion_get_stats(const companion* c, companion_stats* stats);

/* Shutdown and save everything */
void companion_destroy(companion* c);

#endif /* COMPANION_H */

/*
 * Artifact Engine — Game Companion
 *
 * The AI agent that lives alongside the game. Sees frames, learns the
 * player, decides when to speak or stay silent. Persists across sessions.
 *
 * Architecture:
 *   Frame Capture → Downscale → Vision Embedding → LLM → Action
 *   Player Profile → Behavior EMA → Novelty Detection → Persistence
 *
 * v0.5.0: Full skeleton with frame capture, event tracking, profile
 *         persistence, behavior embeddings, and scene novelty detection.
 *         LLM inference integration deferred to when engine_forward()
 *         supports multi-context (game + companion).
 *
 * Compile: Always included. Frame capture only functional on Windows.
 */

#include "../include/companion.h"
#include "../include/frame_capture.h"
#include "../include/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
/* Use QueryPerformanceCounter for timing */
static uint64_t companion_time_ms(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (uint64_t)(now.QuadPart * 1000 / freq.QuadPart);
}
#else
#include <sys/time.h>
static uint64_t companion_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
#endif

/* ───── Constants ───── */

#define MAX_EVENTS          256
#define MAX_SCENE_HISTORY   16
#define DEFAULT_EMBED_DIM   256
#define BEHAVIOR_EMA_ALPHA  0.05f   /* slow-moving average for behavior embedding */
#define NOVELTY_THRESHOLD   0.3f    /* cosine distance threshold for "new area" */
#define PROFILE_SAVE_MAGIC  0x41564150  /* "AVAP" */
#define MAX_QUESTIONS        8

/* ───── Event Buffer Entry ───── */

typedef struct {
    game_event_type type;
    char            details[256];
    uint64_t        timestamp_ms;
} event_entry;

/* ───── Question Queue Entry ───── */

typedef struct {
    char     question[512];
    bool     pending;
    uint64_t queued_at_ms;
} question_entry;

/* ───── Internal Companion State ───── */

struct companion {
    companion_config  config;

    /* Frame capture */
    frame_capture*    capture;
    captured_frame    current_frame;
    captured_frame    downscaled_frame;
    uint64_t          last_vision_ms;

    /* Scene understanding */
    scene_state       scene;
    float*            scene_history[MAX_SCENE_HISTORY];  /* ring buffer of embeddings */
    uint32_t          scene_history_idx;
    uint32_t          scene_history_count;

    /* Player profile */
    player_profile    profile;
    bool              profile_dirty;

    /* Event buffer (ring) */
    event_entry       events[MAX_EVENTS];
    uint32_t          event_write_idx;
    uint32_t          event_count;

    /* Question queue */
    question_entry    questions[MAX_QUESTIONS];
    uint32_t          question_count;

    /* Inference state */
    bool              inference_running;
    uint64_t          inference_start_ms;
    companion_action  pending_action;

    /* Cooldown tracking */
    uint64_t          last_action_ms;
    uint64_t          last_save_ms;

    /* Engine reference (for inference — future) */
    engine*           eng;

    /* Stats */
    companion_stats   stats;
};

/* ═══════════════════════════════════════════════════════════
 * Player Profile — JSON Save/Load
 * ═══════════════════════════════════════════════════════════ */

static bool save_profile_json(const player_profile* p, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return false;

    fprintf(f, "{\n");
    fprintf(f, "  \"magic\": %u,\n", PROFILE_SAVE_MAGIC);
    fprintf(f, "  \"gamertag\": \"%s\",\n", p->gamertag);
    fprintf(f, "  \"game_title\": \"%s\",\n", p->game_title);
    fprintf(f, "  \"session_count\": %u,\n", p->session_count);
    fprintf(f, "  \"total_playtime_sec\": %llu,\n", (unsigned long long)p->total_playtime_sec);
    fprintf(f, "  \"embedding_dim\": %u,\n", p->embedding_dim);
    fprintf(f, "  \"embedding_version\": %llu,\n", (unsigned long long)p->embedding_version);
    fprintf(f, "  \"aggression\": %.6f,\n", p->aggression);
    fprintf(f, "  \"exploration\": %.6f,\n", p->exploration);
    fprintf(f, "  \"caution\": %.6f,\n", p->caution);
    fprintf(f, "  \"social\": %.6f,\n", p->social);
    fprintf(f, "  \"completionism\": %.6f,\n", p->completionism);
    fprintf(f, "  \"skill_estimate\": %.6f,\n", p->skill_estimate);
    fprintf(f, "  \"current_objective\": \"%s\",\n", p->current_objective);
    fprintf(f, "  \"deaths_this_session\": %u,\n", p->deaths_this_session);
    fprintf(f, "  \"achievements_this_session\": %u,\n", p->achievements_this_session);

    /* Save behavior embedding as array */
    if (p->behavior_embedding && p->embedding_dim > 0) {
        fprintf(f, "  \"behavior_embedding\": [");
        for (uint32_t i = 0; i < p->embedding_dim; i++) {
            fprintf(f, "%.8f", p->behavior_embedding[i]);
            if (i < p->embedding_dim - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\n");
    } else {
        fprintf(f, "  \"behavior_embedding\": []\n");
    }

    fprintf(f, "}\n");
    fclose(f);
    return true;
}

/* Simple JSON parser for profile loading (handles our specific format) */
static bool load_profile_json(player_profile* p, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return false;

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json = (char*)malloc(fsize + 1);
    if (!json) { fclose(f); return false; }
    fread(json, 1, fsize, f);
    json[fsize] = 0;
    fclose(f);

    /* Parse key-value pairs (naive but works for our format) */
    char* ptr;

    /* String fields */
    ptr = strstr(json, "\"gamertag\":");
    if (ptr) sscanf(ptr, "\"gamertag\": \"%63[^\"]\"", p->gamertag);

    ptr = strstr(json, "\"game_title\":");
    if (ptr) sscanf(ptr, "\"game_title\": \"%127[^\"]\"", p->game_title);

    ptr = strstr(json, "\"current_objective\":");
    if (ptr) sscanf(ptr, "\"current_objective\": \"%255[^\"]\"", p->current_objective);

    /* Integer fields */
    ptr = strstr(json, "\"session_count\":");
    if (ptr) sscanf(ptr, "\"session_count\": %u", &p->session_count);

    ptr = strstr(json, "\"total_playtime_sec\":");
    if (ptr) sscanf(ptr, "\"total_playtime_sec\": %llu", (unsigned long long*)&p->total_playtime_sec);

    ptr = strstr(json, "\"embedding_dim\":");
    if (ptr) sscanf(ptr, "\"embedding_dim\": %u", &p->embedding_dim);

    ptr = strstr(json, "\"embedding_version\":");
    if (ptr) sscanf(ptr, "\"embedding_version\": %llu", (unsigned long long*)&p->embedding_version);

    /* Float fields */
    ptr = strstr(json, "\"aggression\":");
    if (ptr) sscanf(ptr, "\"aggression\": %f", &p->aggression);

    ptr = strstr(json, "\"exploration\":");
    if (ptr) sscanf(ptr, "\"exploration\": %f", &p->exploration);

    ptr = strstr(json, "\"caution\":");
    if (ptr) sscanf(ptr, "\"caution\": %f", &p->caution);

    ptr = strstr(json, "\"social\":");
    if (ptr) sscanf(ptr, "\"social\": %f", &p->social);

    ptr = strstr(json, "\"completionism\":");
    if (ptr) sscanf(ptr, "\"completionism\": %f", &p->completionism);

    ptr = strstr(json, "\"skill_estimate\":");
    if (ptr) sscanf(ptr, "\"skill_estimate\": %f", &p->skill_estimate);

    /* Behavior embedding */
    if (p->embedding_dim > 0) {
        p->behavior_embedding = (float*)calloc(p->embedding_dim, sizeof(float));
        ptr = strstr(json, "\"behavior_embedding\": [");
        if (ptr && p->behavior_embedding) {
            ptr = strchr(ptr, '[') + 1;
            for (uint32_t i = 0; i < p->embedding_dim; i++) {
                while (*ptr == ' ' || *ptr == ',') ptr++;
                if (*ptr == ']') break;
                p->behavior_embedding[i] = (float)atof(ptr);
                while (*ptr && *ptr != ',' && *ptr != ']') ptr++;
            }
        }
    }

    free(json);
    return true;
}

/* ═══════════════════════════════════════════════════════════
 * Behavior Embedding — Exponential Moving Average
 * ═══════════════════════════════════════════════════════════ */

/*
 * Update behavior embedding with a new observation vector.
 * Uses EMA: new = alpha * observation + (1 - alpha) * old
 *
 * The observation vector is derived from game events:
 *   - Dimensions map to different behavioral axes
 *   - Events push specific dimensions up or down
 *   - Over time, the embedding converges to a stable player profile
 */
static void update_behavior_embedding(player_profile* p, const float* observation) {
    if (!p->behavior_embedding || !observation) return;

    for (uint32_t i = 0; i < p->embedding_dim; i++) {
        p->behavior_embedding[i] =
            BEHAVIOR_EMA_ALPHA * observation[i] +
            (1.0f - BEHAVIOR_EMA_ALPHA) * p->behavior_embedding[i];
    }
    p->embedding_version++;
}

/* Build an observation vector from a game event */
static void event_to_observation(float* obs, uint32_t dim,
                                  game_event_type event, const char* details) {
    (void)details;
    memset(obs, 0, dim * sizeof(float));
    if (dim < 8) return;  /* need at least 8 dims */

    /*
     * Observation vector mapping (first 8 dims):
     *   [0] aggression    [1] exploration    [2] caution
     *   [3] social        [4] skill          [5] patience
     *   [6] curiosity     [7] persistence
     */
    switch (event) {
        case GAME_EVENT_DEATH:
            obs[2] -= 0.2f;   /* died = less cautious */
            obs[4] -= 0.1f;   /* skill indicator */
            obs[7] += 0.1f;   /* persisting = patience */
            break;
        case GAME_EVENT_KILL:
            obs[0] += 0.3f;   /* aggressive action */
            obs[4] += 0.1f;   /* skill */
            break;
        case GAME_EVENT_ACHIEVEMENT:
            obs[4] += 0.2f;   /* skill / completion */
            obs[6] += 0.1f;   /* curiosity */
            break;
        case GAME_EVENT_LEVEL_UP:
            obs[4] += 0.15f;
            obs[7] += 0.1f;   /* persistence */
            break;
        case GAME_EVENT_ITEM_FOUND:
            obs[1] += 0.2f;   /* exploration */
            obs[6] += 0.2f;   /* curiosity */
            break;
        case GAME_EVENT_AREA_ENTERED:
            obs[1] += 0.3f;   /* exploration */
            obs[2] += 0.1f;   /* willingness to venture */
            break;
        case GAME_EVENT_BOSS_ENCOUNTER:
            obs[0] += 0.2f;   /* aggression (chose to fight) */
            obs[2] += 0.1f;   /* caution (prepared enough) */
            break;
        case GAME_EVENT_SAVE_POINT:
            obs[2] += 0.3f;   /* very cautious */
            obs[5] += 0.1f;   /* patience */
            break;
        case GAME_EVENT_CUSTOM:
            /* Custom events: no default mapping */
            break;
    }
}

/* ═══════════════════════════════════════════════════════════
 * Scene Novelty Detection
 * ═══════════════════════════════════════════════════════════ */

/* Cosine distance between two vectors */
static float cosine_distance(const float* a, const float* b, uint32_t dim) {
    float dot = 0, norm_a = 0, norm_b = 0;
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-8f || norm_b < 1e-8f) return 1.0f;
    float sim = dot / (sqrtf(norm_a) * sqrtf(norm_b));
    return 1.0f - sim;  /* 0 = identical, 2 = opposite */
}

/* Check if current scene is novel compared to recent history */
static float compute_novelty(companion* c, const float* scene_embedding) {
    if (!scene_embedding || c->scene_history_count == 0) return 1.0f;

    float min_dist = 2.0f;
    uint32_t dim = c->config.ctx_length > 0 ? DEFAULT_EMBED_DIM : DEFAULT_EMBED_DIM;

    for (uint32_t i = 0; i < c->scene_history_count && i < MAX_SCENE_HISTORY; i++) {
        if (c->scene_history[i]) {
            float dist = cosine_distance(scene_embedding, c->scene_history[i], dim);
            if (dist < min_dist) min_dist = dist;
        }
    }

    return min_dist;
}

/* Add scene to history ring buffer */
static void record_scene(companion* c, const float* embedding, uint32_t dim) {
    if (!embedding) return;

    uint32_t idx = c->scene_history_idx % MAX_SCENE_HISTORY;

    if (!c->scene_history[idx]) {
        c->scene_history[idx] = (float*)malloc(dim * sizeof(float));
    }

    if (c->scene_history[idx]) {
        memcpy(c->scene_history[idx], embedding, dim * sizeof(float));
        c->scene_history_idx++;
        if (c->scene_history_count < MAX_SCENE_HISTORY)
            c->scene_history_count++;
    }
}

/* ═══════════════════════════════════════════════════════════
 * Companion API
 * ═══════════════════════════════════════════════════════════ */

companion* companion_create(const companion_config* config) {
    if (!config) return NULL;

    companion* c = (companion*)calloc(1, sizeof(companion));
    if (!c) return NULL;

    c->config = *config;
    c->stats.session_start_time = companion_time_ms();

    /* ── Initialize frame capture ── */
    c->capture = frame_capture_init(
        config->capture_fps > 0 ? config->capture_fps : 2,
        false  /* CPU mode for now — need pixel data for downscale */
    );
    if (!c->capture) {
        fprintf(stderr, "companion: frame capture init failed (continuing without vision)\n");
    }

    /* ── Initialize player profile ── */
    c->profile.embedding_dim = DEFAULT_EMBED_DIM;
    c->profile.behavior_embedding = (float*)calloc(DEFAULT_EMBED_DIM, sizeof(float));
    strncpy(c->profile.gamertag, "Player", sizeof(c->profile.gamertag) - 1);

    /* Try loading existing profile */
    if (config->profile_path[0]) {
        if (load_profile_json(&c->profile, config->profile_path)) {
            printf("companion: loaded player profile (%u sessions, %llu sec playtime)\n",
                   c->profile.session_count,
                   (unsigned long long)c->profile.total_playtime_sec);
        } else {
            printf("companion: no existing profile, starting fresh\n");
        }
    }

    c->profile.session_count++;
    c->profile.deaths_this_session = 0;
    c->profile.achievements_this_session = 0;
    c->profile_dirty = true;

    /* ── Initialize scene state ── */
    c->scene.embedding_dim = DEFAULT_EMBED_DIM;
    c->scene.scene_embedding = (float*)calloc(DEFAULT_EMBED_DIM, sizeof(float));

    printf("companion: initialized (vision=%s, chattiness=%.1f, profile=%s)\n",
           c->capture ? "yes" : "no",
           config->chattiness,
           config->profile_path[0] ? config->profile_path : "none");

    return c;
}

companion_action companion_tick(companion* c, uint64_t game_time_ms) {
    if (!c) {
        companion_action none = { COMPANION_ACTION_NONE };
        return none;
    }

    uint64_t now_ms = companion_time_ms();
    companion_action action = { COMPANION_ACTION_NONE, "", 0, 0, 0 };

    /* ── 1. Frame capture + vision (if enough time since last) ── */
    if (c->capture) {
        uint32_t vision_interval = c->config.vision_interval_ms > 0 ?
                                   c->config.vision_interval_ms : 1000;

        if (now_ms - c->last_vision_ms >= vision_interval) {
            captured_frame frame;
            if (frame_capture_next(c->capture, &frame)) {
                c->current_frame = frame;

                /*
                 * Vision pipeline (placeholder for v0.5.0):
                 * 1. Downscale to 224×224
                 * 2. Convert to tensor
                 * 3. Run through vision encoder (GGUF model)
                 * 4. Get scene embedding
                 *
                 * For now: use frame stats as a simple "scene" signal.
                 */

                /* Simple scene change detection based on pixel luminance */
                if (frame.pixels && frame.width > 0 && frame.height > 0) {
                    /* Sample center pixels for basic scene understanding */
                    uint32_t cx = frame.width / 2;
                    uint32_t cy = frame.height / 2;
                    uint32_t stride = frame.stride;

                    /* Average luminance of center region */
                    float avg_lum = 0;
                    int samples = 0;
                    for (int dy = -10; dy <= 10; dy++) {
                        for (int dx = -10; dx <= 10; dx++) {
                            uint32_t px = cx + dx;
                            uint32_t py = cy + dy;
                            if (px < frame.width && py < frame.height) {
                                const uint8_t* p = frame.pixels + py * stride + px * 4;
                                avg_lum += 0.299f * p[2] + 0.587f * p[1] + 0.114f * p[0];
                                samples++;
                            }
                        }
                    }
                    if (samples > 0) avg_lum /= samples;

                    /* Update scene state heuristics */
                    c->scene.danger_level = avg_lum < 50 ? 0.7f : 0.2f;  /* dark = dangerous */
                    c->scene.loading_screen = avg_lum < 5 || avg_lum > 250;
                    c->scene.in_menu = false;  /* TODO: detect UI elements */
                    c->scene.in_cutscene = false;

                    /* Placeholder scene embedding from luminance pattern */
                    if (c->scene.scene_embedding) {
                        c->scene.scene_embedding[0] = avg_lum / 255.0f;
                        c->scene.novelty = compute_novelty(c, c->scene.scene_embedding);
                        record_scene(c, c->scene.scene_embedding, DEFAULT_EMBED_DIM);
                    }

                    c->stats.avg_capture_ms =
                        (c->stats.avg_capture_ms * c->stats.total_inferences + 1.0) /
                        (c->stats.total_inferences + 1);
                }

                /* Free pixel copy if we made one */
                if (frame.pixels && !frame.on_gpu) {
                    free(frame.pixels);
                }

                c->last_vision_ms = now_ms;
            }
        }
    }

    /* ── 2. Check for pending questions ── */
    for (uint32_t i = 0; i < c->question_count; i++) {
        if (c->questions[i].pending) {
            /*
             * TODO: Queue question for LLM inference.
             * For v0.5.0: return a placeholder response.
             */
            action.type = COMPANION_ACTION_SPEAK;
            snprintf(action.content, sizeof(action.content),
                     "Hmm, let me think about that... \"%s\"", c->questions[i].question);
            action.confidence = 0.5f;
            action.urgency = 0.3f;
            action.cooldown_ms = 5000;

            c->questions[i].pending = false;
            c->stats.total_actions_taken++;
            c->last_action_ms = now_ms;
            return action;
        }
    }

    /* ── 3. Proactive actions based on game state ── */
    uint64_t cooldown = 10000;  /* minimum 10s between actions */
    float chattiness_scale = c->config.chattiness;
    cooldown = (uint64_t)(cooldown / (chattiness_scale > 0.1f ? chattiness_scale : 0.1f));

    if (now_ms - c->last_action_ms < cooldown) {
        return action;  /* still in cooldown */
    }

    /* Check recent events for reaction opportunities */
    if (c->event_count > 0) {
        uint32_t last_idx = (c->event_write_idx - 1) % MAX_EVENTS;
        event_entry* last = &c->events[last_idx];

        /* Only react to recent events (within 5 seconds) */
        if (now_ms - last->timestamp_ms < 5000) {
            switch (last->type) {
                case GAME_EVENT_DEATH:
                    if (c->profile.deaths_this_session > 3) {
                        action.type = COMPANION_ACTION_SUGGEST;
                        snprintf(action.content, sizeof(action.content),
                                 "That's %u deaths now. Maybe try a different approach?",
                                 c->profile.deaths_this_session);
                        action.confidence = 0.7f;
                        action.urgency = 0.4f;
                    } else {
                        action.type = COMPANION_ACTION_COMFORT;
                        snprintf(action.content, sizeof(action.content),
                                 "Tough break. You'll get them next time.");
                        action.confidence = 0.8f;
                        action.urgency = 0.2f;
                    }
                    action.cooldown_ms = 15000;
                    break;

                case GAME_EVENT_ACHIEVEMENT:
                    action.type = COMPANION_ACTION_CELEBRATE;
                    snprintf(action.content, sizeof(action.content),
                             "Nice! Achievement unlocked. %s",
                             last->details[0] ? last->details : "");
                    action.confidence = 0.9f;
                    action.urgency = 0.5f;
                    action.cooldown_ms = 10000;
                    break;

                case GAME_EVENT_BOSS_ENCOUNTER:
                    action.type = COMPANION_ACTION_WARN;
                    snprintf(action.content, sizeof(action.content),
                             "Boss incoming. Stay sharp.");
                    action.confidence = 0.85f;
                    action.urgency = 0.8f;
                    action.cooldown_ms = 30000;
                    break;

                case GAME_EVENT_AREA_ENTERED:
                    if (c->scene.novelty > NOVELTY_THRESHOLD) {
                        action.type = COMPANION_ACTION_HINT;
                        snprintf(action.content, sizeof(action.content),
                                 "New area. %s", last->details[0] ? last->details : "Look around.");
                        action.confidence = 0.6f;
                        action.urgency = 0.2f;
                        action.cooldown_ms = 20000;
                    }
                    break;

                default:
                    break;
            }
        }
    }

    if (action.type != COMPANION_ACTION_NONE) {
        c->stats.total_actions_taken++;
        c->last_action_ms = now_ms;
    }

    /* ── 4. Auto-save profile periodically ── */
    if (c->config.auto_save && c->profile_dirty) {
        uint32_t save_interval = c->config.save_interval_sec > 0 ?
                                 c->config.save_interval_sec : 60;
        if (now_ms - c->last_save_ms > (uint64_t)save_interval * 1000) {
            companion_save_profile(c);
            c->last_save_ms = now_ms;
        }
    }

    /* Update playtime */
    c->profile.total_playtime_sec =
        (uint64_t)((now_ms - c->stats.session_start_time) / 1000);

    return action;
}

void companion_notify_event(companion* c, game_event_type event,
                            const char* details) {
    if (!c) return;

    /* Add to event ring buffer */
    uint32_t idx = c->event_write_idx % MAX_EVENTS;
    c->events[idx].type = event;
    c->events[idx].timestamp_ms = companion_time_ms();
    if (details) {
        strncpy(c->events[idx].details, details, sizeof(c->events[idx].details) - 1);
        c->events[idx].details[sizeof(c->events[idx].details) - 1] = 0;
    } else {
        c->events[idx].details[0] = 0;
    }
    c->event_write_idx++;
    if (c->event_count < MAX_EVENTS) c->event_count++;

    /* Update player profile stats */
    switch (event) {
        case GAME_EVENT_DEATH:
            c->profile.deaths_this_session++;
            c->profile.last_death_time = companion_time_ms();
            /* Dying decreases skill estimate slightly */
            c->profile.skill_estimate =
                c->profile.skill_estimate * 0.98f;
            break;
        case GAME_EVENT_KILL:
            c->profile.aggression =
                c->profile.aggression * 0.95f + 0.05f * 0.8f;
            c->profile.skill_estimate =
                c->profile.skill_estimate * 0.99f + 0.01f * 0.7f;
            break;
        case GAME_EVENT_ACHIEVEMENT:
            c->profile.achievements_this_session++;
            c->profile.completionism =
                c->profile.completionism * 0.95f + 0.05f * 0.9f;
            break;
        case GAME_EVENT_ITEM_FOUND:
            c->profile.exploration =
                c->profile.exploration * 0.95f + 0.05f * 0.8f;
            break;
        case GAME_EVENT_AREA_ENTERED:
            c->profile.exploration =
                c->profile.exploration * 0.95f + 0.05f * 0.9f;
            break;
        case GAME_EVENT_BOSS_ENCOUNTER:
            c->profile.aggression =
                c->profile.aggression * 0.95f + 0.05f * 0.7f;
            break;
        case GAME_EVENT_SAVE_POINT:
            c->profile.caution =
                c->profile.caution * 0.95f + 0.05f * 0.9f;
            break;
        default:
            break;
    }

    /* Update behavior embedding */
    if (c->profile.behavior_embedding) {
        float* obs = (float*)calloc(c->profile.embedding_dim, sizeof(float));
        if (obs) {
            event_to_observation(obs, c->profile.embedding_dim, event, details);
            update_behavior_embedding(&c->profile, obs);
            free(obs);
        }
    }

    /* Append to recent events narrative */
    const char* event_names[] = {
        "died", "killed enemy", "achievement", "level up",
        "item found", "new area", "boss encounter", "save point", "event"
    };
    const char* ename = event < 9 ? event_names[event] : "event";

    char entry[256];
    if (details && details[0]) {
        snprintf(entry, sizeof(entry), "[%s: %s] ", ename, details);
    } else {
        snprintf(entry, sizeof(entry), "[%s] ", ename);
    }

    /* Append to recent_events (circular, truncate if too long) */
    size_t current_len = strlen(c->profile.recent_events);
    size_t entry_len = strlen(entry);
    if (current_len + entry_len < sizeof(c->profile.recent_events) - 1) {
        strcat(c->profile.recent_events, entry);
    } else {
        /* Truncate old events from the beginning */
        memmove(c->profile.recent_events,
                c->profile.recent_events + entry_len,
                current_len - entry_len + 1);
        strcat(c->profile.recent_events, entry);
    }

    c->profile_dirty = true;
}

companion_action companion_ask(companion* c, const char* question) {
    companion_action action = { COMPANION_ACTION_NONE, "", 0, 0, 0 };
    if (!c || !question) return action;

    /* Queue the question */
    if (c->question_count < MAX_QUESTIONS) {
        question_entry* q = &c->questions[c->question_count++];
        strncpy(q->question, question, sizeof(q->question) - 1);
        q->question[sizeof(q->question) - 1] = 0;
        q->pending = true;
        q->queued_at_ms = companion_time_ms();
    }

    /*
     * For v0.5.0: return immediate placeholder.
     * When LLM inference is integrated, this queues the question
     * and companion_tick() returns the answer when ready.
     */
    action.type = COMPANION_ACTION_TEXT;
    snprintf(action.content, sizeof(action.content),
             "Thinking about: \"%s\"...", question);
    action.confidence = 0.3f;
    action.urgency = 0.5f;
    action.cooldown_ms = 3000;

    return action;
}

const scene_state* companion_get_scene(const companion* c) {
    return c ? &c->scene : NULL;
}

const player_profile* companion_get_profile(const companion* c) {
    return c ? &c->profile : NULL;
}

bool companion_save_profile(companion* c) {
    if (!c || !c->config.profile_path[0]) return false;

    bool ok = save_profile_json(&c->profile, c->config.profile_path);
    if (ok) {
        c->profile_dirty = false;
        printf("companion: saved profile to %s\n", c->config.profile_path);
    } else {
        fprintf(stderr, "companion: failed to save profile to %s\n",
                c->config.profile_path);
    }
    return ok;
}

bool companion_load_profile(companion* c, const char* profile_path) {
    if (!c || !profile_path) return false;

    /* Free old embedding */
    if (c->profile.behavior_embedding) {
        free(c->profile.behavior_embedding);
        c->profile.behavior_embedding = NULL;
    }

    memset(&c->profile, 0, sizeof(c->profile));

    if (load_profile_json(&c->profile, profile_path)) {
        strncpy(c->config.profile_path, profile_path,
                sizeof(c->config.profile_path) - 1);
        printf("companion: loaded profile from %s\n", profile_path);
        return true;
    }
    return false;
}

void companion_get_stats(const companion* c, companion_stats* stats) {
    if (!c || !stats) return;

    *stats = c->stats;

    /* Update VRAM used (approximate) */
    stats->vram_used = 0;  /* TODO: track DML resource allocations */

    /* Update capture timing from frame capture */
    if (c->capture) {
        frame_capture_stats fcs;
        frame_capture_get_stats(c->capture, &fcs);
        stats->avg_capture_ms = fcs.avg_capture_ms;
    }
}

void companion_destroy(companion* c) {
    if (!c) return;

    /* Save profile on exit */
    if (c->config.profile_path[0] && c->profile_dirty) {
        companion_save_profile(c);
    }

    /* Free frame capture */
    if (c->capture) frame_capture_destroy(c->capture);

    /* Free downscaled frame */
    if (c->downscaled_frame.pixels) free(c->downscaled_frame.pixels);

    /* Free scene history */
    for (int i = 0; i < MAX_SCENE_HISTORY; i++) {
        if (c->scene_history[i]) free(c->scene_history[i]);
    }

    /* Free scene embedding */
    if (c->scene.scene_embedding) free(c->scene.scene_embedding);

    /* Free behavior embedding */
    if (c->profile.behavior_embedding) free(c->profile.behavior_embedding);

    printf("companion: destroyed (session %u, %u deaths, %u achievements)\n",
           c->profile.session_count,
           c->profile.deaths_this_session,
           c->profile.achievements_this_session);

    free(c);
}

#pragma once

#include "common.h"
#include "preset.h"
#include "server-common.h"
#include "server-models.h"

#include <mutex>
#include <condition_variable>
#include <optional>
#include <set>
#include <map>

/**
 * In-process model manager for non-router mode multi-model support.
 *
 * Unlike router mode (which manages child processes), this manager loads
 * and unloads models within the same process using server_context's
 * load_model() / destroy() methods.
 *
 * State diagram:
 *
 * UNLOADED ──► LOADING ──► LOADED
 *  ▲                     │
 *  └── failed ───────────┘
 */

struct server_model_info {
    std::string name;           // canonical model name
    std::set<std::string> aliases; // aliases that resolve to this model
    std::set<std::string> tags;   // informational tags
    std::string model_path;     // path to the GGUF file
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0;      // for LRU eviction (milliseconds since epoch)
    int exit_code = 0;          // exit code if failed
    bool cached = false;        // GGUF file is cached in page cache for fast swapping
    common_preset preset;       // per-model preset for applying to common_params at load/swap time

    bool is_ready() const {
        return status == SERVER_MODEL_STATUS_LOADED;
    }

    bool is_running() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }
};

// Forward declaration
class server_context;

class server_model_manager {
public:
    server_model_manager(int models_max, bool models_autoload);

    // Register a model (from CLI args, presets, etc.)
    void add_model(server_model_info&& info);

    // Check if a model exists (by --id name, --alias, or internal model name)
    bool has_model(const std::string& name) const;

    // Get model metadata (resolves by --id name, --alias, or internal model name)
    // Returns the canonical name in the result for use in subsequent operations
    std::optional<server_model_info> get_meta(const std::string& name) const;

    // Get all model metadata
    std::vector<server_model_info> get_all_meta() const;

    // Update a model's file path
    void set_model_path(const std::string& name, const std::string& path);

    // Load a model (with LRU eviction if needed)
    // The common_params passed here is constructed from the model's preset (via preset.apply_to_params())
    void load(const std::string& name, server_context& ctx, common_params params);

    // Unload a model
    void unload(const std::string& name, server_context& ctx);

    // Unload all models
    void unload_all(server_context& ctx);

    // Unload the LRU model
    void unload_lru(server_context& ctx);

    // Ensure a model is loaded (load if necessary, with LRU eviction)
    // Returns true if model is ready, false if loading is in progress
    // last_used is updated AFTER the request finishes processing (not on arrival),
    // so a model with no in-flight requests is eligible for eviction
    bool ensure_model_ready(const std::string& name, server_context& ctx, common_params params);

    // Wait until a model finishes loading (thread-safe)
    void wait_until_loading_finished(const std::string& name);

    // Cache a model's GGUF file in page cache (for fast swapping)
    void cache(const std::string& name);

    // Cache all models' GGUF files in page cache
    void cache_all();

    // Get the per-model preset for a given model name (resolves aliases)
    // Returns empty optional if model not found or no preset set
    std::optional<common_preset> get_preset(const std::string& name) const;

private:
    // Find the LRU model name (must be called with mutex_ held)
    // Returns empty string if no model to evict
    std::string find_lru_model() const;

    // Resolve a model name to canonical name (must be called with mutex_ held)
    // Returns empty string if not found
    std::string resolve_model_name(const std::string& name) const;

    // Internal load (must be called with mutex_ held)
    void load_locked(const std::string& name, server_context& ctx, common_params params);

    // Internal unload (must be called with mutex_ held)
    void unload_locked(const std::string& name);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::map<std::string, server_model_info> mapping_;
    int models_max_;
    bool models_autoload_;
};

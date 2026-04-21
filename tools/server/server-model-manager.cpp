#include "server-model-manager.h"
#include "server-context.h"

#include <algorithm>
#include <chrono>

//
// server_model_manager
//

server_model_manager::server_model_manager(int models_max, bool models_autoload)
    : models_max_(models_max), models_autoload_(models_autoload) {
}

void server_model_manager::add_model(server_model_info&& info) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (mapping_.find(info.name) != mapping_.end()) {
        SRV_WRN("model '%s' already registered, skipping\n", info.name.c_str());
        return;
    }

    // parse aliases from comma-separated string
    std::string alias_str;
    // alias is stored in the info struct already from preset parsing
    std::set<std::string> new_aliases;
    for (const auto & alias : info.aliases) {
        new_aliases.insert(string_strip(alias));
    }
    info.aliases = std::move(new_aliases);

    std::string registered_name = info.name;
    mapping_[registered_name] = std::move(info);
    SRV_INF("registered model '%s'\n", registered_name.c_str());
}

bool server_model_manager::has_model(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mutex_);
    return resolve_model_name(name) != "";
}

std::optional<server_model_info> server_model_manager::get_meta(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::string canonical = resolve_model_name(name);
    if (canonical.empty()) {
        return std::nullopt;
    }
    auto it = mapping_.find(canonical);
    if (it != mapping_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<server_model_info> server_model_manager::get_all_meta() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<server_model_info> result;
    result.reserve(mapping_.size());
    for (const auto& [name, info] : mapping_) {
        if (!name.empty()) {
            result.push_back(info);
        }
    }
    return result;
}

void server_model_manager::set_model_path(const std::string& name, const std::string& path) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = mapping_.find(name);
    if (it != mapping_.end()) {
        it->second.model_path = path;
    }
}

bool server_model_manager::ensure_model_ready(const std::string& name, server_context& ctx, common_params params) {
    std::unique_lock<std::mutex> lk(mutex_);

    std::string canonical = resolve_model_name(name);
    if (canonical.empty()) {
        SRV_ERR("model '%s' not found\n", name.c_str());
        return false;
    }

    auto& info = mapping_[canonical];

    // Already loaded? Update last_used and return
    if (info.status == SERVER_MODEL_STATUS_LOADED) {
        info.last_used = ggml_time_ms();
        return true;
    }

    // Currently loading? Wait for it
    if (info.status == SERVER_MODEL_STATUS_LOADING) {
        // Release lock and wait
        cv_.wait(lk, [&info]() {
            return info.status != SERVER_MODEL_STATUS_LOADING;
        });
        // After wait, check if still loaded
        if (info.status == SERVER_MODEL_STATUS_LOADED) {
            return true;
        }
        // If failed, fall through to retry
        SRV_WRN("model '%s' loading failed, retrying...\n", canonical.c_str());
    }

    // Need to load - check capacity
    if (models_max_ > 0) {
        size_t count_active = 0;
        for (const auto& m : mapping_) {
            if (m.second.is_running()) {
                count_active++;
            }
        }
        if (count_active >= (size_t)models_max_) {
            // Evict LRU
            std::string lru = find_lru_model();
            if (!lru.empty()) {
                SRV_INF("models_max limit reached, unloading LRU model '%s'\n", lru.c_str());
                // Release lock for unload (which will re-acquire)
                lk.unlock();
                unload(lru, ctx);
                lk.lock();
                // Re-check capacity
                count_active = 0;
                for (const auto& m : mapping_) {
                    if (m.second.is_running()) {
                        count_active++;
                    }
                }
                if (count_active >= (size_t)models_max_) {
                    SRV_INF("still at models_max limit after LRU eviction\n", nullptr);
                    return false;
                }
            }
        }
    }

    // Load the model
    load_locked(canonical, ctx, std::move(params));
    return true;
}

void server_model_manager::load(const std::string& name, server_context& ctx, common_params params) {
    std::unique_lock<std::mutex> lk(mutex_);

    std::string canonical = resolve_model_name(name);
    if (canonical.empty()) {
        SRV_ERR("model '%s' not found\n", name.c_str());
        return;
    }

    auto& info = mapping_[canonical];
    if (info.status == SERVER_MODEL_STATUS_LOADED) {
        SRV_INF("model '%s' already loaded\n", canonical.c_str());
        return;
    }

    // Check capacity
    if (models_max_ > 0) {
        size_t count_active = 0;
        for (const auto& m : mapping_) {
            if (m.second.is_running()) {
                count_active++;
            }
        }
        if (count_active >= (size_t)models_max_) {
            // Evict LRU
            std::string lru = find_lru_model();
            if (!lru.empty()) {
                SRV_INF("models_max limit reached, unloading LRU model '%s'\n", lru.c_str());
                lk.unlock();
                unload(lru, ctx);
                lk.lock();
            }
        }
    }

    load_locked(canonical, ctx, std::move(params));
}

void server_model_manager::load_locked(const std::string& name, server_context& ctx, common_params params) {
    auto& info = mapping_[name];
    info.status = SERVER_MODEL_STATUS_LOADING;
    info.last_used = ggml_time_ms();

    // Use this model's path, not the global params path
    std::string saved_path = params.model.path;
    params.model.path = info.model_path;
    SRV_INF("loading model '%s' (path: %s)\n", name.c_str(), params.model.path.c_str());

    // Call server_context's swap_model which handles the actual loading
    bool success = ctx.swap_model(params);

    // Restore the global params path
    params.model.path = saved_path;

    if (success) {
        info.status = SERVER_MODEL_STATUS_LOADED;
        SRV_INF("model '%s' loaded successfully\n", name.c_str());
    } else {
        info.status = SERVER_MODEL_STATUS_UNLOADED;
        info.exit_code = 1;
        SRV_ERR("model '%s' failed to load\n", name.c_str());
    }

    cv_.notify_all();
}

void server_model_manager::unload(const std::string& name, server_context& ctx) {
    std::lock_guard<std::mutex> lk(mutex_);

    std::string canonical = resolve_model_name(name);
    if (canonical.empty()) {
        SRV_WRN("model '%s' not found, nothing to unload\n", name.c_str());
        return;
    }

    auto& info = mapping_[canonical];
    if (!info.is_running()) {
        SRV_WRN("model '%s' is not running, nothing to unload\n", canonical.c_str());
        return;
    }

    SRV_INF("unloading model '%s'\n", canonical.c_str());
    info.status = SERVER_MODEL_STATUS_UNLOADED;
    info.last_used = 0;

    // Call server_context's unload
    ctx.unload_current_model();

    cv_.notify_all();
}

void server_model_manager::unload_locked(const std::string& name) {
    auto it = mapping_.find(name);
    if (it == mapping_.end()) {
        return;
    }
    it->second.status = SERVER_MODEL_STATUS_UNLOADED;
    it->second.last_used = 0;
    it->second.exit_code = 0;
}

void server_model_manager::unload_all(server_context& ctx) {
    std::lock_guard<std::mutex> lk(mutex_);

    for (auto& [name, info] : mapping_) {
        if (info.is_running()) {
            SRV_INF("unloading model '%s'\n", name.c_str());
            info.status = SERVER_MODEL_STATUS_UNLOADED;
            info.last_used = 0;
        }
    }
    ctx.unload_current_model();
}

void server_model_manager::wait_until_loading_finished(const std::string& name) {
    std::unique_lock<std::mutex> lk(mutex_);
    std::string canonical = resolve_model_name(name);
    if (canonical.empty()) {
        return;
    }

    cv_.wait(lk, [&canonical, this]() {
        auto it = mapping_.find(canonical);
        if (it == mapping_.end()) {
            return true;
        }
        return it->second.status != SERVER_MODEL_STATUS_LOADING;
    });
}

void server_model_manager::unload_lru(server_context& ctx) {
    std::string lru = find_lru_model();
    if (!lru.empty()) {
        unload(lru, ctx);
    }
}

std::string server_model_manager::find_lru_model() const {
    std::string lru_name;
    int64_t lru_last_used = INT64_MAX;

    for (const auto& [name, info] : mapping_) {
        if (info.is_running() && info.last_used < lru_last_used) {
            lru_name = name;
            lru_last_used = info.last_used;
        }
    }

    return lru_name;
}

std::string server_model_manager::resolve_model_name(const std::string& name) const {
    // 1. Exact match by canonical name
    auto it = mapping_.find(name);
    if (it != mapping_.end()) {
        return it->first;
    }

    // 2. Exact match by alias
    for (const auto& [canonical, info] : mapping_) {
        if (info.aliases.count(name) > 0) {
            return canonical;
        }
    }

    return "";
}

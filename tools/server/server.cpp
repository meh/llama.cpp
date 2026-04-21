#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-model-manager.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include "arg.h"
#include "download.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "preset.h"

// Forward declaration for res_ok (defined in server-models.cpp)
extern void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data);

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // own arguments required by this example
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    // see https://github.com/ggml-org/llama.cpp/issues/12836
    if (params.embedding && params.n_batch > params.n_ubatch) {
        LOG_WRN("%s: embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", __func__, params.n_batch, params.n_ubatch);
        LOG_WRN("%s: setting n_batch = n_ubatch = %d to avoid assertion failure\n", __func__, params.n_ubatch);
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        LOG_INF("%s: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n", __func__);

        params.n_parallel = 4;
        params.kv_unified = true;
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    // struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("build_info: %s\n", llama_build_info());
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return 1;
    }

    // Multi-model manager for non-router mode
    std::unique_ptr<server_model_manager> model_manager;
    std::string model_manager_base_model_name;

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server);
    server_tools tools;

    // Check if CLI has -- separated models (multi-model mode)
    common_preset_context ctx_preset_cli(LLAMA_EXAMPLE_SERVER);
    auto cli_load_result = ctx_preset_cli.load_from_args(argc, argv);
    bool cli_has_model_presets = !cli_load_result.model_presets.empty();

    bool is_router_server = params.model.path.empty() && !cli_has_model_presets;
    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to initialize router models: %s\n", __func__, e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));
    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.webui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
    }
    // EXPERIMENTAL built-in tools
    if (!params.server_tools.empty()) {
        tools.setup(params.server_tools);
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        LOG_INF("%s: starting router server, no model will be loaded in this process\n", __func__);

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (models_routes.has_value()) {
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            ctx_http.stop();
        };

    } else {
        // Non-router mode: load model directly
        // Check if multi-model support is enabled
        // Also check if -- separator was used with multiple models
        common_preset_context ctx_preset_cli(LLAMA_EXAMPLE_SERVER);
        auto cli_load_result = ctx_preset_cli.load_from_args(argc, argv);
        bool cli_has_model_presets = !cli_load_result.model_presets.empty();
        bool multi_model_enabled = params.models_max > 1 || !params.models_preset.empty() || !params.models_dir.empty() || cli_has_model_presets;

        if (multi_model_enabled) {
            LOG_INF("%s: starting multi-model server (models_max=%d)\n", __func__, params.models_max);
        } else {
            LOG_INF("%s: starting single-model server\n", __func__);
        }

        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server, &model_manager]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            ctx_http.stop();
            ctx_server.terminate();
            if (model_manager) {
                model_manager->unload_all(ctx_server);
            }
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
            return 1;
        }

        // If CLI had -- separated model presets, the first one is the base model
        // Use its path and override params so ctx_server.load_model loads the right model
        std::string cli_base_model_path;
        std::string cli_base_model_name;
        if (cli_has_model_presets && !cli_load_result.model_presets.empty()) {
            const auto & first_preset = cli_load_result.model_presets[0];
            first_preset.get_option("LLAMA_ARG_MODEL", cli_base_model_path);
            if (cli_base_model_path.empty()) {
                first_preset.get_option("-m", cli_base_model_path);
            }
            if (cli_base_model_path.empty()) {
                first_preset.get_option("LLAMA_ARG_HF_REPO", cli_base_model_path);
            }
            if (cli_base_model_path.empty()) {
                first_preset.get_option("LLAMA_ARG_HF_FILE", cli_base_model_path);
            }
            // Resolve HF repo to local path for the base model (first preset)
            // This must happen before common_params_parse_ex processes subsequent presets
            // which would overwrite params.model.hf_repo
            std::string hf_repo_check;
            bool has_hf_repo = first_preset.get_option("LLAMA_ARG_HF_REPO", hf_repo_check);
            LOG_INF("%s: first preset: cli_base_model_path='%s' has_hf_repo=%d\n", __func__, cli_base_model_path.c_str(), has_hf_repo);
            if (!cli_base_model_path.empty() && has_hf_repo) {
                std::string hf_repo;
                first_preset.get_option("LLAMA_ARG_HF_REPO", hf_repo);
                common_download_opts opts;
                opts.bearer_token = params.hf_token;
                opts.offline = params.offline;
                // Create a temp model struct for download
                common_params_model tmp_model;
                tmp_model.hf_repo = hf_repo;
                // Get hf_file from LLAMA_ARG_HF_FILE or derive from the repo's GGUF file list
                first_preset.get_option("LLAMA_ARG_HF_FILE", tmp_model.hf_file);
                if (tmp_model.hf_file.empty()) {
                    // Leave hf_file empty — common_download_model will use find_best_model
                    // which matches the quant tag from hf_repo (e.g., Q4_K_M)
                }
                LOG_INF("%s: tmp model hf_repo='%s' hf_file='%s'\n", __func__, hf_repo.c_str(), tmp_model.hf_file.c_str());
                auto download_result = common_download_model(tmp_model, opts, true);
                if (!download_result.model_path.empty()) {
                    cli_base_model_path = download_result.model_path;
                    LOG_INF("%s: resolved HF repo '%s' to local path '%s'\n", __func__, hf_repo.c_str(), cli_base_model_path.c_str());
                }
            }
            if (!cli_base_model_path.empty()) {
                params.model.path = cli_base_model_path;
                cli_base_model_name = std::filesystem::path(cli_base_model_path).filename().string();
                LOG_INF("%s: using CLI -- separated model as base: %s\n", __func__, cli_base_model_path.c_str());
            }
            // Extract --id from the first model preset (index 0 in model_presets = first block after --)
            // params.model_id is overwritten by subsequent --id flags during common_params_parse
            std::string cli_base_model_id;
            if (!cli_load_result.model_presets.empty()) {
                cli_load_result.model_presets[0].get_option("LLAMA_ARG_ID", cli_base_model_id);
            }
            if (!cli_base_model_id.empty()) {
                params.model_id = cli_base_model_id;
                LOG_INF("%s: base model id from first model preset: %s\n", __func__, cli_base_model_id.c_str());
            }
        }

        // Determine base model name
        for (const auto & alias : params.model_alias) {
            if (!alias.empty()) {
                model_manager_base_model_name = alias;
                break;
            }
        }
        if (model_manager_base_model_name.empty() && cli_base_model_name.empty()) {
            if (!params.model.name.empty()) {
                model_manager_base_model_name = params.model.name;
            }
        }
        if (model_manager_base_model_name.empty() && !cli_base_model_name.empty()) {
            model_manager_base_model_name = cli_base_model_name;
        }
        // --id overrides the model name
        if (!params.model_id.empty()) {
            model_manager_base_model_name = params.model_id;
        }
        if (model_manager_base_model_name.empty()) {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params.model.path);
            model_manager_base_model_name = model_path.filename().string();
        }

        // For CLI -- separated models, create model manager before loading
        // so that additional models can be registered alongside the base model
        if (cli_has_model_presets) {
            model_manager = std::make_unique<server_model_manager>(params.models_max, params.models_autoload);

            // Register the base model
            server_model_info base_info;
            base_info.name = model_manager_base_model_name;
            base_info.model_path = params.model.path;
            base_info.aliases = params.model_alias;
            base_info.tags = params.model_tags;
            base_info.status = SERVER_MODEL_STATUS_LOADED;
            base_info.last_used = ggml_time_ms();
            if (!base_info.name.empty()) {
                model_manager->add_model(std::move(base_info));
            }

            // Register additional models from CLI -- separated model presets
            for (size_t i = 1; i < cli_load_result.model_presets.size(); i++) {
                const auto & mp = cli_load_result.model_presets[i];
                std::string model_path;
                mp.get_option("LLAMA_ARG_MODEL", model_path);
                if (model_path.empty()) {
                    mp.get_option("-m", model_path);
                }
                if (model_path.empty()) {
                    mp.get_option("LLAMA_ARG_HF_REPO", model_path);
                }
                if (!model_path.empty()) {
                    std::string preset_name = std::filesystem::path(model_path).filename().string();
                    // Check if --id was specified for this preset
                    std::string id_str;
                    mp.get_option("LLAMA_ARG_ID", id_str);
                    if (!id_str.empty()) {
                        preset_name = id_str;
                    }
                    server_model_info info;
                    info.name = preset_name;
                    info.model_path = model_path;
                    info.status = SERVER_MODEL_STATUS_UNLOADED;
                    info.last_used = 0;

                    // Parse --alias and --tags from preset
                    std::string alias_str;
                    if (mp.get_option("LLAMA_ARG_ALIAS", alias_str)) {
                        std::stringstream ss(alias_str);
                        std::string token;
                        while (std::getline(ss, token, ',')) {
                            token = string_strip(token);
                            if (!token.empty()) {
                                info.aliases.insert(token);
                            }
                        }
                    }
                    std::string tag_str;
                    if (mp.get_option("LLAMA_ARG_TAGS", tag_str)) {
                        std::stringstream ss(tag_str);
                        std::string token;
                        while (std::getline(ss, token, ',')) {
                            token = string_strip(token);
                            if (!token.empty()) {
                                info.tags.insert(token);
                            }
                        }
                    }

                    SRV_INF("registering model '%s' (status=%d)\n", info.name.c_str(), (int)info.status);
                    model_manager->add_model(std::move(info));
                }
            }
        }

        // Wire up model_manager pointer to routes (CLI case)
        if (cli_has_model_presets) {
            routes.set_model_manager(model_manager.get());
        }

        // Resolve HF repo to local path for additional CLI models (base model already resolved above)
        // Also update each model's path in the model manager
        if (cli_has_model_presets) {
            for (size_t i = 1; i < cli_load_result.model_presets.size(); i++) {
                const auto & mp = cli_load_result.model_presets[i];
                std::string hf_repo;
                if (mp.get_option("LLAMA_ARG_HF_REPO", hf_repo) && !hf_repo.empty()) {
                    common_download_opts opts;
                    opts.bearer_token = params.hf_token;
                    opts.offline = params.offline;
                    common_params_model tmp_model;
                    tmp_model.hf_repo = hf_repo;
                    auto download_result = common_download_model(tmp_model, opts, true);
                    if (!download_result.model_path.empty()) {
                        // Find the model name for this preset
                        std::string id_str;
                        mp.get_option("LLAMA_ARG_ID", id_str);
                        std::string model_path;
                        mp.get_option("LLAMA_ARG_MODEL", model_path);
                        if (model_path.empty()) {
                            model_path = hf_repo;
                        }
                        std::string preset_name = id_str.empty() ? std::filesystem::path(model_path).filename().string() : id_str;
                        model_manager->set_model_path(preset_name, download_result.model_path);
                        SRV_INF("%s: resolved model '%s' HF repo '%s' to '%s'\n", __func__, preset_name.c_str(), hf_repo.c_str(), download_result.model_path.c_str());
                    }
                }
            }
        }

        // Restore base model path for loading (additional model resolution overwrote it)
        if (cli_has_model_presets && !cli_base_model_path.empty()) {
            params.model.path = cli_base_model_path;
        }

        // load the model
        LOG_INF("%s: loading model\n", __func__);

        if (server_models::is_child_server()) {
            ctx_server.on_sleeping_changed([&](bool sleeping) {
                server_models::notify_router_sleeping_state(sleeping);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            LOG_ERR("%s: exiting due to model loading error\n", __func__);
            return 1;
        }

        // For CLI -- separated models, autoload any unloaded models
        // Build a map from model name to its preset for applying preset-specific params
        std::map<std::string, const common_preset*> cli_preset_map;
        for (size_t i = 1; i < cli_load_result.model_presets.size(); i++) {
            const auto & mp = cli_load_result.model_presets[i];
            std::string id_str;
            mp.get_option("LLAMA_ARG_ID", id_str);
            std::string model_path;
            mp.get_option("LLAMA_ARG_MODEL", model_path);
            if (model_path.empty()) {
                mp.get_option("-m", model_path);
            }
            std::string preset_name = id_str.empty() ? std::filesystem::path(model_path).filename().string() : id_str;
            cli_preset_map[preset_name] = &mp;
        }

        if (cli_has_model_presets && params.models_autoload && model_manager) {
            std::vector<std::string> models_to_load;
            for (const auto & info : model_manager->get_all_meta()) {
                SRV_INF("autoload: checking model '%s' status=%d\n", info.name.c_str(), info.status);
                if (info.status == SERVER_MODEL_STATUS_UNLOADED && !info.name.empty()) {
                    models_to_load.push_back(info.name);
                }
            }
            SRV_INF("autoload: %zu models to load on startup\n", models_to_load.size());
            if (!models_to_load.empty()) {
                if ((int)models_to_load.size() > params.models_max) {
                    SRV_WRN("number of models to load on startup (%zu) exceeds models_max (%d), loading first %d\n",
                        models_to_load.size(), params.models_max, params.models_max);
                    models_to_load.resize(params.models_max);
                }
                for (const auto & model_name : models_to_load) {
                    SRV_INF("(startup) loading model %s\n", model_name.c_str());
                    // Apply preset-specific params if available
                    common_params model_params = params;
                    auto it = cli_preset_map.find(model_name);
                    if (it != cli_preset_map.end()) {
                        it->second->apply_to_params(model_params);
                    }
                    model_manager->load(model_name, ctx_server, model_params);
                }
            }
        }

        // Create model manager if multi-model mode is enabled (non-CLI -- case)
        if (multi_model_enabled && !cli_has_model_presets) {
            model_manager = std::make_unique<server_model_manager>(params.models_max, params.models_autoload);

            // Load additional models from presets
            common_preset_context ctx_preset(LLAMA_EXAMPLE_SERVER);
            common_presets cached_models = ctx_preset.load_from_cache();
            common_presets local_models;
            if (!params.models_dir.empty()) {
                local_models = ctx_preset.load_from_models_dir(params.models_dir);
            }
            common_presets custom_presets;
            if (!params.models_preset.empty()) {
                common_preset global;
                custom_presets = ctx_preset.load_from_ini(params.models_preset, global);
            }

            // Cascade and merge presets (same logic as router mode)
            common_presets final_presets;
            for (const auto & [name, preset] : cached_models) {
                final_presets[name] = preset;
            }
            for (const auto & [name, preset] : local_models) {
                final_presets[name] = preset;
            }
            for (const auto & [name, custom] : custom_presets) {
                if (final_presets.find(name) != final_presets.end()) {
                    common_preset & target = final_presets[name];
                    target.merge(custom);
                } else {
                    final_presets[name] = custom;
                }
            }

            // Create a base preset from CLI model params for merging
            common_preset base_preset;
            if (!params.model.path.empty()) {
                base_preset.set_option(ctx_preset, "LLAMA_ARG_MODEL", params.model.path);
            }

            // Apply base preset to all presets
            for (auto & [name, preset] : final_presets) {
                preset.merge(base_preset);
            }

            // Register additional models from presets
            std::vector<std::string> models_to_load;
            for (const auto & [name, preset] : final_presets) {
                // Skip the base model and empty names
                if (name == model_manager_base_model_name || name.empty()) {
                    continue;
                }

                std::string model_path;
                preset.get_option("LLAMA_ARG_MODEL", model_path);
                if (model_path.empty()) {
                    // Try alternate key
                    preset.get_option("-m", model_path);
                }
                if (model_path.empty()) {
                    SRV_WRN("presets '%s': no model path defined, skipping\n", name.c_str());
                    continue;
                }

                server_model_info info;
                info.name = name;
                info.model_path = model_path;
                info.status = SERVER_MODEL_STATUS_UNLOADED;
                info.last_used = 0;

                // Parse --alias and --tags from preset
                std::string alias_str;
                if (preset.get_option("LLAMA_ARG_ALIAS", alias_str)) {
                    std::stringstream ss(alias_str);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        token = string_strip(token);
                        if (!token.empty()) {
                            info.aliases.insert(token);
                        }
                    }
                }
                std::string tag_str;
                if (preset.get_option("LLAMA_ARG_TAGS", tag_str)) {
                    std::stringstream ss(tag_str);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        token = string_strip(token);
                        if (!token.empty()) {
                            info.tags.insert(token);
                        }
                    }
                }

                model_manager->add_model(std::move(info));

                // Check autoload
                std::string val;
                if (preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val)) {
                    if (common_arg_utils::is_truthy(val)) {
                        models_to_load.push_back(name);
                    }
                }
            }

            // Register the base model (non-router multi-model mode)
            if (!model_manager_base_model_name.empty()) {
                server_model_info base_info;
                base_info.name = model_manager_base_model_name;
                base_info.model_path = params.model.path;
                base_info.aliases = params.model_alias;
                base_info.tags = params.model_tags;
                base_info.status = SERVER_MODEL_STATUS_LOADED;
                base_info.last_used = ggml_time_ms();
                model_manager->add_model(std::move(base_info));
            }

            // Log available models
            SRV_INF("Available models (%zu)\n", model_manager->get_all_meta().size());
            for (const auto & info : model_manager->get_all_meta()) {
                SRV_INF("  %s (%s)\n", info.name.c_str(),
                    info.status == SERVER_MODEL_STATUS_LOADED ? "loaded" :
                    info.status == SERVER_MODEL_STATUS_LOADING ? "loading" : "unloaded");
            }

            // Load autoload models (respecting models_max)
            // If --models-autoload is true, also load all registered models that aren't already loaded
            if (params.models_autoload) {
                for (const auto & info : model_manager->get_all_meta()) {
                    if (info.status == SERVER_MODEL_STATUS_UNLOADED && !info.name.empty()) {
                        models_to_load.push_back(info.name);
                    }
                }
            }
            if (!models_to_load.empty()) {
                if ((int)models_to_load.size() > params.models_max) {
                    SRV_WRN("number of models to load on startup (%zu) exceeds models_max (%d), loading first %d\n",
                        models_to_load.size(), params.models_max, params.models_max);
                    models_to_load.resize(params.models_max);
                }
                for (const auto & model_name : models_to_load) {
                    SRV_INF("(startup) loading model %s\n", model_name.c_str());
                    // Apply preset-specific params if available
                    common_params model_params = params;
                    auto preset_it = final_presets.find(model_name);
                    if (preset_it != final_presets.end()) {
                        preset_it->second.apply_to_params(model_params);
                    }
                    model_manager->load(model_name, ctx_server, model_params);
                }
            }

            // Wire up model_manager pointer to routes
            routes.set_model_manager(model_manager.get());

            // Register /models/load and /models/unload endpoints
            ctx_http.post("/models/load", ex_wrapper([model_mgr = model_manager.get(), &ctx_server, &params, model_manager_base_model_name](const server_http_req & req) -> server_http_res_ptr {
                auto res = std::make_unique<server_http_res>();
                json body = json::parse(req.body);
                std::string name = json_value(body, "model", std::string());
                std::string path = json_value(body, "path", std::string());

                if (name.empty() && path.empty()) {
                    res->status = 400;
                    res->data = safe_json_to_str({{"error", format_error_response("model name or path is required", ERROR_TYPE_INVALID_REQUEST)}});
                    return res;
                }

                // If a path is provided, register the model first
                if (!path.empty() && !model_mgr->has_model(name)) {
                    server_model_info info;
                    info.name = name;
                    info.model_path = path;
                    info.status = SERVER_MODEL_STATUS_UNLOADED;
                    model_mgr->add_model(std::move(info));
                }

                if (!model_mgr->has_model(name)) {
                    res->status = 404;
                    res->data = safe_json_to_str({{"error", format_error_response("model not found", ERROR_TYPE_NOT_FOUND)}});
                    return res;
                }

                // Check if already loaded
                auto meta = model_mgr->get_meta(name);
                if (meta.has_value() && meta->is_ready()) {
                    res->status = 400;
                    res->data = safe_json_to_str({{"error", format_error_response("model already loaded", ERROR_TYPE_INVALID_REQUEST)}});
                    return res;
                }

                // For arbitrary path loads, update params
                common_params load_params = params;
                if (!path.empty()) {
                    load_params.model.path = path;
                }

                // Load the model via model manager (handles LRU eviction)
                model_mgr->load(name, ctx_server, load_params);
                res_ok(res, {{"success", true}});
                return res;
            }));

            ctx_http.post("/models/unload", ex_wrapper([model_mgr2 = model_manager.get(), &ctx_server, model_manager_base_model_name](const server_http_req & req) -> server_http_res_ptr {
                auto res = std::make_unique<server_http_res>();
                json body = json::parse(req.body);
                std::string name = json_value(body, "model", std::string());

                if (name.empty()) {
                    res->status = 400;
                    res->data = safe_json_to_str({{"error", format_error_response("model name is required", ERROR_TYPE_INVALID_REQUEST)}});
                    return res;
                }

                if (!model_mgr2->has_model(name)) {
                    res->status = 404;
                    res->data = safe_json_to_str({{"error", format_error_response("model not found", ERROR_TYPE_NOT_FOUND)}});
                    return res;
                }

                // Don't allow unloading the base model
                if (name == model_manager_base_model_name) {
                    res->status = 400;
                    res->data = safe_json_to_str({{"error", format_error_response("base model cannot be unloaded", ERROR_TYPE_INVALID_REQUEST)}});
                    return res;
                }

                model_mgr2->unload(name, ctx_server);
                res_ok(res, {{"success", true}});
                return res;
            }));

            // Update /models endpoint for multi-model listing
            routes.get_models = ex_wrapper([model_mgr = model_manager.get(), &routes](const server_http_req & req) -> server_http_res_ptr {
                auto res = std::make_unique<server_http_res>();

                // For multi-model mode, return both available and loaded models
                if (model_mgr) {
                    json models_json = json::array();
                    std::time_t t = std::time(0);

                    auto all_models = model_mgr->get_all_meta();
                    for (const auto & info : all_models) {
                        json status {
                            {"value", server_model_status_to_string(info.status)},
                        };
                        if (info.status == SERVER_MODEL_STATUS_LOADED) {
                            status["args"] = {{"-m", info.model_path}};
                        }
                        models_json.push_back(json {
                            {"id", info.name},
                            {"aliases", info.aliases},
                            {"tags", info.tags},
                            {"object", "model"},
                            {"owned_by", "llamacpp"},
                            {"created", t},
                            {"status", status},
                        });
                    }

                    res_ok(res, {{"data", models_json}, {"object", "list"}});
                } else {
                    // Single model mode: use existing handler
                    return routes.get_models(req);
                }
                return res;
            });
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        LOG_INF("%s: model loaded\n", __func__);

        shutdown_handler = [&](int) {
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // TODO: refactor in common/console
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        LOG_INF("%s: router server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
        LOG_INF("%s: NOTE: router mode is experimental\n", __func__);
        LOG_INF("%s:       it is not recommended to use this mode in untrusted environments\n", __func__);
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        LOG_INF("%s: server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
        LOG_INF("%s: starting the main loop...\n", __func__);

        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (server_models::is_child_server()) {
            monitor_thread = server_models::setup_child_server(shutdown_handler);
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}

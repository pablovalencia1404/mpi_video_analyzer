#include <filesystem>
#include <cstdlib>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

#include "rescue/mpi_runtime.hpp"
#include "rescue/types.hpp"

namespace {

std::filesystem::path resolve_project_root(char** argv) {
    if (const char* root_env = std::getenv("RESCUE_PROJECT_ROOT"); root_env != nullptr && root_env[0] != '\0') {
        return root_env;
    }

    std::error_code error_code;
    const std::filesystem::path executable = std::filesystem::absolute(argv[0], error_code);
    if (error_code) {
        return std::filesystem::current_path();
    }

    const std::filesystem::path executable_dir = executable.parent_path();
    if (executable_dir.has_filename() && executable_dir.filename() == "build") {
        return executable_dir.parent_path();
    }

    return executable_dir;
}

std::string build_library_path(const std::filesystem::path& root) {
    const std::string existing = std::getenv("LD_LIBRARY_PATH") != nullptr ? std::getenv("LD_LIBRARY_PATH") : "";

    const std::filesystem::path candidates[] = {
        root / ".venv/lib/python3.12/site-packages/onnxruntime/capi",
        root / ".venv/lib/python3.12/site-packages/nvidia/cu13/lib",
        root / ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
        root / ".venv/lib/python3.12/site-packages/nvidia/cusparselt/lib",
        root / ".venv/lib/python3.12/site-packages/nvidia/nccl/lib",
        root / ".venv/lib64/python3.12/site-packages/onnxruntime/capi",
        root / ".venv/lib64/python3.12/site-packages/nvidia/cu13/lib",
        root / ".venv/lib64/python3.12/site-packages/nvidia/cudnn/lib",
        root / ".venv/lib64/python3.12/site-packages/nvidia/cusparselt/lib",
        root / ".venv/lib64/python3.12/site-packages/nvidia/nccl/lib",
    };

    std::string merged;
    for (const auto& candidate : candidates) {
        if (!std::filesystem::exists(candidate)) {
            continue;
        }

        if (!merged.empty()) {
            merged.push_back(':');
        }

        merged += candidate.string();
    }

    if (!existing.empty()) {
        if (!merged.empty()) {
            merged.push_back(':');
        }

        merged += existing;
    }

    return merged;
}

void ensure_runtime_library_path(char** argv) {
    if (std::getenv("RESCUE_LIBRARY_PATH_READY") != nullptr) {
        return;
    }

    const std::filesystem::path project_root = resolve_project_root(argv);
    const std::string library_path = build_library_path(project_root);
    if (library_path.empty()) {
        setenv("RESCUE_PROJECT_ROOT", project_root.c_str(), 1);
        setenv("RESCUE_LIBRARY_PATH_READY", "1", 1);
        return;
    }

    if (setenv("RESCUE_PROJECT_ROOT", project_root.c_str(), 1) != 0) {
        throw std::runtime_error("Failed to set RESCUE_PROJECT_ROOT.");
    }

    const char* existing = std::getenv("LD_LIBRARY_PATH");
    if (existing != nullptr && library_path == existing) {
        setenv("RESCUE_LIBRARY_PATH_READY", "1", 1);
        return;
    }

    if (setenv("LD_LIBRARY_PATH", library_path.c_str(), 1) != 0) {
        throw std::runtime_error("Failed to update LD_LIBRARY_PATH.");
    }

    if (setenv("RESCUE_LIBRARY_PATH_READY", "1", 1) != 0) {
        throw std::runtime_error("Failed to set runtime bootstrap guard.");
    }

    std::error_code error_code;
    const std::filesystem::path self_path = std::filesystem::read_symlink("/proc/self/exe", error_code);
    const std::string executable = error_code ? std::filesystem::absolute(argv[0]).string() : self_path.string();

    execv(executable.c_str(), argv);
    throw std::runtime_error("Failed to relaunch process with updated LD_LIBRARY_PATH.");
}

void print_usage() {
    std::cerr
        << "Usage: rescue_video_analyzer --input <video> [--output-dir dir] [--batch-size N]\n"
        << "                             [--resize-width px] [--edge-threshold N]\n"
        << "                             [--model path] [--score-threshold x]\n"
        << "                             [--nms-threshold x] [--top-k N]\n"
        << "                             [--scheduler dynamic|static-contiguous|static-round-robin]\n"
        << "                             [--no-annotated-video]\n"
        << "                             (legacy aliases: --yolo-model, --yolo-conf, --yolo-nms)\n";
}

rescue::SchedulerMode parse_scheduler_mode(const std::string& value) {
    if (value == "dynamic") {
        return rescue::SchedulerMode::Dynamic;
    }
    if (value == "static-contiguous") {
        return rescue::SchedulerMode::StaticContiguous;
    }
    if (value == "static-round-robin") {
        return rescue::SchedulerMode::StaticRoundRobin;
    }

    throw std::runtime_error(
        "Unknown scheduler mode: " + value +
        ". Expected dynamic, static-contiguous or static-round-robin.");
}

rescue::AppConfig parse_args(int argc, char** argv) {
    rescue::AppConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--input") {
            config.input_path = require_value("--input");
        } else if (arg == "--output-dir") {
            config.output_dir = require_value("--output-dir");
        } else if (arg == "--batch-size") {
            config.batch_size = std::stoi(require_value("--batch-size"));
        } else if (arg == "--resize-width") {
            config.resize_width = std::stoi(require_value("--resize-width"));
        } else if (arg == "--edge-threshold") {
            config.edge_threshold = std::stoi(require_value("--edge-threshold"));
        } else if (arg == "--model" || arg == "--person-model" || arg == "--yolo-model") {
            config.model_path = require_value(arg.c_str());
        } else if (arg == "--score-threshold" || arg == "--person-score" || arg == "--yolo-conf") {
            config.score_threshold = std::stod(require_value(arg.c_str()));
        } else if (arg == "--nms-threshold" || arg == "--person-nms" || arg == "--yolo-nms") {
            config.nms_threshold = std::stod(require_value(arg.c_str()));
        } else if (arg == "--top-k" || arg == "--person-top-k" || arg == "--yolo-top-k") {
            config.top_k = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--scheduler") {
            config.scheduler_mode = parse_scheduler_mode(require_value("--scheduler"));
        } else if (arg == "--no-annotated-video") {
            config.generate_annotated_video = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.input_path.empty()) {
        throw std::runtime_error("The --input argument is required.");
    }

    if (!std::filesystem::exists(config.input_path)) {
        throw std::runtime_error("Input video does not exist: " + config.input_path);
    }

    if (config.batch_size <= 0 || config.resize_width <= 0 || config.edge_threshold < 0) {
        throw std::runtime_error("Invalid numeric configuration.");
    }

    if (config.model_path.empty()) {
        throw std::runtime_error("The --model argument is required.");
    }

    if (!std::filesystem::exists(config.model_path)) {
        throw std::runtime_error("Model file does not exist: " + config.model_path);
    }

    if (config.score_threshold < 0.0 || config.score_threshold > 1.0) {
        throw std::runtime_error("--score-threshold must be in [0, 1].");
    }

    if (config.nms_threshold < 0.0 || config.nms_threshold > 1.0) {
        throw std::runtime_error("--nms-threshold must be in [0, 1].");
    }

    if (config.top_k <= 0) {
        throw std::runtime_error("--top-k must be > 0.");
    }

    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        ensure_runtime_library_path(argv);
        const auto config = parse_args(argc, argv);
        return rescue::run_application(argc, argv, config);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        print_usage();
        return 1;
    }
}

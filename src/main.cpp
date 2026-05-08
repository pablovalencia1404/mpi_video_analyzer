#include <filesystem>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

#ifdef RESCUE_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

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
        std::filesystem::path("/usr/lib/wsl/lib"),
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
        << "                             [--processing-width px] [--processing-height px]\n"
        << "                             [--gpu-device auto|N]\n"
        << "                             [--gpu-backend auto|cuda|opencl]\n"
        << "                             [--resize-width px] [--edge-threshold N]\n"
        << "                             [--model path] [--score-threshold x]\n"
        << "                             [--nms-threshold x] [--top-k N]\n"
        << "                             [--scheduler dynamic|static-contiguous|static-round-robin]\n"
        << "                             [--no-annotated-video]\n"
        << "                             (processing width/height default to 1920x1080)\n"
        << "                             (legacy aliases: --yolo-model, --yolo-conf, --yolo-nms)\n";
}

int parse_gpu_device_id(const std::string& value) {
    if (value == "auto") {
        return -1;
    }

    int parsed = 0;
    try {
        std::size_t consumed = 0;
        parsed = std::stoi(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid value for --gpu-device: " + value + ". Expected auto or a non-negative integer.");
    }

    if (parsed < 0) {
        throw std::runtime_error("Invalid value for --gpu-device: " + value + ". Expected auto or a non-negative integer.");
    }

    return parsed;
}

rescue::GpuBackend parse_gpu_backend(const std::string& value) {
    if (value == "auto") {
        return rescue::GpuBackend::Auto;
    }
    if (value == "cuda") {
        return rescue::GpuBackend::Cuda;
    }
    if (value == "opencl") {
        return rescue::GpuBackend::OpenCL;
    }

    throw std::runtime_error(
        "Unknown gpu backend: " + value +
        ". Expected auto, cuda, or opencl.");
}

bool has_help_flag(int argc, char** argv) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            return true;
        }
    }
    return false;
}

bool running_under_mpi_launcher() {
    const char* const mpi_env_vars[] = {
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMI_RANK",
        "PMIX_RANK",
        "MPI_LOCALRANKID",
    };

    for (const char* env_var : mpi_env_vars) {
        const char* value = std::getenv(env_var);
        if (value != nullptr && value[0] != '\0') {
            return true;
        }
    }

    return false;
}

#ifdef RESCUE_ENABLE_CUDA
int visible_cuda_device_count() {
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice) {
        return 0;
    }
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to query visible CUDA devices: ") + cudaGetErrorString(error));
    }
    return count;
}
#endif

std::string resolve_executable_path(char** argv) {
    std::error_code error_code;
    const std::filesystem::path self_path = std::filesystem::read_symlink("/proc/self/exe", error_code);
    if (!error_code) {
        return self_path.string();
    }

    return std::filesystem::absolute(argv[0]).string();
}

void ensure_default_mpi_world(int argc, char** argv, const rescue::AppConfig& config) {
    if (has_help_flag(argc, argv) ||
        running_under_mpi_launcher() ||
        std::getenv("RESCUE_MPI_AUTOSTARTED") != nullptr) {
        return;
    }

    int cuda_count = 0;
#ifdef RESCUE_ENABLE_CUDA
    cuda_count = visible_cuda_device_count();
#endif

    int world_size = 2;
    if (config.gpu_backend == rescue::GpuBackend::Cuda) {
        if (cuda_count <= 0) {
            throw std::runtime_error("No visible CUDA devices were found, cannot auto-launch MPI.");
        }
        world_size = cuda_count + 1;
    } else if (config.gpu_backend == rescue::GpuBackend::OpenCL) {
        world_size = 2; // 1 master + 1 OpenCL worker
    } else if (config.gpu_backend == rescue::GpuBackend::Auto) {
        world_size = cuda_count > 0 ? cuda_count + 2 : 2;
    }

    const std::string executable = resolve_executable_path(argv);

    if (setenv("RESCUE_MPI_AUTOSTARTED", "1", 1) != 0) {
        throw std::runtime_error("Failed to set RESCUE_MPI_AUTOSTARTED.");
    }

    std::vector<std::string> mpirun_args;
    mpirun_args.reserve(static_cast<std::size_t>(argc) + 4);
    mpirun_args.push_back("mpirun");
    mpirun_args.push_back("-np");
    mpirun_args.push_back(std::to_string(world_size));
    mpirun_args.push_back(executable);
    for (int index = 1; index < argc; ++index) {
        mpirun_args.push_back(argv[index]);
    }

    std::vector<char*> exec_argv;
    exec_argv.reserve(mpirun_args.size() + 1);
    for (auto& arg : mpirun_args) {
        exec_argv.push_back(arg.data());
    }
    exec_argv.push_back(nullptr);

    execvp("mpirun", exec_argv.data());
    throw std::runtime_error("Failed to launch mpirun automatically. Ensure OpenMPI is installed and available in PATH.");
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
        } else if (arg == "--processing-width") {
            config.processing_width = std::stoi(require_value("--processing-width"));
        } else if (arg == "--processing-height") {
            config.processing_height = std::stoi(require_value("--processing-height"));
        } else if (arg == "--gpu-device") {
            config.gpu_device_id = parse_gpu_device_id(require_value("--gpu-device"));
        } else if (arg == "--gpu-backend") {
            config.gpu_backend = parse_gpu_backend(require_value("--gpu-backend"));
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

    if (config.batch_size <= 0 || config.processing_width < 0 || config.processing_height < 0 ||
        config.resize_width <= 0 || config.edge_threshold < 0) {
        throw std::runtime_error("Invalid numeric configuration.");
    }

    const bool native_processing_size =
        config.processing_width == 0 && config.processing_height == 0;
    const bool explicit_processing_size =
        config.processing_width > 0 && config.processing_height > 0;
    if (!native_processing_size && !explicit_processing_size) {
        throw std::runtime_error(
            "--processing-width and --processing-height must both be omitted/0 or both be positive.");
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
        const char* home = std::getenv("HOME");
        if (home != nullptr) {
            std::filesystem::path cache_dir = std::filesystem::path(home) / ".cache" / "opencv";
            std::error_code ec;
            std::filesystem::create_directories(cache_dir, ec);
            setenv("OPENCV_OCL4DNN_CONFIG_PATH", cache_dir.string().c_str(), 0);
        }

        ensure_runtime_library_path(argv);
        const auto config = parse_args(argc, argv);
        ensure_default_mpi_world(argc, argv, config);
        return rescue::run_application(argc, argv, config);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        print_usage();
        return 1;
    }
}

import sys

with open('src/mpi_runtime.cpp', 'r') as f:
    content = f.read()

# Replace includes
content = content.replace('#include "rescue/gpu_preprocess.hpp"', '''
#ifdef RESCUE_ENABLE_CUDA
#include "rescue/gpu_preprocess_cuda.hpp"
#endif
#include "rescue/gpu_preprocess_opencl.hpp"
''')

content = content.replace('#include <cuda_runtime.h>', '''
#ifdef RESCUE_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
''')

# Replace cuda_check
content = content.replace('void cuda_check(cudaError_t error, const char* call) {', '''
#ifdef RESCUE_ENABLE_CUDA
void cuda_check(cudaError_t error, const char* call) {''')
content = content.replace('    }\n}\n\nint to_mpi_count', '''    }\n}\n#endif\n\nint to_mpi_count''')

# Replace activate_worker_gpu
content = content.replace('WorkerGpuContext activate_worker_gpu(const AppConfig& config, const LocalTopology& topology) {', '''
#ifdef RESCUE_ENABLE_CUDA
WorkerGpuContext activate_worker_gpu_cuda(const AppConfig& config, const LocalTopology& topology) {''')

content = content.replace('    context.processor_name = topology.processor_name;\n    return context;\n}\n\ntemplate <typename T>', '''    context.processor_name = topology.processor_name;
    return context;
}
#endif

WorkerGpuContext activate_worker_gpu_opencl(const AppConfig& config, const LocalTopology& topology) {
    WorkerGpuContext context;
    context.device_id = 0;
    context.visible_device_count = 1;
    context.local_worker_rank = topology.local_worker_rank;
    context.local_worker_count = std::max(1, topology.local_worker_count);
    context.shared_device = false;
    context.binding_mode = "auto(opencl)";
    context.device_name = "OpenCL Device";
    context.processor_name = topology.processor_name;
    return context;
}

template <typename T>''')

# Replace run_worker
run_worker_start = content.find('void run_worker(int rank, int world_size, const AppConfig& config, const LocalTopology& local_topology) {')
run_worker_end = content.find('}\n\n}  // namespace', run_worker_start) + 1

original_run_worker = content[run_worker_start:run_worker_end]

templated_run_worker = original_run_worker.replace(
    'void run_worker(int rank, int world_size, const AppConfig& config, const LocalTopology& local_topology) {',
    '''template <typename ArtifactType, typename DetectorType, typename PreprocessFunc>
void run_worker_impl(int rank, int world_size, const AppConfig& config, const LocalTopology& local_topology,
                     const WorkerGpuContext& gpu_context, DetectorType* detector_ptr, PreprocessFunc preprocess) {'''
)
templated_run_worker = templated_run_worker.replace(
    '''    const auto gpu_context = activate_worker_gpu(config, local_topology);
    AppConfig worker_config = config;
    worker_config.gpu_device_id = gpu_context.device_id;

    PeopleDetector detector(worker_config);
    const cv::Size detector_input_size = detector.input_size();''',
    '''    const cv::Size detector_input_size = detector_ptr->input_size();'''
)
templated_run_worker = templated_run_worker.replace('detector.backend_name()', 'detector_ptr->backend_name()')
templated_run_worker = templated_run_worker.replace('preprocess_frame_cuda', 'preprocess')
templated_run_worker = templated_run_worker.replace('detector.detect', 'detector_ptr->detect')

dispatcher_code = '''
void run_worker(int rank, int world_size, const AppConfig& config, const LocalTopology& local_topology) {
    bool use_cuda = false;
#ifdef RESCUE_ENABLE_CUDA
    if (config.gpu_backend == GpuBackend::Auto || config.gpu_backend == GpuBackend::Cuda) {
        int count = 0;
        cudaGetDeviceCount(&count);
        if (count > 0 || config.gpu_backend == GpuBackend::Cuda) {
            use_cuda = true;
        }
    }
#else
    if (config.gpu_backend == GpuBackend::Cuda) {
        throw std::runtime_error("CUDA backend requested but not compiled.");
    }
#endif

    if (use_cuda) {
#ifdef RESCUE_ENABLE_CUDA
        const auto gpu_context = activate_worker_gpu_cuda(config, local_topology);
        AppConfig worker_config = config;
        worker_config.gpu_device_id = gpu_context.device_id;
        auto detector = create_cuda_detector(worker_config);
        run_worker_impl<GpuFrameArtifacts, CudaPeopleDetector>(
            rank, world_size, worker_config, local_topology, gpu_context, detector.get(), preprocess_frame_cuda);
#endif
    } else {
        const auto gpu_context = activate_worker_gpu_opencl(config, local_topology);
        AppConfig worker_config = config;
        worker_config.gpu_device_id = gpu_context.device_id;
        auto detector = create_opencl_detector(worker_config);
        run_worker_impl<OpenCLFrameArtifacts, OpenCLPeopleDetector>(
            rank, world_size, worker_config, local_topology, gpu_context, detector.get(), preprocess_frame_opencl);
    }
}
'''

content = content[:run_worker_start] + templated_run_worker + '\n' + dispatcher_code + content[run_worker_end:]

with open('src/mpi_runtime.cpp', 'w') as f:
    f.write(content)

print("Rewrote src/mpi_runtime.cpp")

#include "rescue/mpi_runtime.hpp"


#ifdef RESCUE_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <mpi.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "rescue/detector.hpp"

#ifdef RESCUE_ENABLE_CUDA
#include "rescue/gpu_preprocess_cuda.hpp"
#endif
#include "rescue/gpu_preprocess_opencl.hpp"

#include "rescue/output_writer.hpp"
#include "rescue/video_io.hpp"

namespace rescue {

namespace {

constexpr int kTagBatchHeader = 100;
constexpr int kTagBatchPayload = 101;
constexpr int kTagResultHeader = 200;
constexpr int kTagResultPayload = 201;

struct LocalTopology {
    int local_rank = 0;
    int local_size = 1;
    int local_worker_rank = 0;
    int local_worker_count = 1;
    std::string processor_name;
};

struct WorkerGpuContext {
    int device_id = 0;
    int visible_device_count = 0;
    int local_worker_rank = 0;
    int local_worker_count = 1;
    bool shared_device = false;
    std::string binding_mode = "auto(local-worker-rank)";
    std::string device_name;
    std::string processor_name;
};

struct ResultHeader {
    std::int32_t start_frame = 0;
    std::int32_t frame_count = 0;
};

struct SerializedFrameBatch {
    BatchHeader header;
    std::vector<std::uint8_t> payload;
};


#ifdef RESCUE_ENABLE_CUDA
void cuda_check(cudaError_t error, const char* call) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + call + ": " + cudaGetErrorString(error));
    }
}
#endif

int to_mpi_count(std::size_t value, const char* context) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(context) + " exceeds MPI count limits.");
    }

    return static_cast<int>(value);
}

std::size_t bytes_per_frame(const BatchHeader& header) {
    if (header.frame_width <= 0 || header.frame_height <= 0) {
        throw std::runtime_error("Received an invalid frame size in the MPI batch header.");
    }

    if (header.frame_type != CV_8UC3) {
        throw std::runtime_error("The MPI frame transport currently expects CV_8UC3 frames.");
    }

    return static_cast<std::size_t>(header.frame_width) *
           static_cast<std::size_t>(header.frame_height) *
           static_cast<std::size_t>(CV_ELEM_SIZE(header.frame_type));
}

SerializedFrameBatch serialize_frame_batch(const BatchHeader& header, const std::vector<cv::Mat>& frames) {
    if (frames.empty()) {
        throw std::runtime_error("Cannot serialize an empty frame batch.");
    }

    SerializedFrameBatch batch;
    batch.header = header;
    batch.header.frame_width = frames.front().cols;
    batch.header.frame_height = frames.front().rows;
    batch.header.frame_type = frames.front().type();

    const cv::Size expected_size(batch.header.frame_width, batch.header.frame_height);
    const int expected_type = batch.header.frame_type;
    for (const auto& frame : frames) {
        if (frame.size() != expected_size) {
            throw std::runtime_error("A frame batch contains mixed frame sizes.");
        }

        if (frame.type() != expected_type) {
            throw std::runtime_error("A frame batch contains mixed frame types.");
        }
    }

    const std::size_t frame_bytes = bytes_per_frame(batch.header);
    batch.payload.resize(frame_bytes * frames.size());

    for (std::size_t index = 0; index < frames.size(); ++index) {
        const cv::Mat contiguous = frames[index].isContinuous() ? frames[index] : frames[index].clone();
        std::memcpy(batch.payload.data() + (index * frame_bytes), contiguous.data, frame_bytes);
    }

    return batch;
}

std::vector<std::uint8_t> receive_frame_payload(const BatchHeader& header) {
    const std::size_t frame_bytes = bytes_per_frame(header);
    const std::size_t total_bytes = frame_bytes * static_cast<std::size_t>(header.frame_count);
    std::vector<std::uint8_t> payload(total_bytes);
    if (!payload.empty()) {
        MPI_Recv(
            payload.data(),
            to_mpi_count(payload.size(), "frame payload"),
            MPI_BYTE,
            0,
            kTagBatchPayload,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }

    return payload;
}

std::vector<cv::Mat> materialize_frame_views(const BatchHeader& header, std::vector<std::uint8_t>& payload) {
    const std::size_t frame_bytes = bytes_per_frame(header);
    const std::size_t expected_bytes = frame_bytes * static_cast<std::size_t>(header.frame_count);
    if (payload.size() != expected_bytes) {
        throw std::runtime_error("The received frame payload size does not match the batch header.");
    }

    std::vector<cv::Mat> frames;
    frames.reserve(static_cast<std::size_t>(header.frame_count));
    for (std::int32_t index = 0; index < header.frame_count; ++index) {
        frames.emplace_back(
            header.frame_height,
            header.frame_width,
            header.frame_type,
            payload.data() + (static_cast<std::size_t>(index) * frame_bytes));
    }

    return frames;
}

LocalTopology query_local_topology(int world_rank) {
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);

    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    std::vector<int> local_world_ranks(static_cast<std::size_t>(local_size), -1);
    MPI_Allgather(&world_rank, 1, MPI_INT, local_world_ranks.data(), 1, MPI_INT, local_comm);

    int local_worker_rank = 0;
    int local_worker_count = 0;
    for (int index = 0; index < local_size; ++index) {
        if (local_world_ranks[static_cast<std::size_t>(index)] == 0) {
            continue;
        }

        if (index < local_rank) {
            ++local_worker_rank;
        }
        ++local_worker_count;
    }

    MPI_Comm_free(&local_comm);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int processor_name_length = 0;
    MPI_Get_processor_name(processor_name, &processor_name_length);

    LocalTopology topology;
    topology.local_rank = local_rank;
    topology.local_size = local_size;
    topology.local_worker_rank = world_rank == 0 ? -1 : local_worker_rank;
    topology.local_worker_count = std::max(0, local_worker_count);
    topology.processor_name.assign(processor_name, static_cast<std::size_t>(processor_name_length));
    return topology;
}


#ifdef RESCUE_ENABLE_CUDA
WorkerGpuContext activate_worker_gpu_cuda(const AppConfig& config, const LocalTopology& topology) {
    int visible_device_count = 0;
    cuda_check(cudaGetDeviceCount(&visible_device_count), "cudaGetDeviceCount");
    if (visible_device_count <= 0) {
        throw std::runtime_error(
            "No visible CUDA devices were found for this worker. Ensure each MPI worker runs on a node with NVIDIA GPUs.");
    }

    const bool fixed_device = config.gpu_device_id >= 0;
    const int device_id = fixed_device ? config.gpu_device_id : (topology.local_worker_rank % visible_device_count);
    if (device_id < 0 || device_id >= visible_device_count) {
        throw std::runtime_error(
            "Requested GPU device " + std::to_string(config.gpu_device_id) +
            " is not available on node " + topology.processor_name +
            ". Visible CUDA devices: " + std::to_string(visible_device_count) + ".");
    }

    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");

    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device_id), "cudaGetDeviceProperties");

    WorkerGpuContext context;
    context.device_id = device_id;
    context.visible_device_count = visible_device_count;
    context.local_worker_rank = topology.local_worker_rank;
    context.local_worker_count = std::max(1, topology.local_worker_count);
    context.shared_device = topology.local_worker_count > visible_device_count;
    context.binding_mode = fixed_device ? "fixed" : "auto(local-worker-rank)";
    context.device_name = properties.name;
    context.processor_name = topology.processor_name;
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

template <typename T>
MPI_Datatype create_plain_struct_type();

template <>
MPI_Datatype create_plain_struct_type<BatchHeader>() {
    MPI_Datatype type;
    MPI_Type_contiguous(static_cast<int>(sizeof(BatchHeader)), MPI_BYTE, &type);
    MPI_Type_commit(&type);
    return type;
}

template <>
MPI_Datatype create_plain_struct_type<ResultHeader>() {
    MPI_Datatype type;
    MPI_Type_contiguous(static_cast<int>(sizeof(ResultHeader)), MPI_BYTE, &type);
    MPI_Type_commit(&type);
    return type;
}

template <>
MPI_Datatype create_plain_struct_type<FrameResult>() {
    MPI_Datatype type;
    MPI_Type_contiguous(static_cast<int>(sizeof(FrameResult)), MPI_BYTE, &type);
    MPI_Type_commit(&type);
    return type;
}

template <>
MPI_Datatype create_plain_struct_type<FramePacket>() {
    MPI_Datatype type;
    MPI_Type_contiguous(static_cast<int>(sizeof(FramePacket)), MPI_BYTE, &type);
    MPI_Type_commit(&type);
    return type;
}

static_assert(std::is_trivially_copyable_v<BatchHeader>);
static_assert(std::is_trivially_copyable_v<ResultHeader>);
static_assert(std::is_trivially_copyable_v<FrameResult>);
static_assert(std::is_trivially_copyable_v<FramePacket>);

void send_stop_signal(int worker_rank, MPI_Datatype batch_type) {
    BatchHeader stop_header{};
    MPI_Send(&stop_header, 1, batch_type, worker_rank, kTagBatchHeader, MPI_COMM_WORLD);
}

void send_frame_batch(int worker_rank, const SerializedFrameBatch& batch, MPI_Datatype batch_type) {
    MPI_Send(&batch.header, 1, batch_type, worker_rank, kTagBatchHeader, MPI_COMM_WORLD);
    if (!batch.payload.empty()) {
        MPI_Send(
            batch.payload.data(),
            to_mpi_count(batch.payload.size(), "frame payload"),
            MPI_BYTE,
            worker_rank,
            kTagBatchPayload,
            MPI_COMM_WORLD);
    }
}

std::vector<FramePacket> receive_results(
    int worker_rank,
    MPI_Datatype result_header_type,
    MPI_Datatype frame_packet_type) {
    ResultHeader header{};
    MPI_Recv(&header, 1, result_header_type, worker_rank, kTagResultHeader, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<FramePacket> results(header.frame_count);
    if (header.frame_count > 0) {
        MPI_Recv(
            results.data(),
            header.frame_count,
            frame_packet_type,
            worker_rank,
            kTagResultPayload,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }
    return results;
}

std::vector<BatchHeader> build_batch_plan(int frame_count, int batch_size) {
    std::vector<BatchHeader> tasks;
    for (int next_frame = 0; next_frame < frame_count; next_frame += batch_size) {
        BatchHeader header{};
        header.start_frame = next_frame;
        header.frame_count = std::min(batch_size, frame_count - next_frame);
        tasks.push_back(header);
    }
    return tasks;
}

std::vector<std::vector<BatchHeader>> build_static_worker_queues(
    int world_size,
    int frame_count,
    const AppConfig& config) {
    std::vector<std::vector<BatchHeader>> worker_queues(static_cast<std::size_t>(world_size));
    const int worker_count = std::max(0, world_size - 1);
    if (worker_count == 0 || frame_count <= 0) {
        return worker_queues;
    }

    if (config.scheduler_mode == SchedulerMode::StaticContiguous) {
        int start_frame = 0;
        const int base_frames = frame_count / worker_count;
        const int extra_frames = frame_count % worker_count;

        for (int worker_rank = 1; worker_rank < world_size; ++worker_rank) {
            const int frames_for_worker = base_frames + (worker_rank <= extra_frames ? 1 : 0);
            const int end_frame = start_frame + frames_for_worker;
            for (int frame = start_frame; frame < end_frame; frame += config.batch_size) {
                BatchHeader header{};
                header.start_frame = frame;
                header.frame_count = std::min(config.batch_size, end_frame - frame);
                worker_queues[static_cast<std::size_t>(worker_rank)].push_back(header);
            }
            start_frame = end_frame;
        }
        return worker_queues;
    }

    const auto tasks = build_batch_plan(frame_count, config.batch_size);
    if (config.scheduler_mode == SchedulerMode::StaticRoundRobin) {
        for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
            const int worker_rank = 1 + static_cast<int>(task_index % static_cast<std::size_t>(worker_count));
            worker_queues[static_cast<std::size_t>(worker_rank)].push_back(tasks[task_index]);
        }
    }

    return worker_queues;
}

void run_master(int world_size, const AppConfig& config) {
    VideoReader reader(config.input_path);
    const auto& metadata = reader.metadata();
    const int worker_count = world_size - 1;
    const auto master_start = std::chrono::steady_clock::now();
    const cv::Size processing_size =
        config.processing_width > 0 && config.processing_height > 0
            ? cv::Size(config.processing_width, config.processing_height)
            : cv::Size(metadata.width, metadata.height);

    std::cout << "Master rank 0 scheduling " << worker_count
              << " worker(s)"
              << " | scheduler=" << scheduler_mode_name(config.scheduler_mode)
              << " | processing_size=" << processing_size.width << "x" << processing_size.height
              << '\n';
    if (config.gpu_device_id >= 0 && worker_count > 1) {
        std::cerr
            << "Warning: --gpu-device pins every worker to the same visible CUDA device index on its node. "
            << "Use the default auto mode to spread workers across a GPU farm."
            << '\n';
    }

    MPI_Datatype batch_type = create_plain_struct_type<BatchHeader>();
    MPI_Datatype result_header_type = create_plain_struct_type<ResultHeader>();
    MPI_Datatype frame_packet_type = create_plain_struct_type<FramePacket>();

    std::vector<FramePacket> all_packets;
    all_packets.reserve(metadata.frame_count);

    const auto static_worker_queues = build_static_worker_queues(world_size, metadata.frame_count, config);
    std::vector<std::size_t> worker_queue_positions(static_cast<std::size_t>(world_size), 0);
    const auto dynamic_tasks = build_batch_plan(metadata.frame_count, config.batch_size);
    std::size_t dynamic_task_index = 0;
    int active_workers = 0;

    auto dispatch_next_task = [&](int worker_rank) -> bool {
        BatchHeader task{};
        if (config.scheduler_mode == SchedulerMode::Dynamic) {
            if (dynamic_task_index >= dynamic_tasks.size()) {
                return false;
            }

            task = dynamic_tasks[dynamic_task_index];
            ++dynamic_task_index;
        } else {
            auto& queue_position = worker_queue_positions[static_cast<std::size_t>(worker_rank)];
            const auto& queue = static_worker_queues[static_cast<std::size_t>(worker_rank)];
            if (queue_position >= queue.size()) {
                return false;
            }

            task = queue[queue_position];
            ++queue_position;
        }

        std::vector<cv::Mat> frames;
        BatchHeader actual_header{};
        if (!reader.read_batch_from(task.start_frame, task.frame_count, actual_header, frames)) {
            throw std::runtime_error(
                "Master could not read frames starting at " + std::to_string(task.start_frame));
        }

        actual_header.processing_width = processing_size.width;
        actual_header.processing_height = processing_size.height;
        actual_header.fps = metadata.fps;

        const auto batch = serialize_frame_batch(actual_header, frames);
        send_frame_batch(worker_rank, batch, batch_type);
        return true;
    };

    for (int worker_rank = 1; worker_rank < world_size; ++worker_rank) {
        if (dispatch_next_task(worker_rank)) {
            ++active_workers;
        } else {
            send_stop_signal(worker_rank, batch_type);
        }
    }

    while (active_workers > 0) {
        MPI_Status status{};
        MPI_Probe(MPI_ANY_SOURCE, kTagResultHeader, MPI_COMM_WORLD, &status);
        const int worker_rank = status.MPI_SOURCE;

        auto worker_results = receive_results(worker_rank, result_header_type, frame_packet_type);
        all_packets.insert(all_packets.end(), worker_results.begin(), worker_results.end());

        if (!dispatch_next_task(worker_rank)) {
            send_stop_signal(worker_rank, batch_type);
            --active_workers;
        }
    }

    std::sort(
        all_packets.begin(),
        all_packets.end(),
        [](const FramePacket& lhs, const FramePacket& rhs) { return lhs.result.frame_index < rhs.result.frame_index; });

    const auto distributed_processing_end = std::chrono::steady_clock::now();
    OutputMetrics metrics;
    metrics.mpi_world_size = world_size;
    metrics.requested_workers = worker_count;
    metrics.processing_width = processing_size.width;
    metrics.processing_height = processing_size.height;
    metrics.scheduler_name = scheduler_mode_name(config.scheduler_mode);
    metrics.annotated_video_enabled = config.generate_annotated_video;
    metrics.distributed_processing_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(distributed_processing_end - master_start).count() / 1000.0;

    write_outputs(config.output_dir, config.input_path, metadata, all_packets, metrics);

    MPI_Type_free(&batch_type);
    MPI_Type_free(&result_header_type);
    MPI_Type_free(&frame_packet_type);
}

template <typename ArtifactType, typename DetectorType, typename PreprocessFunc>
void run_worker_impl(int rank, int world_size, const AppConfig& config, const LocalTopology& local_topology,
                     const WorkerGpuContext& gpu_context, DetectorType* detector_ptr, PreprocessFunc preprocess) {
    MPI_Datatype batch_type = create_plain_struct_type<BatchHeader>();
    MPI_Datatype result_header_type = create_plain_struct_type<ResultHeader>();
    MPI_Datatype frame_packet_type = create_plain_struct_type<FramePacket>();

    const cv::Size detector_input_size = detector_ptr->input_size();

    std::cout << "Rank " << rank << " detector backend=" << detector_ptr->backend_name()
              << " | model_input=" << detector_input_size.width << "x" << detector_input_size.height
              << " | scheduler=" << scheduler_mode_name(config.scheduler_mode)
              << " | gpu_device=" << gpu_context.device_id
              << " | gpu_name=" << gpu_context.device_name
              << " | gpu_binding=" << gpu_context.binding_mode
              << " | visible_cuda_devices=" << gpu_context.visible_device_count
              << " | local_worker_rank=" << gpu_context.local_worker_rank
              << " | local_workers_on_node=" << gpu_context.local_worker_count
              << " | shared_gpu=" << (gpu_context.shared_device ? "yes" : "no")
              << " | node=" << gpu_context.processor_name
              << '\n';
    bool warmup_done = false;

    while (true) {
        BatchHeader header{};
        MPI_Recv(&header, 1, batch_type, 0, kTagBatchHeader, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (header.frame_count == 0) {
            break;
        }

        if (header.frame_width <= 0 || header.frame_height <= 0 || header.frame_type != CV_8UC3) {
            throw std::runtime_error("Worker received an unsupported frame batch layout.");
        }

        const cv::Size processing_size =
            header.processing_width > 0 && header.processing_height > 0
                ? cv::Size(header.processing_width, header.processing_height)
                : cv::Size(header.frame_width, header.frame_height);

        const auto batch_receive_start = std::chrono::steady_clock::now();
        std::vector<std::uint8_t> payload = receive_frame_payload(header);
        const auto batch_receive_end = std::chrono::steady_clock::now();
        const double batch_receive_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(batch_receive_end - batch_receive_start).count() / 1000.0;

        if (!warmup_done) {
            const auto warmup_start = std::chrono::steady_clock::now();
            cv::Mat warmup_frame(
                std::max(1, processing_size.height),
                std::max(1, processing_size.width),
                CV_8UC3,
                cv::Scalar(114, 114, 114));
            const auto warmup_artifacts = preprocess(warmup_frame, detector_input_size, config.edge_threshold);
            (void)detector_ptr->detect(warmup_artifacts);
            const auto warmup_end = std::chrono::steady_clock::now();
            const double warmup_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(warmup_end - warmup_start).count() / 1000.0;
            std::clog << "Rank " << rank << " warmup_ms=" << warmup_ms
                      << " processing_size=" << processing_size.width << "x" << processing_size.height << '\n';
            warmup_done = true;
        }

        std::vector<cv::Mat> frames = materialize_frame_views(header, payload);

        std::vector<FramePacket> results;
        results.reserve(frames.size());
        double batch_preprocess_ms = 0.0;
        double batch_detection_ms = 0.0;
        std::int32_t batch_people_count = 0;

        for (std::size_t i = 0; i < frames.size(); ++i) {
            const auto frame_start = std::chrono::steady_clock::now();

            const auto preprocess_start = frame_start;
            cv::Mat processing_frame;
            if (frames[i].size() != processing_size) {
                cv::resize(frames[i], processing_frame, processing_size, 0.0, 0.0, cv::INTER_LINEAR);
            } else {
                processing_frame = frames[i];
            }

            const auto artifacts = preprocess(processing_frame, detector_input_size, config.edge_threshold);
            const auto preprocess_end = std::chrono::steady_clock::now();

            const auto detection_start = preprocess_end;
            const auto detections = detector_ptr->detect(artifacts);
            const auto detection_end = std::chrono::steady_clock::now();

            const double preprocess_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count() / 1000.0;
            const double detection_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(detection_end - detection_start).count() / 1000.0;
            const double receive_ms_per_frame = frames.empty() ? 0.0 : batch_receive_ms / static_cast<double>(frames.size());

            const double elapsed_ms = receive_ms_per_frame + preprocess_ms + detection_ms;
            const std::size_t total_detections = detections.size();
            const std::int32_t capped_detections = static_cast<std::int32_t>(
                std::min<std::size_t>(total_detections, kMaxDetectionsPerFrame));

            FramePacket packet{};
            packet.result.frame_index = header.start_frame + static_cast<std::int32_t>(i);
            packet.result.worker_rank = rank;
            packet.result.gpu_device_id = gpu_context.device_id;
            packet.result.people_count = capped_detections;
            packet.result.timestamp_ms = header.fps > 0.0
                ? static_cast<double>(packet.result.frame_index) * (1000.0 / header.fps)
                : 0.0;
            packet.result.mean_intensity = artifacts.mean_intensity;
            packet.result.edge_density = artifacts.edge_density;
            packet.result.read_ms = receive_ms_per_frame;
            packet.result.preprocess_ms = preprocess_ms;
            packet.result.detection_ms = detection_ms;
            packet.result.processing_ms = elapsed_ms;
            packet.detection_count = capped_detections;
            for (std::size_t detection_index = 0; detection_index < static_cast<std::size_t>(capped_detections); ++detection_index) {
                packet.detections[detection_index] = detections[detection_index];
            }
            results.push_back(packet);
            batch_preprocess_ms += preprocess_ms;
            batch_detection_ms += detection_ms;
            batch_people_count += capped_detections;
        }

        const double average_receive_ms = frames.empty() ? 0.0 : batch_receive_ms / static_cast<double>(frames.size());
        const double average_preprocess_ms = results.empty() ? 0.0 : batch_preprocess_ms / static_cast<double>(results.size());
        const double average_detection_ms = results.empty() ? 0.0 : batch_detection_ms / static_cast<double>(results.size());

        std::clog << "Rank " << rank
                  << " backend=" << detector_ptr->backend_name()
                  << " model_input=" << detector_input_size.width << "x" << detector_input_size.height
                  << " batch_start=" << header.start_frame
                  << " frames=" << results.size()
                  << " avg_receive_ms=" << average_receive_ms
                  << " avg_preprocess_ms=" << average_preprocess_ms
                  << " avg_detection_ms=" << average_detection_ms
                  << " detections=" << batch_people_count
                  << '\n';

        ResultHeader result_header{};
        result_header.start_frame = header.start_frame;
        result_header.frame_count = static_cast<std::int32_t>(results.size());
        MPI_Send(&result_header, 1, result_header_type, 0, kTagResultHeader, MPI_COMM_WORLD);
        MPI_Send(
            results.data(),
            static_cast<int>(results.size()),
            frame_packet_type,
            0,
            kTagResultPayload,
            MPI_COMM_WORLD);
    }

    MPI_Type_free(&batch_type);
    MPI_Type_free(&result_header_type);
    MPI_Type_free(&frame_packet_type);
}

void run_worker(int rank, int world_size, const AppConfig& config, const LocalTopology& local_topology) {
    bool use_cuda = false;
#ifdef RESCUE_ENABLE_CUDA
    if (config.gpu_backend == GpuBackend::Cuda) {
        use_cuda = true;
    } else if (config.gpu_backend == GpuBackend::Auto) {
        int count = 0;
        cudaGetDeviceCount(&count);
        if (count > 0) {
            if (config.gpu_device_id >= 0) {
                use_cuda = true;
            } else {
                use_cuda = (local_topology.local_worker_rank < count);
            }
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


}  // namespace

int run_application(int argc, char** argv, const AppConfig& config) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const auto local_topology = query_local_topology(world_rank);

    int exit_code = 0;
    bool aborted = false;
    try {
        if (world_size < 2) {
            throw std::runtime_error("Run with at least 2 MPI ranks: 1 master + 1 worker.");
        }

        if (world_rank == 0) {
            run_master(world_size, config);
            std::cout << "Results written to " << std::filesystem::path(config.output_dir).lexically_normal() << '\n';
        } else {
            run_worker(world_rank, world_size, config, local_topology);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Runtime error on rank " << world_rank << ": " << ex.what() << '\n';
        exit_code = 1;
        aborted = true;
        MPI_Abort(MPI_COMM_WORLD, exit_code);
    }

    if (!aborted) {
        MPI_Finalize();
    }
    return exit_code;
}

}  // namespace rescue

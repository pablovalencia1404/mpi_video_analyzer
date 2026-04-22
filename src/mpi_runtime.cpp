#include "rescue/mpi_runtime.hpp"

#include <mpi.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rescue/detector.hpp"
#include "rescue/gpu_preprocess.hpp"
#include "rescue/output_writer.hpp"
#include "rescue/video_io.hpp"

namespace rescue {

namespace {

constexpr int kTagBatchHeader = 100;
constexpr int kTagResultHeader = 200;
constexpr int kTagResultPayload = 201;

struct ResultHeader {
    std::int32_t start_frame = 0;
    std::int32_t frame_count = 0;
};

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

void send_batch(int worker_rank, const BatchHeader& header, MPI_Datatype batch_type) {
    MPI_Send(&header, 1, batch_type, worker_rank, kTagBatchHeader, MPI_COMM_WORLD);
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
    const int cuda_workers = world_size - 1;
    const auto master_start = std::chrono::steady_clock::now();

    std::cout << "Master rank 0 scheduling " << cuda_workers
              << " CUDA worker(s) on one shared GPU"
              << " | scheduler=" << scheduler_mode_name(config.scheduler_mode)
              << '\n';
    if (cuda_workers > 2) {
        std::cerr << "Warning: more than 2 CUDA workers share one GPU; this is contention analysis, not scale-out."
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
        if (config.scheduler_mode == SchedulerMode::Dynamic) {
            if (dynamic_task_index >= dynamic_tasks.size()) {
                return false;
            }

            send_batch(worker_rank, dynamic_tasks[dynamic_task_index], batch_type);
            ++dynamic_task_index;
            return true;
        }

        auto& queue_position = worker_queue_positions[static_cast<std::size_t>(worker_rank)];
        const auto& queue = static_worker_queues[static_cast<std::size_t>(worker_rank)];
        if (queue_position >= queue.size()) {
            return false;
        }

        send_batch(worker_rank, queue[queue_position], batch_type);
        ++queue_position;
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
    metrics.requested_cuda_workers = cuda_workers;
    metrics.scheduler_name = scheduler_mode_name(config.scheduler_mode);
    metrics.annotated_video_enabled = config.generate_annotated_video;
    metrics.distributed_processing_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(distributed_processing_end - master_start).count() / 1000.0;

    write_outputs(config.output_dir, config.input_path, metadata, all_packets, metrics);

    MPI_Type_free(&batch_type);
    MPI_Type_free(&result_header_type);
    MPI_Type_free(&frame_packet_type);
}

void run_worker(int rank, int world_size, const AppConfig& config) {
    MPI_Datatype batch_type = create_plain_struct_type<BatchHeader>();
    MPI_Datatype result_header_type = create_plain_struct_type<ResultHeader>();
    MPI_Datatype frame_packet_type = create_plain_struct_type<FramePacket>();

    VideoReader reader(config.input_path);
    PeopleDetector detector(config);
    const cv::Size detector_input_size = detector.input_size();

    if (config.resize_width != detector_input_size.width) {
        std::clog << "Rank " << rank
                  << " note: the bundled ONNX model uses a fixed "
                  << detector_input_size.width << "x" << detector_input_size.height
                  << " input, so --resize-width=" << config.resize_width
                  << " is ignored by the detector path." << '\n';
    }

    std::cout << "Rank " << rank << " detector backend=" << detector.backend_name()
              << " | model_input=" << detector_input_size.width << "x" << detector_input_size.height
              << " | scheduler=" << scheduler_mode_name(config.scheduler_mode)
              << " | gpu_active=yes"
              << " | shared_gpu=" << (world_size > 2 ? "yes" : "no") << '\n';

    {
        const auto warmup_start = std::chrono::steady_clock::now();
        cv::Mat warmup_frame(
            std::max(1, reader.metadata().height),
            std::max(1, reader.metadata().width),
            CV_8UC3,
            cv::Scalar(114, 114, 114));
        const auto warmup_artifacts = preprocess_frame_cuda(warmup_frame, detector_input_size, config.edge_threshold);
        (void)detector.detect(warmup_artifacts);
        const auto warmup_end = std::chrono::steady_clock::now();
        const double warmup_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(warmup_end - warmup_start).count() / 1000.0;
        std::clog << "Rank " << rank << " warmup_ms=" << warmup_ms << '\n';
    }

    while (true) {
        BatchHeader header{};
        MPI_Recv(&header, 1, batch_type, 0, kTagBatchHeader, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (header.frame_count == 0) {
            break;
        }

        BatchHeader actual_header{};
        std::vector<cv::Mat> frames;
        const auto batch_read_start = std::chrono::steady_clock::now();
        if (!reader.read_batch_from(header.start_frame, header.frame_count, actual_header, frames)) {
            throw std::runtime_error(
                "Worker " + std::to_string(rank) + " could not read frames starting at " +
                std::to_string(header.start_frame));
        }
        const auto batch_read_end = std::chrono::steady_clock::now();
        const double batch_read_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(batch_read_end - batch_read_start).count() / 1000.0;

        std::vector<FramePacket> results;
        results.reserve(frames.size());
        double batch_preprocess_ms = 0.0;
        double batch_detection_ms = 0.0;
        std::int32_t batch_people_count = 0;

        for (std::size_t i = 0; i < frames.size(); ++i) {
            const auto frame_start = std::chrono::steady_clock::now();

            const auto preprocess_start = frame_start;
            const auto artifacts = preprocess_frame_cuda(frames[i], detector_input_size, config.edge_threshold);
            const auto preprocess_end = std::chrono::steady_clock::now();

            const auto detection_start = preprocess_end;
            const auto detections = detector.detect(artifacts);
            const auto detection_end = std::chrono::steady_clock::now();

            const double preprocess_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count() / 1000.0;
            const double detection_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(detection_end - detection_start).count() / 1000.0;
            const double read_ms_per_frame = frames.empty() ? 0.0 : batch_read_ms / static_cast<double>(frames.size());

            const double elapsed_ms = read_ms_per_frame + preprocess_ms + detection_ms;
            const std::int32_t detected_people = static_cast<std::int32_t>(detections.size());

            FramePacket packet{};
            packet.result.frame_index = actual_header.start_frame + static_cast<std::int32_t>(i);
            packet.result.worker_rank = rank;
            packet.result.people_count = detected_people;
            packet.result.timestamp_ms = reader.metadata().fps > 0.0
                ? static_cast<double>(packet.result.frame_index) * (1000.0 / reader.metadata().fps)
                : 0.0;
            packet.result.mean_intensity = artifacts.mean_intensity;
            packet.result.edge_density = artifacts.edge_density;
            packet.result.read_ms = read_ms_per_frame;
            packet.result.preprocess_ms = preprocess_ms;
            packet.result.detection_ms = detection_ms;
            packet.result.processing_ms = elapsed_ms;
            packet.detection_count = static_cast<std::int32_t>(std::min<std::size_t>(detections.size(), kMaxDetectionsPerFrame));
            for (std::size_t detection_index = 0; detection_index < static_cast<std::size_t>(packet.detection_count); ++detection_index) {
                packet.detections[detection_index] = detections[detection_index];
            }
            results.push_back(packet);
            batch_preprocess_ms += preprocess_ms;
            batch_detection_ms += detection_ms;
            batch_people_count += detected_people;
        }

        const double average_read_ms = frames.empty() ? 0.0 : batch_read_ms / static_cast<double>(frames.size());
        const double average_preprocess_ms = results.empty() ? 0.0 : batch_preprocess_ms / static_cast<double>(results.size());
        const double average_detection_ms = results.empty() ? 0.0 : batch_detection_ms / static_cast<double>(results.size());

        std::clog << "Rank " << rank
                  << " backend=" << detector.backend_name()
                  << " model_input=" << detector_input_size.width << "x" << detector_input_size.height
                  << " batch_start=" << actual_header.start_frame
                  << " frames=" << results.size()
                  << " avg_read_ms=" << average_read_ms
                  << " avg_preprocess_ms=" << average_preprocess_ms
                  << " avg_detection_ms=" << average_detection_ms
                  << " detections=" << batch_people_count
                  << '\n';

        ResultHeader result_header{};
        result_header.start_frame = actual_header.start_frame;
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

}  // namespace

int run_application(int argc, char** argv, const AppConfig& config) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            run_worker(world_rank, world_size, config);
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

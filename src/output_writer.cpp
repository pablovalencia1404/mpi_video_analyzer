#include "rescue/output_writer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "rescue/tracker.hpp"

namespace rescue {

namespace {

double average_people(const std::vector<FramePacket>& packets) {
    if (packets.empty()) {
        return 0.0;
    }

    const auto total = std::accumulate(
        packets.begin(),
        packets.end(),
        0.0,
        [](double acc, const FramePacket& packet) { return acc + static_cast<double>(packet.result.people_count); });
    return total / static_cast<double>(packets.size());
}

int max_people(const std::vector<FramePacket>& packets) {
    if (packets.empty()) {
        return 0;
    }

    return std::max_element(
               packets.begin(),
               packets.end(),
               [](const FramePacket& lhs, const FramePacket& rhs) { return lhs.result.people_count < rhs.result.people_count; })
        ->result.people_count;
}

int tracking_missing_frames(const VideoMetadata& metadata) {
    if (metadata.fps <= 0.0) {
        return 32;
    }

    return std::clamp(static_cast<int>(std::lround(metadata.fps * 2.0)), 24, 72);
}

SimpleTracker build_visual_tracker(const VideoMetadata& metadata) {
    return SimpleTracker(
        0.18f,
        tracking_missing_frames(metadata),
        0.55f,
        1.35f,
        0.35f,
        0.40f);
}

std::int64_t accumulated_detections(const std::vector<FramePacket>& packets) {
    return std::accumulate(
        packets.begin(),
        packets.end(),
        std::int64_t{0},
        [](std::int64_t acc, const FramePacket& packet) {
            return acc + static_cast<std::int64_t>(packet.result.people_count);
        });
}

std::int64_t accumulated_unique_people(const std::vector<FramePacket>& ordered_packets, const VideoMetadata& metadata) {
    SimpleTracker tracker = build_visual_tracker(metadata);
    std::unordered_set<int> seen_track_ids;

    for (const auto& packet : ordered_packets) {
        std::vector<DetectionBox> detections;
        detections.reserve(static_cast<std::size_t>(packet.detection_count));
        for (int detection_index = 0; detection_index < packet.detection_count; ++detection_index) {
            detections.push_back(packet.detections[static_cast<std::size_t>(detection_index)]);
        }

        const auto tracked_boxes = tracker.update(detections);
        for (const auto& track : tracked_boxes) {
            seen_track_ids.insert(track.track_id);
        }
    }

    return static_cast<std::int64_t>(seen_track_ids.size());
}

template <typename Fn>
double average_metric(const std::vector<FramePacket>& packets, Fn metric) {
    if (packets.empty()) {
        return 0.0;
    }

    const auto total = std::accumulate(
        packets.begin(),
        packets.end(),
        0.0,
        [&](double acc, const FramePacket& packet) { return acc + metric(packet.result); });
    return total / static_cast<double>(packets.size());
}

std::size_t unique_worker_count(const std::vector<FramePacket>& packets) {
    std::set<int> worker_ranks;
    for (const auto& packet : packets) {
        worker_ranks.insert(packet.result.worker_rank);
    }
    return worker_ranks.size();
}

struct WorkerAggregate {
    int worker_rank = 0;
    int gpu_device_id = -1;
    int frame_count = 0;
    int people_total = 0;
    int first_frame = std::numeric_limits<int>::max();
    int last_frame = -1;
    double read_ms_total = 0.0;
    double preprocess_ms_total = 0.0;
    double detection_ms_total = 0.0;
    double processing_ms_total = 0.0;
};

std::map<int, WorkerAggregate> build_worker_aggregates(const std::vector<FramePacket>& packets) {
    std::map<int, WorkerAggregate> aggregates;
    for (const auto& packet : packets) {
        auto& aggregate = aggregates[packet.result.worker_rank];
        aggregate.worker_rank = packet.result.worker_rank;
        if (aggregate.gpu_device_id < 0) {
            aggregate.gpu_device_id = packet.result.gpu_device_id;
        }
        ++aggregate.frame_count;
        aggregate.people_total += packet.result.people_count;
        aggregate.first_frame = std::min(aggregate.first_frame, static_cast<int>(packet.result.frame_index));
        aggregate.last_frame = std::max(aggregate.last_frame, static_cast<int>(packet.result.frame_index));
        aggregate.read_ms_total += packet.result.read_ms;
        aggregate.preprocess_ms_total += packet.result.preprocess_ms;
        aggregate.detection_ms_total += packet.result.detection_ms;
        aggregate.processing_ms_total += packet.result.processing_ms;
    }
    return aggregates;
}

void write_worker_stats_json(
    std::ostream& json,
    const std::map<int, WorkerAggregate>& worker_aggregates,
    std::size_t total_frames) {
    json << "  \"worker_stats\": [\n";
    bool first = true;
    for (const auto& [worker_rank, aggregate] : worker_aggregates) {
        (void)worker_rank;
        if (!first) {
            json << ",\n";
        }
        first = false;

        const double frame_share =
            total_frames > 0 ? static_cast<double>(aggregate.frame_count) / static_cast<double>(total_frames) : 0.0;
        const double average_read_ms =
            aggregate.frame_count > 0 ? aggregate.read_ms_total / static_cast<double>(aggregate.frame_count) : 0.0;
        const double average_preprocess_ms =
            aggregate.frame_count > 0 ? aggregate.preprocess_ms_total / static_cast<double>(aggregate.frame_count) : 0.0;
        const double average_detection_ms =
            aggregate.frame_count > 0 ? aggregate.detection_ms_total / static_cast<double>(aggregate.frame_count) : 0.0;
        const double average_processing_ms =
            aggregate.frame_count > 0 ? aggregate.processing_ms_total / static_cast<double>(aggregate.frame_count) : 0.0;

        json << "    {\n";
        json << "      \"worker_rank\": " << aggregate.worker_rank << ",\n";
        json << "      \"gpu_device_id\": " << aggregate.gpu_device_id << ",\n";
        json << "      \"frames\": " << aggregate.frame_count << ",\n";
        json << "      \"frame_share\": " << frame_share << ",\n";
        json << "      \"people_total\": " << aggregate.people_total << ",\n";
        json << "      \"first_frame\": " << aggregate.first_frame << ",\n";
        json << "      \"last_frame\": " << aggregate.last_frame << ",\n";
        json << "      \"average_read_ms\": " << average_read_ms << ",\n";
        json << "      \"average_preprocess_ms\": " << average_preprocess_ms << ",\n";
        json << "      \"average_detection_ms\": " << average_detection_ms << ",\n";
        json << "      \"average_processing_ms\": " << average_processing_ms << "\n";
        json << "    }";
    }
    json << "\n  ],\n";
}

void write_load_balance_json(
    std::ostream& json,
    const std::map<int, WorkerAggregate>& worker_aggregates,
    int requested_workers,
    std::size_t total_frames) {
    int min_frames = 0;
    int max_frames = 0;
    double average_active_frames = 0.0;

    if (!worker_aggregates.empty()) {
        min_frames = std::numeric_limits<int>::max();
        for (const auto& [worker_rank, aggregate] : worker_aggregates) {
            (void)worker_rank;
            min_frames = std::min(min_frames, aggregate.frame_count);
            max_frames = std::max(max_frames, aggregate.frame_count);
            average_active_frames += static_cast<double>(aggregate.frame_count);
        }
        average_active_frames /= static_cast<double>(worker_aggregates.size());
    }

    const double average_requested_frames =
        requested_workers > 0 ? static_cast<double>(total_frames) / static_cast<double>(requested_workers) : 0.0;
    const double max_to_mean_ratio =
        average_active_frames > 0.0 ? static_cast<double>(max_frames) / average_active_frames : 0.0;

    json << "  \"load_balance\": {\n";
    json << "    \"active_workers\": " << worker_aggregates.size() << ",\n";
    json << "    \"average_frames_per_active_worker\": " << average_active_frames << ",\n";
    json << "    \"average_frames_per_requested_worker\": " << average_requested_frames << ",\n";
    json << "    \"min_frames_per_worker\": " << min_frames << ",\n";
    json << "    \"max_frames_per_worker\": " << max_frames << ",\n";
    json << "    \"max_to_mean_ratio\": " << max_to_mean_ratio << "\n";
    json << "  },\n";
}

std::string write_annotated_video(
    const std::string& output_dir,
    const std::string& input_path,
    const VideoMetadata& metadata,
    const cv::Size& processing_size,
    const std::map<int, FramePacket>& packets_by_frame) {
    const double fps = metadata.fps > 0.0 ? metadata.fps : 30.0;
    const cv::Size frame_size = processing_size;

    const std::vector<std::pair<std::string, std::string>> writer_options = {
        {"mp4v", ".mp4"},
        {"MJPG", ".avi"},
    };

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        throw std::runtime_error("Unable to open input video for annotation.");
    }

    for (const auto& [codec_name, extension] : writer_options) {
        const auto output_path = std::filesystem::path(output_dir) / ("annotated_people" + extension);
        const int fourcc = cv::VideoWriter::fourcc(codec_name[0], codec_name[1], codec_name[2], codec_name[3]);
        cv::VideoWriter writer;

        if (!writer.open(output_path.string(), fourcc, fps, frame_size, true)) {
            continue;
        }

        SimpleTracker tracker = build_visual_tracker(metadata);
        std::unordered_set<int> seen_track_ids;
        capture.set(cv::CAP_PROP_POS_FRAMES, 0);

        cv::Mat frame;
        int frame_index = 0;
        std::int64_t cumulative_unique_people = 0;
        while (capture.read(frame)) {
            cv::Mat annotated_frame;
            if (frame.size() != frame_size) {
                cv::resize(frame, annotated_frame, frame_size, 0.0, 0.0, cv::INTER_LINEAR);
            } else {
                annotated_frame = frame;
            }

            int current_people = 0;
            const auto packet_it = packets_by_frame.find(frame_index);
            std::vector<DetectionBox> detections;
            if (packet_it != packets_by_frame.end()) {
                detections.reserve(static_cast<std::size_t>(packet_it->second.detection_count));
                for (int detection_index = 0; detection_index < packet_it->second.detection_count; ++detection_index) {
                    detections.push_back(packet_it->second.detections[static_cast<std::size_t>(detection_index)]);
                }
            }

            const auto tracked_boxes = tracker.update(detections);
            current_people = static_cast<int>(tracked_boxes.size());

            for (const auto& track : tracked_boxes) {
                if (seen_track_ids.insert(track.track_id).second) {
                    ++cumulative_unique_people;
                }

                    const cv::Rect2f clipped = track.box & cv::Rect2f(
                        0.0f,
                        0.0f,
                        static_cast<float>(annotated_frame.cols),
                        static_cast<float>(annotated_frame.rows));
                    if (clipped.width <= 0.0f || clipped.height <= 0.0f) {
                        continue;
                    }

                    const cv::Point top_left(
                        static_cast<int>(std::round(clipped.x)),
                        static_cast<int>(std::round(clipped.y)));
                    const cv::Point bottom_right(
                        static_cast<int>(std::round(clipped.x + clipped.width)),
                        static_cast<int>(std::round(clipped.y + clipped.height)));

                    cv::rectangle(annotated_frame, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
            }

            const std::string overlay_text =
                "Personas actuales: " + std::to_string(current_people) +
                " | Acumulado real: " + std::to_string(cumulative_unique_people);
            int baseline = 0;
            const double font_scale = 0.8;
            const int thickness = 2;
            const cv::Size text_size =
                cv::getTextSize(overlay_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
            const cv::Point text_origin(16, 24 + text_size.height);
            const cv::Point box_top_left(8, 8);
            const cv::Point box_bottom_right(
                std::min(annotated_frame.cols - 8, box_top_left.x + text_size.width + 16),
                std::min(annotated_frame.rows - 8, box_top_left.y + text_size.height + baseline + 20));
            cv::rectangle(annotated_frame, box_top_left, box_bottom_right, cv::Scalar(20, 20, 20), cv::FILLED);
            cv::putText(
                annotated_frame,
                overlay_text,
                text_origin,
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                cv::Scalar(0, 255, 255),
                thickness,
                cv::LINE_AA);

            writer.write(annotated_frame);
            ++frame_index;
        }

        if (writer.isOpened()) {
            return output_path.string();
        }
    }

    throw std::runtime_error("Unable to create annotated video output.");
}

}  // namespace

void write_outputs(
    const std::string& output_dir,
    const std::string& input_path,
    const VideoMetadata& metadata,
    const std::vector<FramePacket>& ordered_packets,
    const OutputMetrics& metrics) {
    const auto output_start = std::chrono::steady_clock::now();
    std::filesystem::create_directories(output_dir);

    std::map<int, FramePacket> packets_by_frame;
    for (const auto& packet : ordered_packets) {
        packets_by_frame.emplace(packet.result.frame_index, packet);
    }
    const auto worker_aggregates = build_worker_aggregates(ordered_packets);
    const std::int64_t unique_people_total = accumulated_unique_people(ordered_packets, metadata);
    const std::int64_t detection_accumulated = accumulated_detections(ordered_packets);

    const auto csv_path = std::filesystem::path(output_dir) / "frame_results.csv";
    const auto json_path = std::filesystem::path(output_dir) / "summary.json";
    std::string annotated_video_path;
    if (metrics.annotated_video_enabled) {
        annotated_video_path = write_annotated_video(
            output_dir,
            input_path,
            metadata,
            cv::Size(metrics.processing_width, metrics.processing_height),
            packets_by_frame);
    }

    std::ofstream csv(csv_path);
    if (!csv) {
        throw std::runtime_error("Unable to create CSV output.");
    }

    csv << "frame_index,worker_rank,gpu_device_id,timestamp_ms,people_count,mean_intensity,edge_density,read_ms,preprocess_ms,detection_ms,processing_ms\n";
    csv << std::fixed << std::setprecision(3);
    for (const auto& packet : ordered_packets) {
        const auto& frame = packet.result;
        csv << frame.frame_index << ','
            << frame.worker_rank << ','
            << frame.gpu_device_id << ','
            << frame.timestamp_ms << ','
            << frame.people_count << ','
            << frame.mean_intensity << ','
            << frame.edge_density << ','
            << frame.read_ms << ','
            << frame.preprocess_ms << ','
            << frame.detection_ms << ','
            << frame.processing_ms << '\n';
    }

    std::ofstream json(json_path);
    if (!json) {
        throw std::runtime_error("Unable to create JSON summary.");
    }

    const auto output_measured_end = std::chrono::steady_clock::now();
    const double output_write_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(output_measured_end - output_start).count() / 1000.0;
    const double total_wall_ms = metrics.distributed_processing_ms + output_write_ms;

    json << std::fixed << std::setprecision(3);
    json << "{\n";
    json << "  \"input_video\": \"" << input_path << "\",\n";
    json << "  \"width\": " << metadata.width << ",\n";
    json << "  \"height\": " << metadata.height << ",\n";
    json << "  \"processing_width\": " << metrics.processing_width << ",\n";
    json << "  \"processing_height\": " << metrics.processing_height << ",\n";
    json << "  \"fps\": " << metadata.fps << ",\n";
    json << "  \"frame_count\": " << metadata.frame_count << ",\n";
    json << "  \"duration_seconds\": " << metadata.duration_seconds << ",\n";
    if (annotated_video_path.empty()) {
        json << "  \"annotated_video\": null,\n";
    } else {
        json << "  \"annotated_video\": \"" << annotated_video_path << "\",\n";
    }
    json << "  \"annotated_video_enabled\": " << (metrics.annotated_video_enabled ? "true" : "false") << ",\n";
    json << "  \"scheduler\": \"" << metrics.scheduler_name << "\",\n";
    json << "  \"mpi_world_size\": " << metrics.mpi_world_size << ",\n";
    json << "  \"requested_workers\": " << metrics.requested_workers << ",\n";
    json << "  \"requested_cuda_workers\": " << metrics.requested_workers << ",\n";
    json << "  \"cuda_worker_count\": " << unique_worker_count(ordered_packets) << ",\n";
    json << "  \"tracking_mode\": \"visualization_only\",\n";
    json << "  \"average_people_per_frame\": " << average_people(ordered_packets) << ",\n";
    json << "  \"max_people_in_frame\": " << max_people(ordered_packets) << ",\n";
    json << "  \"accumulated_people\": " << unique_people_total << ",\n";
    json << "  \"accumulated_detections\": " << detection_accumulated << ",\n";
    json << "  \"average_read_ms\": " << average_metric(ordered_packets, [](const FrameResult& result) { return result.read_ms; }) << ",\n";
    json << "  \"average_preprocess_ms\": " << average_metric(ordered_packets, [](const FrameResult& result) { return result.preprocess_ms; }) << ",\n";
    json << "  \"average_detection_ms\": " << average_metric(ordered_packets, [](const FrameResult& result) { return result.detection_ms; }) << ",\n";
    json << "  \"average_processing_ms\": " << average_metric(ordered_packets, [](const FrameResult& result) { return result.processing_ms; }) << ",\n";
    write_load_balance_json(json, worker_aggregates, metrics.requested_workers, ordered_packets.size());
    write_worker_stats_json(json, worker_aggregates, ordered_packets.size());
    json << "  \"distributed_processing_ms\": " << metrics.distributed_processing_ms << ",\n";
    json << "  \"output_write_ms\": " << output_write_ms << ",\n";
    json << "  \"total_wall_ms\": " << total_wall_ms << "\n";
    json << "}\n";
}

}  // namespace rescue

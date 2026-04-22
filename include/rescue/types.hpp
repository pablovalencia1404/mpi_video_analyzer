#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rescue {

enum class SchedulerMode {
    Dynamic,
    StaticContiguous,
    StaticRoundRobin,
};

inline const char* scheduler_mode_name(SchedulerMode mode) {
    switch (mode) {
        case SchedulerMode::Dynamic:
            return "dynamic";
        case SchedulerMode::StaticContiguous:
            return "static-contiguous";
        case SchedulerMode::StaticRoundRobin:
            return "static-round-robin";
    }
    return "dynamic";
}

struct AppConfig {
    std::string input_path;
    std::string output_dir = "output";
    int batch_size = 16;
    int resize_width = 640;
    int edge_threshold = 80;
    std::string model_path = "models/yolo11s.onnx";
    double score_threshold = 0.2;
    double nms_threshold = 0.45;
    int top_k = 5000;
    SchedulerMode scheduler_mode = SchedulerMode::Dynamic;
    bool generate_annotated_video = true;
};

struct VideoMetadata {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    int frame_count = 0;
    double duration_seconds = 0.0;
};

struct BatchHeader {
    std::int32_t start_frame = 0;
    std::int32_t frame_count = 0;
};

struct FrameResult {
    std::int32_t frame_index = 0;
    std::int32_t worker_rank = 0;
    double timestamp_ms = 0.0;
    std::int32_t people_count = 0;
    double mean_intensity = 0.0;
    double edge_density = 0.0;
    double read_ms = 0.0;
    double preprocess_ms = 0.0;
    double detection_ms = 0.0;
    double processing_ms = 0.0;
};

struct DetectionBox {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float confidence = 0.0f;
    std::int32_t class_id = 0;
};

inline constexpr std::size_t kMaxDetectionsPerFrame = 128;

struct FramePacket {
    FrameResult result;
    std::int32_t detection_count = 0;
    std::array<DetectionBox, kMaxDetectionsPerFrame> detections{};
};

struct BatchResult {
    std::vector<FramePacket> frames;
};

}  // namespace rescue

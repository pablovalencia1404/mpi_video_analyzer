#pragma once

#include <string>
#include <vector>

#include "rescue/types.hpp"

namespace rescue {

struct OutputMetrics {
    int mpi_world_size = 0;
    int requested_cuda_workers = 0;
    double distributed_processing_ms = 0.0;
    double output_write_ms = 0.0;
    double total_wall_ms = 0.0;
    std::string scheduler_name = "dynamic";
    bool annotated_video_enabled = true;
};

void write_outputs(
    const std::string& output_dir,
    const std::string& input_path,
    const VideoMetadata& metadata,
    const std::vector<FramePacket>& ordered_packets,
    const OutputMetrics& metrics);

}  // namespace rescue

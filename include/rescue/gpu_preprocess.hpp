#pragma once

#include <cstddef>

#include <opencv2/core.hpp>

namespace rescue {

struct GpuFrameArtifacts {
    GpuFrameArtifacts() = default;
    ~GpuFrameArtifacts();
    GpuFrameArtifacts(GpuFrameArtifacts&& other) noexcept;
    GpuFrameArtifacts& operator=(GpuFrameArtifacts&& other) noexcept;
    GpuFrameArtifacts(const GpuFrameArtifacts&) = delete;
    GpuFrameArtifacts& operator=(const GpuFrameArtifacts&) = delete;

    float* device_input_tensor = nullptr;
    std::size_t input_tensor_bytes = 0;
    cv::Size original_frame_size{};
    cv::Size model_input_size{};
    double resize_ratio = 1.0;
    int pad_x = 0;
    int pad_y = 0;
    double mean_intensity = 0.0;
    double edge_density = 0.0;
};

GpuFrameArtifacts preprocess_frame_cuda(
    const cv::Mat& bgr_frame,
    const cv::Size& model_input_size,
    int edge_threshold);

}  // namespace rescue

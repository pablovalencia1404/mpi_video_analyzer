#pragma once

#include <cstddef>

#include <opencv2/core.hpp>

namespace rescue {

struct OpenCLFrameArtifacts {
    cv::UMat input_tensor;
    cv::Size original_frame_size{};
    cv::Size model_input_size{};
    double resize_ratio = 1.0;
    int pad_x = 0;
    int pad_y = 0;
    double mean_intensity = 0.0;
    double edge_density = 0.0;
};

OpenCLFrameArtifacts preprocess_frame_opencl(
    const cv::Mat& bgr_frame,
    const cv::Size& model_input_size,
    int edge_threshold);

}  // namespace rescue

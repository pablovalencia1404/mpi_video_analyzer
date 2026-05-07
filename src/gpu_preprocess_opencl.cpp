#include "rescue/gpu_preprocess_opencl.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace rescue {

OpenCLFrameArtifacts preprocess_frame_opencl(
    const cv::Mat& bgr_frame,
    const cv::Size& model_input_size,
    int edge_threshold) {
    if (bgr_frame.empty()) {
        throw std::runtime_error("OpenCL preprocessing received an empty frame.");
    }
    if (bgr_frame.type() != CV_8UC3) {
        throw std::runtime_error("OpenCL preprocessing expects CV_8UC3 input.");
    }
    if (model_input_size.width <= 0 || model_input_size.height <= 0) {
        throw std::runtime_error("OpenCL preprocessing received an invalid model input size.");
    }

    cv::UMat u_frame;
    bgr_frame.copyTo(u_frame);

    const double resize_ratio = std::min(
        static_cast<double>(model_input_size.width) / static_cast<double>(u_frame.cols),
        static_cast<double>(model_input_size.height) / static_cast<double>(u_frame.rows));

    const int resized_width = std::max(1, static_cast<int>(std::lround(u_frame.cols * resize_ratio)));
    const int resized_height = std::max(1, static_cast<int>(std::lround(u_frame.rows * resize_ratio)));
    const int pad_x = (model_input_size.width - resized_width) / 2;
    const int pad_y = (model_input_size.height - resized_height) / 2;

    cv::UMat resized;
    cv::resize(u_frame, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    cv::UMat padded;
    cv::copyMakeBorder(resized, padded, pad_y, model_input_size.height - resized_height - pad_y,
                       pad_x, model_input_size.width - resized_width - pad_x,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::UMat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0/255.0, cv::Size(), cv::Scalar(0,0,0), true, false);

    cv::UMat gray;
    cv::cvtColor(padded, gray, cv::COLOR_BGR2GRAY);

    cv::Rect content_roi(pad_x, pad_y, resized_width, resized_height);
    cv::UMat gray_content = gray(content_roi);

    double mean_intensity = 0.0;
    double edge_density = 0.0;
    const double content_pixels = static_cast<double>(resized_width) * static_cast<double>(resized_height);

    if (content_pixels > 0.0) {
        cv::Scalar mean_val = cv::mean(gray_content);
        mean_intensity = mean_val[0];

        cv::UMat sobel_x, sobel_y;
        cv::Sobel(gray_content, sobel_x, CV_16S, 1, 0, 3);
        cv::Sobel(gray_content, sobel_y, CV_16S, 0, 1, 3);

        cv::UMat abs_x, abs_y;
        cv::convertScaleAbs(sobel_x, abs_x);
        cv::convertScaleAbs(sobel_y, abs_y);

        cv::UMat sobel_mag;
        cv::addWeighted(abs_x, 1.0, abs_y, 1.0, 0, sobel_mag);

        cv::UMat edges;
        cv::compare(sobel_mag, edge_threshold, edges, cv::CMP_GT);
        int edge_count = cv::countNonZero(edges);
        
        edge_density = static_cast<double>(edge_count) / content_pixels;
    }

    OpenCLFrameArtifacts artifacts;
    artifacts.input_tensor = blob;
    artifacts.original_frame_size = bgr_frame.size();
    artifacts.model_input_size = model_input_size;
    artifacts.resize_ratio = resize_ratio;
    artifacts.pad_x = pad_x;
    artifacts.pad_y = pad_y;
    artifacts.mean_intensity = mean_intensity;
    artifacts.edge_density = edge_density;

    return artifacts;
}

}  // namespace rescue

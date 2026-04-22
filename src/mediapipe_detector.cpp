#include "rescue/detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace rescue {

namespace {

void set_cuda_backend(cv::dnn::Net& net) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

bool line_has_yes_flag(const std::string& build_info, const std::string& key) {
    const std::size_t key_pos = build_info.find(key);
    if (key_pos == std::string::npos) {
        return false;
    }

    const std::size_t line_end = build_info.find('\n', key_pos);
    const std::string line = build_info.substr(
        key_pos,
        line_end == std::string::npos ? std::string::npos : (line_end - key_pos));
    return line.find("YES") != std::string::npos;
}

bool opencv_has_cuda_dnn_support() {
    const std::string build_info = cv::getBuildInformation();
    return line_has_yes_flag(build_info, "NVIDIA CUDA:") && line_has_yes_flag(build_info, "cuDNN:");
}

cv::Mat to_bgr(const cv::Mat& frame) {
    if (frame.channels() == 3) {
        return frame;
    }

    cv::Mat converted;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, converted, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, converted, cv::COLOR_BGRA2BGR);
    } else {
        throw std::runtime_error("Unsupported frame format for person detection.");
    }

    return converted;
}

std::pair<cv::Mat, cv::Size> letterbox(const cv::Mat& image, const cv::Size& target_size) {
    cv::Mat padded(target_size, image.type(), cv::Scalar::all(0));

    const double ratio = std::min(
        static_cast<double>(target_size.width) / static_cast<double>(image.cols),
        static_cast<double>(target_size.height) / static_cast<double>(image.rows));

    cv::Size resized_size(
        std::max(1, static_cast<int>(image.cols * ratio)),
        std::max(1, static_cast<int>(image.rows * ratio)));

    cv::Mat resized;
    cv::resize(image, resized, resized_size, 0.0, 0.0, cv::INTER_LINEAR);

    const int pad_w = target_size.width - resized_size.width;
    const int pad_h = target_size.height - resized_size.height;
    const int left = pad_w / 2;
    const int top = pad_h / 2;

    resized.copyTo(padded(cv::Rect(left, top, resized_size.width, resized_size.height)));

    return {padded, cv::Size(static_cast<int>(left / ratio), static_cast<int>(top / ratio))};
}

std::vector<cv::Point2f> build_mediapipe_anchors() {
    std::vector<cv::Point2f> anchors;
    anchors.reserve(2254);

    const auto append_grid = [&](int grid_size, int copies) {
        const float step = 1.0f / static_cast<float>(grid_size);
        for (int y = 0; y < grid_size; ++y) {
            const float anchor_y = (static_cast<float>(y) + 0.5f) * step;
            for (int x = 0; x < grid_size; ++x) {
                const float anchor_x = (static_cast<float>(x) + 0.5f) * step;
                for (int copy = 0; copy < copies; ++copy) {
                    anchors.emplace_back(anchor_x, anchor_y);
                }
            }
        }
    };

    append_grid(28, 2);
    append_grid(14, 2);
    append_grid(7, 6);

    if (anchors.size() != 2254) {
        throw std::runtime_error("Unexpected MediaPipe anchor count generated for person detection.");
    }

    return anchors;
}

}  // namespace

PeopleDetector::PeopleDetector(const AppConfig& config)
    : model_path_(config.model_path),
      score_threshold_(static_cast<float>(config.score_threshold)),
      nms_threshold_(static_cast<float>(config.nms_threshold)),
      top_k_(config.top_k),
      anchors_(build_mediapipe_anchors()) {
    if (!std::filesystem::exists(model_path_)) {
        throw std::runtime_error(
            "MediaPipe model not found: " + model_path_ +
            ". Download person_detection_mediapipe_2023mar.onnx and pass --model <path>.");
    }

    if (!opencv_has_cuda_dnn_support()) {
        throw std::runtime_error(
            "OpenCV DNN CUDA/cuDNN support is required. Rebuild/install OpenCV with DNN CUDA enabled.");
    }

    model_ = cv::dnn::readNetFromONNX(model_path_);
    set_cuda_backend(model_);
}

std::vector<cv::Rect> PeopleDetector::detect(const cv::Mat& frame) {
    return detect_mediapipe(frame);
}

std::vector<cv::Rect> PeopleDetector::detect_mediapipe(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::runtime_error("Empty frame passed to person detector.");
    }

    const cv::Mat bgr_frame = to_bgr(frame);
    const auto [padded_frame, pad_bias] = letterbox(bgr_frame, input_size_);

    cv::Mat blob;
    cv::dnn::blobFromImage(
        padded_frame,
        blob,
        1.0 / 127.5,
        input_size_,
        cv::Scalar::all(127.5),
        true,
        false,
        CV_32F);

    model_.setInput(blob);

    std::vector<cv::Mat> outputs;
    try {
        model_.forward(outputs, model_.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& ex) {
        throw std::runtime_error(
            std::string("MediaPipe CUDA execution failed: ") + ex.what() +
            ". Ensure OpenCV DNN CUDA is available and libcuda is resolved correctly (in WSL, export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}).");
    }

    if (outputs.size() != 2) {
        throw std::runtime_error("Unexpected number of MediaPipe outputs. Expected 2 tensors.");
    }

    const cv::Mat* box_output = &outputs[0];
    const cv::Mat* score_output = &outputs[1];
    if (outputs[0].total() < outputs[1].total()) {
        box_output = &outputs[1];
        score_output = &outputs[0];
    }

    const std::size_t score_total = score_output->total();
    const std::size_t box_total = box_output->total();
    if (box_total != score_total * 12) {
        throw std::runtime_error(
            "MediaPipe output shape mismatch: expected box tensor to contain 12 values per candidate.");
    }

    const int candidate_count = static_cast<int>(score_total);
    if (candidate_count != static_cast<int>(anchors_.size())) {
        throw std::runtime_error(
            "MediaPipe anchor count mismatch: expected " + std::to_string(anchors_.size()) +
            ", got " + std::to_string(candidate_count) + ".");
    }

    const cv::Mat box_deltas = box_output->reshape(1, candidate_count);
    const cv::Mat score_logits = score_output->reshape(1, candidate_count);

    std::vector<cv::Rect2d> candidate_boxes;
    std::vector<float> candidate_scores;
    candidate_boxes.reserve(candidate_count);
    candidate_scores.reserve(candidate_count);

    const double scale = static_cast<double>(std::max(frame.cols, frame.rows));
    const float* scores_data = score_logits.ptr<float>();

    for (int idx = 0; idx < candidate_count; ++idx) {
        const float* row = box_deltas.ptr<float>(idx);

        float score = scores_data[idx];
        score = std::clamp(score, -100.0f, 100.0f);
        score = 1.0f / (1.0f + std::exp(-score));
        if (score < score_threshold_) {
            continue;
        }

        const cv::Point2f& anchor = anchors_[idx];
        const double center_x = (static_cast<double>(row[0]) / input_size_.width + anchor.x) * scale - pad_bias.width;
        const double center_y = (static_cast<double>(row[1]) / input_size_.height + anchor.y) * scale - pad_bias.height;
        const double half_width = (static_cast<double>(row[2]) / input_size_.width) * scale * 0.5;
        const double half_height = (static_cast<double>(row[3]) / input_size_.height) * scale * 0.5;

        const double x1 = center_x - half_width;
        const double y1 = center_y - half_height;
        const double x2 = center_x + half_width;
        const double y2 = center_y + half_height;

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        candidate_boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        candidate_scores.push_back(score);
    }

    if (candidate_boxes.empty()) {
        return {};
    }

    std::vector<int> keep_indices;
    cv::dnn::NMSBoxes(
        candidate_boxes,
        candidate_scores,
        score_threshold_,
        nms_threshold_,
        keep_indices,
        1.0f,
        top_k_);

    std::vector<cv::Rect> detections;
    detections.reserve(keep_indices.size());

    for (const int idx : keep_indices) {
        const cv::Rect2d& box = candidate_boxes[idx];
        const int left = std::max(0, static_cast<int>(std::round(box.x)));
        const int top = std::max(0, static_cast<int>(std::round(box.y)));
        const int right = std::min(frame.cols, static_cast<int>(std::round(box.x + box.width)));
        const int bottom = std::min(frame.rows, static_cast<int>(std::round(box.y + box.height)));

        if (right <= left || bottom <= top) {
            continue;
        }

        detections.emplace_back(left, top, right - left, bottom - top);
    }

    return detections;
}

}  // namespace rescue
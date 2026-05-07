#include "rescue/detector.hpp"
#include "rescue/gpu_preprocess_opencl.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <cmath>

namespace rescue {

namespace {
struct LetterboxInfo {
    double ratio = 1.0;
    int pad_x = 0;
    int pad_y = 0;
};
constexpr std::int32_t kPersonClassId = 0;

cv::Rect2d clip_box(const cv::Rect2d& box, const cv::Size& frame_size) {
    const double left = std::max(0.0, std::min(box.x, static_cast<double>(frame_size.width)));
    const double top = std::max(0.0, std::min(box.y, static_cast<double>(frame_size.height)));
    const double right = std::max(0.0, std::min(box.x + box.width, static_cast<double>(frame_size.width)));
    const double bottom = std::max(0.0, std::min(box.y + box.height, static_cast<double>(frame_size.height)));
    return {left, top, std::max(0.0, right - left), std::max(0.0, bottom - top)};
}

cv::Mat ensure_2d_output(const cv::Mat& output) {
    if (output.dims == 2) {
        return output;
    }
    if (output.dims == 3 && output.size[0] == 1) {
        return output.reshape(1, output.size[1]);
    }
    throw std::runtime_error("Unsupported YOLO output tensor shape.");
}

std::vector<DetectionBox> decode_yolo_output(
    const cv::Mat& output,
    const LetterboxInfo& letterbox_info,
    const cv::Size& frame_size,
    float score_threshold,
    float nms_threshold,
    int top_k) {
    cv::Mat reshaped = ensure_2d_output(output);
    if (reshaped.rows < reshaped.cols) {
        reshaped = reshaped.t();
    }

    std::vector<DetectionBox> detections;
    const bool end_to_end_format = reshaped.cols == 6;
    if (end_to_end_format) {
        for (int row_index = 0; row_index < reshaped.rows; ++row_index) {
            const float* row = reshaped.ptr<float>(row_index);
            const float confidence = row[4];
            const int class_id = static_cast<int>(std::lround(row[5]));
            if (class_id != kPersonClassId || confidence < score_threshold) {
                continue;
            }

            const double raw_left = row[0];
            const double raw_top = row[1];
            const double raw_right = row[2];
            const double raw_bottom = row[3];

            const bool normalized = std::max(
                {std::abs(raw_left), std::abs(raw_top), std::abs(raw_right), std::abs(raw_bottom)}) <= 2.0;
            const double scale_x = normalized ? static_cast<double>(frame_size.width) : 1.0;
            const double scale_y = normalized ? static_cast<double>(frame_size.height) : 1.0;

            const double left = (raw_left * scale_x - letterbox_info.pad_x) / letterbox_info.ratio;
            const double top = (raw_top * scale_y - letterbox_info.pad_y) / letterbox_info.ratio;
            const double right = (raw_right * scale_x - letterbox_info.pad_x) / letterbox_info.ratio;
            const double bottom = (raw_bottom * scale_y - letterbox_info.pad_y) / letterbox_info.ratio;

            cv::Rect2d box(left, top, right - left, bottom - top);
            box = clip_box(box, frame_size);
            if (box.width <= 0.0 || box.height <= 0.0) {
                continue;
            }

            detections.push_back({
                static_cast<float>(box.x),
                static_cast<float>(box.y),
                static_cast<float>(box.width),
                static_cast<float>(box.height),
                confidence,
                class_id,
            });
        }
    } else {
        std::vector<cv::Rect2d> candidate_boxes;
        std::vector<float> candidate_scores;
        candidate_boxes.reserve(reshaped.rows);
        candidate_scores.reserve(reshaped.rows);

        for (int row_index = 0; row_index < reshaped.rows; ++row_index) {
            const float* row = reshaped.ptr<float>(row_index);
            if (reshaped.cols < 5) {
                throw std::runtime_error("YOLO output tensor does not contain enough columns.");
            }

            float best_score = 0.0f;
            int best_class_id = 0;
            for (int class_id = 0; class_id < reshaped.cols - 4; ++class_id) {
                const float score = row[4 + class_id];
                if (score > best_score) {
                    best_score = score;
                    best_class_id = class_id;
                }
            }

            if (best_class_id != kPersonClassId || best_score < score_threshold) {
                continue;
            }

            const double raw_center_x = row[0];
            const double raw_center_y = row[1];
            const double raw_width = row[2];
            const double raw_height = row[3];

            const bool normalized = std::max(
                {std::abs(raw_center_x), std::abs(raw_center_y), std::abs(raw_width), std::abs(raw_height)}) <= 2.0;
            const double scale_x = normalized ? static_cast<double>(frame_size.width) : 1.0;
            const double scale_y = normalized ? static_cast<double>(frame_size.height) : 1.0;

            const double center_x = (raw_center_x * scale_x - letterbox_info.pad_x) / letterbox_info.ratio;
            const double center_y = (raw_center_y * scale_y - letterbox_info.pad_y) / letterbox_info.ratio;
            const double box_width = (raw_width * scale_x) / letterbox_info.ratio;
            const double box_height = (raw_height * scale_y) / letterbox_info.ratio;

            const double left = center_x - box_width * 0.5;
            const double top = center_y - box_height * 0.5;
            const double right = center_x + box_width * 0.5;
            const double bottom = center_y + box_height * 0.5;

            cv::Rect2d box(left, top, right - left, bottom - top);
            box = clip_box(box, frame_size);
            if (box.width <= 0.0 || box.height <= 0.0) {
                continue;
            }

            candidate_boxes.push_back(box);
            candidate_scores.push_back(best_score);
        }

        if (!candidate_boxes.empty()) {
            std::vector<int> keep_indices;
            cv::dnn::NMSBoxes(
                candidate_boxes,
                candidate_scores,
                score_threshold,
                nms_threshold,
                keep_indices,
                1.0f,
                top_k);

            detections.reserve(keep_indices.size());
            for (const int index : keep_indices) {
                const cv::Rect2d& box = candidate_boxes[index];
                detections.push_back({
                    static_cast<float>(box.x),
                    static_cast<float>(box.y),
                    static_cast<float>(box.width),
                    static_cast<float>(box.height),
                    candidate_scores[index],
                    0,
                });
            }
        }
    }

    std::sort(
        detections.begin(),
        detections.end(),
        [](const DetectionBox& lhs, const DetectionBox& rhs) { return lhs.confidence > rhs.confidence; });

    return detections;
}

} // namespace

class OpenCLPeopleDetectorImpl : public OpenCLPeopleDetector {
public:
    explicit OpenCLPeopleDetectorImpl(const AppConfig& config);
    ~OpenCLPeopleDetectorImpl() override = default;

    std::vector<DetectionBox> detect(const OpenCLFrameArtifacts& frame_artifacts) override;
    const std::string& backend_name() const noexcept override;
    const cv::Size& input_size() const noexcept override;

private:
    cv::dnn::Net net_;
    cv::Size input_size_{};
    float score_threshold_ = 0.2f;
    float nms_threshold_ = 0.45f;
    int top_k_ = 5000;
    std::string backend_name_ = "OpenCV DNN OpenCL";
};

OpenCLPeopleDetectorImpl::OpenCLPeopleDetectorImpl(const AppConfig& config)
    : score_threshold_(static_cast<float>(config.score_threshold)),
      nms_threshold_(static_cast<float>(config.nms_threshold)),
      top_k_(config.top_k) {
    if (!std::filesystem::exists(config.model_path)) {
        throw std::runtime_error(
            "YOLO11 model not found: " + config.model_path +
            ". Place a YOLO11 ONNX model there (recommended: models/yolo11s_1088.onnx) or pass --model <path>.");
    }

    if (config.resize_width <= 0 || (config.resize_width % 32) != 0) {
        throw std::runtime_error("--resize-width must be a positive multiple of 32 for the YOLO detector path.");
    }

    input_size_ = cv::Size(config.resize_width, config.resize_width);

    net_ = cv::dnn::readNetFromONNX(config.model_path);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

    if (!cv::ocl::haveOpenCL()) {
        std::clog << "Warning: OpenCL is not available in OpenCV. Falling back to CPU execution.\n";
        backend_name_ = "OpenCV DNN CPU";
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } else {
        cv::ocl::setUseOpenCL(true);
        backend_name_ = "OpenCV OpenCL (" + cv::ocl::Device::getDefault().name() + ")";
    }
}

const std::string& OpenCLPeopleDetectorImpl::backend_name() const noexcept {
    return backend_name_;
}

const cv::Size& OpenCLPeopleDetectorImpl::input_size() const noexcept {
    return input_size_;
}

std::vector<DetectionBox> OpenCLPeopleDetectorImpl::detect(const OpenCLFrameArtifacts& frame_artifacts) {
    net_.setInput(frame_artifacts.input_tensor);
    cv::Mat output = net_.forward();

    LetterboxInfo letterbox_info{
        frame_artifacts.resize_ratio,
        frame_artifacts.pad_x,
        frame_artifacts.pad_y
    };

    return decode_yolo_output(
        output,
        letterbox_info,
        frame_artifacts.original_frame_size,
        score_threshold_,
        nms_threshold_,
        top_k_);
}

std::unique_ptr<OpenCLPeopleDetector> create_opencl_detector(const AppConfig& config) {
    return std::make_unique<OpenCLPeopleDetectorImpl>(config);
}

} // namespace rescue

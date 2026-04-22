#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "rescue/gpu_preprocess.hpp"
#include "rescue/types.hpp"

namespace rescue {

class OrtYoloSession;

class PeopleDetector {
public:
    explicit PeopleDetector(const AppConfig& config);
    ~PeopleDetector();
    std::vector<DetectionBox> detect(const GpuFrameArtifacts& frame_artifacts);
    const std::string& backend_name() const noexcept;
    const cv::Size& input_size() const noexcept;

private:
    std::unique_ptr<OrtYoloSession> ort_session_;
    cv::Size input_size_{640, 640};
    float score_threshold_ = 0.2f;
    float nms_threshold_ = 0.45f;
    int top_k_ = 5000;
    std::string backend_name_ = "ONNX Runtime CUDA";
};

}  // namespace rescue

#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "rescue/types.hpp"

namespace rescue {

class IPeopleDetector {
public:
    virtual ~IPeopleDetector() = default;
    virtual const std::string& backend_name() const noexcept = 0;
    virtual const cv::Size& input_size() const noexcept = 0;
};

struct GpuFrameArtifacts;
struct OpenCLFrameArtifacts;

class CudaPeopleDetector : public IPeopleDetector {
public:
    virtual std::vector<DetectionBox> detect(const GpuFrameArtifacts& frame_artifacts) = 0;
};

class OpenCLPeopleDetector : public IPeopleDetector {
public:
    virtual std::vector<DetectionBox> detect(const OpenCLFrameArtifacts& frame_artifacts) = 0;
};

std::unique_ptr<CudaPeopleDetector> create_cuda_detector(const AppConfig& config);
std::unique_ptr<OpenCLPeopleDetector> create_opencl_detector(const AppConfig& config);

}  // namespace rescue

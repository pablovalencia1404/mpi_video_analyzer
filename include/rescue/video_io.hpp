#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <vector>

#include "rescue/types.hpp"

namespace rescue {

class VideoReader {
public:
    explicit VideoReader(const std::string& path);

    const VideoMetadata& metadata() const noexcept { return metadata_; }
    bool read_batch(int batch_size, BatchHeader& header, std::vector<cv::Mat>& frames);
    bool read_batch_from(int start_frame, int batch_size, BatchHeader& header, std::vector<cv::Mat>& frames);

private:
    cv::VideoCapture capture_;
    VideoMetadata metadata_;
    int next_frame_index_ = 0;
};

}  // namespace rescue


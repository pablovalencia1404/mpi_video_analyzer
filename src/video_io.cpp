#include "rescue/video_io.hpp"

#include <stdexcept>

namespace rescue {

VideoReader::VideoReader(const std::string& path) : capture_(path) {
    if (!capture_.isOpened()) {
        throw std::runtime_error("Unable to open input video: " + path);
    }

    metadata_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    metadata_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    metadata_.fps = capture_.get(cv::CAP_PROP_FPS);
    metadata_.frame_count = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
    if (metadata_.fps > 0.0) {
        metadata_.duration_seconds = static_cast<double>(metadata_.frame_count) / metadata_.fps;
    }
}

bool VideoReader::read_batch(int batch_size, BatchHeader& header, std::vector<cv::Mat>& frames) {
    const int start_frame = next_frame_index_;
    const bool ok = read_batch_from(start_frame, batch_size, header, frames);
    if (ok) {
        next_frame_index_ = header.start_frame + header.frame_count;
    }
    return ok;
}

bool VideoReader::read_batch_from(int start_frame, int batch_size, BatchHeader& header, std::vector<cv::Mat>& frames) {
    frames.clear();
    header = {};

    if (start_frame < 0 || batch_size <= 0 || start_frame >= metadata_.frame_count) {
        return false;
    }

    if (!capture_.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(start_frame))) {
        throw std::runtime_error("Unable to seek video to frame " + std::to_string(start_frame));
    }

    cv::Mat frame;
    for (int i = 0; i < batch_size; ++i) {
        if (!capture_.read(frame)) {
            break;
        }
        frames.push_back(frame.clone());
    }

    if (frames.empty()) {
        return false;
    }

    header.start_frame = start_frame;
    header.frame_count = static_cast<std::int32_t>(frames.size());
    next_frame_index_ = header.start_frame + header.frame_count;
    return true;
}

}  // namespace rescue

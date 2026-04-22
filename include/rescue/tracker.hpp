#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "rescue/types.hpp"

namespace rescue {

struct TrackedBox {
    int track_id = 0;
    cv::Rect2f box;
    float confidence = 0.0f;
};

class SimpleTracker {
public:
    SimpleTracker(float iou_threshold = 0.3f, int max_missing_frames = 8, float smoothing = 0.65f);

    std::vector<TrackedBox> update(const std::vector<DetectionBox>& detections);

private:
    struct TrackState {
        int track_id = 0;
        cv::Rect2f box;
        float confidence = 0.0f;
        int missing_frames = 0;
    };

    static float intersection_over_union(const cv::Rect2f& lhs, const cv::Rect2f& rhs);

    float iou_threshold_ = 0.3f;
    int max_missing_frames_ = 8;
    float smoothing_ = 0.65f;
    int next_track_id_ = 1;
    std::vector<TrackState> tracks_;
};

}  // namespace rescue
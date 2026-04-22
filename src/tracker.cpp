#include "rescue/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rescue {

namespace {

cv::Rect2f to_rect(const DetectionBox& detection) {
    return cv::Rect2f(detection.x, detection.y, detection.width, detection.height);
}

cv::Rect2f blend_rect(const cv::Rect2f& previous, const cv::Rect2f& next, float smoothing) {
    const float inverse = 1.0f - smoothing;
    return cv::Rect2f(
        previous.x * inverse + next.x * smoothing,
        previous.y * inverse + next.y * smoothing,
        previous.width * inverse + next.width * smoothing,
        previous.height * inverse + next.height * smoothing);
}

}  // namespace

SimpleTracker::SimpleTracker(float iou_threshold, int max_missing_frames, float smoothing)
    : iou_threshold_(iou_threshold), max_missing_frames_(max_missing_frames), smoothing_(smoothing) {}

float SimpleTracker::intersection_over_union(const cv::Rect2f& lhs, const cv::Rect2f& rhs) {
    const float left = std::max(lhs.x, rhs.x);
    const float top = std::max(lhs.y, rhs.y);
    const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);

    const float inter_width = std::max(0.0f, right - left);
    const float inter_height = std::max(0.0f, bottom - top);
    const float intersection = inter_width * inter_height;
    const float union_area = lhs.area() + rhs.area() - intersection;
    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection / union_area;
}

std::vector<TrackedBox> SimpleTracker::update(const std::vector<DetectionBox>& detections) {
    struct MatchCandidate {
        std::size_t track_index = 0;
        std::size_t detection_index = 0;
        float iou = 0.0f;
    };

    std::vector<MatchCandidate> candidates;
    candidates.reserve(tracks_.size() * detections.size());

    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            const float iou = intersection_over_union(tracks_[track_index].box, to_rect(detections[detection_index]));
            if (iou >= iou_threshold_) {
                candidates.push_back({track_index, detection_index, iou});
            }
        }
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const MatchCandidate& lhs, const MatchCandidate& rhs) { return lhs.iou > rhs.iou; });

    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> detection_used(detections.size(), false);
    std::vector<TrackedBox> visible_tracks;
    visible_tracks.reserve(detections.size());

    std::vector<TrackState> next_tracks;
    next_tracks.reserve(tracks_.size() + detections.size());

    for (const auto& candidate : candidates) {
        if (track_used[candidate.track_index] || detection_used[candidate.detection_index]) {
            continue;
        }

        TrackState track = tracks_[candidate.track_index];
        const cv::Rect2f detection_box = to_rect(detections[candidate.detection_index]);
        track.box = blend_rect(track.box, detection_box, smoothing_);
        track.confidence = detections[candidate.detection_index].confidence;
        track.missing_frames = 0;

        track_used[candidate.track_index] = true;
        detection_used[candidate.detection_index] = true;
        next_tracks.push_back(track);
        visible_tracks.push_back({track.track_id, track.box, track.confidence});
    }

    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
        if (track_used[track_index]) {
            continue;
        }

        TrackState track = tracks_[track_index];
        ++track.missing_frames;
        if (track.missing_frames <= max_missing_frames_) {
            next_tracks.push_back(track);
        }
    }

    for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
        if (detection_used[detection_index]) {
            continue;
        }

        const auto& detection = detections[detection_index];
        TrackState track;
        track.track_id = next_track_id_++;
        track.box = to_rect(detection);
        track.confidence = detection.confidence;
        track.missing_frames = 0;
        next_tracks.push_back(track);
        visible_tracks.push_back({track.track_id, track.box, track.confidence});
    }

    tracks_ = std::move(next_tracks);
    return visible_tracks;
}

}  // namespace rescue
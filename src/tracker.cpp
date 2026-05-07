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

cv::Point2f rect_center(const cv::Rect2f& rect) {
    return cv::Point2f(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
}

float rect_diagonal(const cv::Rect2f& rect) {
    return std::sqrt(std::max(1.0f, rect.width * rect.width + rect.height * rect.height));
}

float normalized_center_distance(const cv::Rect2f& lhs, const cv::Rect2f& rhs) {
    const cv::Point2f lhs_center = rect_center(lhs);
    const cv::Point2f rhs_center = rect_center(rhs);
    const float dx = lhs_center.x - rhs_center.x;
    const float dy = lhs_center.y - rhs_center.y;
    const float distance = std::sqrt(dx * dx + dy * dy);
    const float normalization = std::max(1.0f, 0.5f * (rect_diagonal(lhs) + rect_diagonal(rhs)));
    return distance / normalization;
}

float area_ratio(const cv::Rect2f& lhs, const cv::Rect2f& rhs) {
    const float lhs_area = std::max(1.0f, lhs.area());
    const float rhs_area = std::max(1.0f, rhs.area());
    const float smaller = std::min(lhs_area, rhs_area);
    const float larger = std::max(lhs_area, rhs_area);
    return smaller / larger;
}

cv::Rect2f predict_rect(const cv::Rect2f& rect, const cv::Point2f& velocity) {
    return cv::Rect2f(rect.x + velocity.x, rect.y + velocity.y, rect.width, rect.height);
}

cv::Point2f blend_velocity(const cv::Point2f& previous, const cv::Point2f& observed, float smoothing) {
    const float inverse = 1.0f - smoothing;
    return cv::Point2f(
        previous.x * inverse + observed.x * smoothing,
        previous.y * inverse + observed.y * smoothing);
}

}  // namespace

SimpleTracker::SimpleTracker(
    float iou_threshold,
    int max_missing_frames,
    float smoothing,
    float max_center_distance,
    float min_area_ratio,
    float velocity_smoothing)
    : iou_threshold_(iou_threshold),
      max_missing_frames_(max_missing_frames),
      smoothing_(smoothing),
      max_center_distance_(max_center_distance),
      min_area_ratio_(min_area_ratio),
      velocity_smoothing_(velocity_smoothing) {}

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
        float score = 0.0f;
    };

    std::vector<MatchCandidate> candidates;
    candidates.reserve(tracks_.size() * detections.size());

    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
        const cv::Rect2f predicted_track_box = predict_rect(tracks_[track_index].box, tracks_[track_index].velocity);
        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            const cv::Rect2f detection_box = to_rect(detections[detection_index]);
            const float iou = intersection_over_union(predicted_track_box, detection_box);
            const float center_distance = normalized_center_distance(predicted_track_box, detection_box);
            const float box_area_ratio = area_ratio(predicted_track_box, detection_box);
            const float allowed_center_distance =
                max_center_distance_ * (1.0f + 0.15f * static_cast<float>(tracks_[track_index].missing_frames));

            if (iou < iou_threshold_ && !(center_distance <= allowed_center_distance && box_area_ratio >= min_area_ratio_)) {
                continue;
            }

            const float score =
                (3.0f * iou) +
                (1.5f * box_area_ratio) -
                center_distance -
                (0.03f * static_cast<float>(tracks_[track_index].missing_frames));
            candidates.push_back({track_index, detection_index, score});
        }
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const MatchCandidate& lhs, const MatchCandidate& rhs) { return lhs.score > rhs.score; });

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
        const cv::Rect2f predicted_track_box = predict_rect(track.box, track.velocity);
        const cv::Point2f observed_velocity = rect_center(detection_box) - rect_center(track.box);
        track.velocity = blend_velocity(track.velocity, observed_velocity, velocity_smoothing_);
        track.box = blend_rect(predicted_track_box, detection_box, smoothing_);
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
        track.box = predict_rect(track.box, track.velocity);
        track.velocity *= 0.9f;
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
        track.velocity = cv::Point2f(0.0f, 0.0f);
        track.confidence = detection.confidence;
        track.missing_frames = 0;
        next_tracks.push_back(track);
        visible_tracks.push_back({track.track_id, track.box, track.confidence});
    }

    tracks_ = std::move(next_tracks);
    return visible_tracks;
}

}  // namespace rescue

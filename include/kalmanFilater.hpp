#ifndef KALMAN_FILTER
#define KALMAN_FILTER

#include <opencv2/opencv.hpp>
#include <queue>

#include "basic_types.hpp"

class ArmorTracker {
public:
    ArmorTracker(float frame_rate = 30.0f, int max_lost_frames = 10);
    void update(const cv::Point3f &measurement);
    void update_lost();
    cv::Point3f predictFuture(int n_frames) const;
    cv::Point3f getCurrentEstimate() const;
    bool isInitialized() const;
private:
    cv::KalmanFilter kf_;
    float frame_rate_;
    float dt_;
    int max_lost_frames_;
    int lost_count_;
    bool initialized_;
};

class ArmorTracker_time {
public:
    ArmorTracker_time();
    void update(double timestamp, const cv::Point3f& measurement);
    cv::Point3f predictFuture(float dt_seconds) const;
    cv::Point3f getCurrentEstimate() const;
    bool isInitialized() const;

private:
    bool initialized_ = false;
    double last_timestamp_ = 0.0;

    cv::KalmanFilter kf_;
};

#endif
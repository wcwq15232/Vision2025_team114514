#ifndef KALMAN_FILTER
#define KALMAN_FILTER

#include <opencv2/opencv.hpp>
#include <queue>

#include "basic_types.hpp"

class ArmorKalmanFilter {
public:
    explicit ArmorKalmanFilter(int max_lost_frames = 8);
    cv::Point2f correct(const Armor& armor);
    cv::Point2f predictWhenLost();
    cv::Point2f predictSteps(int n) const;
    cv::Point2f getCurrentEstimate() const;
    bool isTracking() const;
    int getLostCount() const;
private:
    cv::KalmanFilter kf_;
    cv::Point2f current_estimate_;
    int max_lost_frames_;
    int lost_count_;
    bool initialized_;
};

#endif
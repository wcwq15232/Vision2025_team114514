#ifndef KALMAN_FILTER
#define KALMAN_FILTER

#include <opencv2/opencv.hpp>
#include <queue>

#include "basic_types.hpp"

class ArmorKalmanFilter {
public:
    ArmorKalmanFilter();
    cv::Point2f correct(const Armor& armor);
    cv::Point2f getPredictedPosition() const;
private:
    cv::KalmanFilter kf;
    bool initialized;
};

#endif
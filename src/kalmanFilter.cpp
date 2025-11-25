#include <opencv2/opencv.hpp>
#include <queue>

#include "basic_types.hpp"
#include "kalmanFilater.hpp"

ArmorKalmanFilter::ArmorKalmanFilter(int max_lost_frames)
    : max_lost_frames_(max_lost_frames), lost_count_(0), initialized_(false) {
    
    kf_ = cv::KalmanFilter(4, 2, 0);

    // 状态: [x, y, vx, vy]
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1
    );

    kf_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0
    );

    // 可根据实际调整这些协方差
    kf_.measurementNoiseCov = (cv::Mat_<float>(2, 2) << 10.0f, 0, 0, 10.0f);
    kf_.processNoiseCov = (cv::Mat_<float>(4, 4) <<
        0.5f, 0, 0, 0,
        0, 0.5f, 0, 0,
        0, 0, 0.1f, 0,
        0, 0, 0, 0.1f
    );

    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1e-2));
    kf_.statePost.setTo(cv::Scalar::all(0));
}

    // 有观测时调用（正常跟踪）
cv::Point2f ArmorKalmanFilter::correct(const Armor& armor) {
    lost_count_ = 0; // 重置丢失计数

    cv::Mat measurement = (cv::Mat_<float>(2, 1) << armor.center.x, armor.center.y);

    if (!initialized_) {
        // 第一次观测：用测量值初始化位置，速度设为0
        kf_.statePost.at<float>(0) = armor.center.x;
        kf_.statePost.at<float>(1) = armor.center.y;
        kf_.statePost.at<float>(2) = 0;
        kf_.statePost.at<float>(3) = 0;
        initialized_ = true;
    }

    kf_.predict();
    cv::Mat estimate = kf_.correct(measurement);
    current_estimate_ = cv::Point2f(estimate.at<float>(0), estimate.at<float>(1));
    return current_estimate_;
}    // 预测未来 n 帧的位置（n >= 1）

    // 无观测时调用（目标丢失，但仍需预测）
cv::Point2f ArmorKalmanFilter::predictWhenLost() {
    if (!initialized_) {
        return cv::Point2f(0, 0);
    }

    lost_count_++;
    if (lost_count_ > max_lost_frames_) {
        // 彻底丢失，重置
        initialized_ = false;
        lost_count_ = 0;
        return cv::Point2f(0, 0);
    }

    // 纯预测（无校正）
    cv::Mat prediction = kf_.predict();
    current_estimate_ = cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
    return current_estimate_;
}

    // 预测未来 n 帧的位置（n >= 1）
cv::Point2f ArmorKalmanFilter::predictSteps(int n) const {
    if (!initialized_) return cv::Point2f(0, 0);

    // 注意：这里我们不修改原 kf_ 状态，而是拷贝一份进行前向模拟
    cv::KalmanFilter temp_kf = kf_;
    cv::Point2f pred;

    for (int i = 0; i < n; ++i) {
        cv::Mat p = temp_kf.predict();
        pred = cv::Point2f(p.at<float>(0), p.at<float>(1));
    }
    return pred;
}

    // 获取当前（最新校正或预测的）位置
cv::Point2f ArmorKalmanFilter::getCurrentEstimate() const {
    return current_estimate_;
}

bool ArmorKalmanFilter::isTracking() const {
    return initialized_ && lost_count_ <= max_lost_frames_;
}

int ArmorKalmanFilter::getLostCount() const {
    return lost_count_;
}
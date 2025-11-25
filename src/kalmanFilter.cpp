#include <opencv2/opencv.hpp>
#include <queue>

#include "basic_types.hpp"
#include "kalmanFilater.hpp"

ArmorTracker::ArmorTracker(float frame_rate, int max_lost_frames)
    : frame_rate_(frame_rate), max_lost_frames_(max_lost_frames),
        lost_count_(0), initialized_(false) {
    dt_ = 1.0f / frame_rate_;

    kf_ = cv::KalmanFilter(6, 3, 0, CV_32F);

    // Initialize matrices
    cv::setIdentity(kf_.transitionMatrix);
    kf_.transitionMatrix.at<float>(0, 3) = dt_;
    kf_.transitionMatrix.at<float>(1, 4) = dt_;
    kf_.transitionMatrix.at<float>(2, 5) = dt_;

    kf_.measurementMatrix = cv::Mat::zeros(3, 6, CV_32F);
    kf_.measurementMatrix.at<float>(0, 0) = 1.0f;
    kf_.measurementMatrix.at<float>(1, 1) = 1.0f;
    kf_.measurementMatrix.at<float>(2, 2) = 1.0f;

    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(0.0025f)); // 5cm^2
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0f));
}

void ArmorTracker::update(const cv::Point3f &measurement) {
    if (!initialized_) {
        // 重新初始化：目标重新出现
        cv::Mat state = cv::Mat::zeros(6, 1, CV_32F);
        state.at<float>(0) = measurement.x;
        state.at<float>(1) = measurement.y;
        state.at<float>(2) = measurement.z;
        // 速度初始化为 0
        state.at<float>(3) = 0.0f;
        state.at<float>(4) = 0.0f;
        state.at<float>(5) = 0.0f;

        kf_.statePost = state.clone();
        // 可选：重置误差协方差，加快收敛
        cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0f));

        initialized_ = true;
        lost_count_ = 0;

        return;
    }

    // 正常更新流程：已有跟踪
    cv::Mat prediction = kf_.predict();
    cv::Mat z = (cv::Mat_<float>(3, 1) << 
        measurement.x, measurement.y, measurement.z);
    kf_.correct(z);
    lost_count_ = 0;
}

void ArmorTracker::update_lost(){
    // 无观测：仅预测
    lost_count_++;
    if (lost_count_ > max_lost_frames_) {
        initialized_ = false; // 重置状态
        // 可选：重置 kf_.statePost 或重新初始化
    }
}

// 预测 n 帧后的位置（外推）
cv::Point3f ArmorTracker::predictFuture(int n_frames) const {
    if (!initialized_) {
        // 可返回 (0,0,0) 或抛异常，或返回最后一次预测
        return cv::Point3f(0, 0, 0);
    }

    cv::Mat state = kf_.statePost.clone(); // [x, y, z, vx, vy, vz]^T
    float dt_total = n_frames * dt_;

    // 外推：x' = x + vx * dt_total，同理 y, z
    float x = state.at<float>(0) + state.at<float>(3) * dt_total;
    float y = state.at<float>(1) + state.at<float>(4) * dt_total;
    float z = state.at<float>(2) + state.at<float>(5) * dt_total;

    return cv::Point3f(x, y, z);
}

// 获取当前估计位置（用于调试或替代 predictFuture(0)）
cv::Point3f ArmorTracker::getCurrentEstimate() const {
    return cv::Point3f(
        kf_.statePost.at<float>(0),
        kf_.statePost.at<float>(1),
        kf_.statePost.at<float>(2)
    );
}

bool ArmorTracker::isInitialized() const { return initialized_; }

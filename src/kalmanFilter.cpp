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

 



ArmorTracker_time::ArmorTracker_time() {
    // 状态：[x, y, z, vx, vy, vz]^T
    // 观测：[x, y, z]
    kf_ = cv::KalmanFilter(6, 3, 0, CV_32F);

    // Measurement matrix: 只观测位置
    kf_.measurementMatrix = cv::Mat::zeros(3, 6, CV_32F);
    kf_.measurementMatrix.at<float>(0, 0) = 1.0f;
    kf_.measurementMatrix.at<float>(1, 1) = 1.0f;
    kf_.measurementMatrix.at<float>(2, 2) = 1.0f;

    // Process noise: 可根据加速度不确定性调整
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2f));

    // Measurement noise: 假设 5cm 标准差 => 方差 0.0025
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(0.0025f));

    // 初始误差协方差
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0f));
}

void ArmorTracker_time::update(double timestamp, const cv::Point3f& measurement) {
    if (!initialized_) {
        // 首次初始化
        cv::Mat state = cv::Mat::zeros(6, 1, CV_32F);
        state.at<float>(0) = measurement.x;
        state.at<float>(1) = measurement.y;
        state.at<float>(2) = measurement.z;
        // 速度初始化为 0
        state.at<float>(3) = 0.0f;
        state.at<float>(4) = 0.0f;
        state.at<float>(5) = 0.0f;

        kf_.statePost = state.clone();
        cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0f));

        initialized_ = true;
        last_timestamp_ = timestamp;
        return;
    }

    // 计算时间差
    double dt = timestamp - last_timestamp_;
    if (dt <= 0) {
        // 时间回退或重复，跳过预测，直接修正（或报错）
        dt = 0;
    }

    // 动态更新 transition matrix: x' = x + vx * dt
    cv::setIdentity(kf_.transitionMatrix);
    kf_.transitionMatrix.at<float>(0, 3) = static_cast<float>(dt);
    kf_.transitionMatrix.at<float>(1, 4) = static_cast<float>(dt);
    kf_.transitionMatrix.at<float>(2, 5) = static_cast<float>(dt);

    // 预测 + 修正
    kf_.predict();
    cv::Mat z = (cv::Mat_<float>(3, 1) << 
        measurement.x, measurement.y, measurement.z);
    kf_.correct(z);

    last_timestamp_ = timestamp;
}

cv::Point3f ArmorTracker_time::predictFuture(float dt_seconds) const {
    if (!initialized_) {
        return cv::Point3f(0, 0, 0);
    }

    // 手动外推：匀速模型
    cv::Mat state = kf_.statePost; // [x, y, z, vx, vy, vz]^T
    float x = state.at<float>(0) + state.at<float>(3) * dt_seconds;
    float y = state.at<float>(1) + state.at<float>(4) * dt_seconds;
    float z = state.at<float>(2) + state.at<float>(5) * dt_seconds;

    return cv::Point3f(x, y, z);
}

cv::Point3f ArmorTracker_time::getCurrentEstimate() const {
    if (!initialized_) {
        return cv::Point3f(0, 0, 0);
    }
    return cv::Point3f(
        kf_.statePost.at<float>(0),
        kf_.statePost.at<float>(1),
        kf_.statePost.at<float>(2)
    );
}

bool ArmorTracker_time::isInitialized() const {
    return initialized_;
}
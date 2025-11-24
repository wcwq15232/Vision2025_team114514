#include <opencv2/opencv.hpp>
#include <queue>

#include "basic_types.hpp"
#include "kalmanFilater.hpp"


ArmorKalmanFilter::ArmorKalmanFilter() {
    // 状态向量: [x, y, vx, vy]
    kf = cv::KalmanFilter(4, 2, 0);

    // 状态转移矩阵 F (匀速模型)
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1
    );

    // 测量矩阵 H: 只观测 x, y
    cv::setIdentity(kf.measurementMatrix);

    // 测量噪声协方差 R (可根据实际调整)
    kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) << 
        10.0f, 0,
        0, 10.0f
    );

    // 过程噪声协方差 Q (可根据运动剧烈程度调整)
    kf.processNoiseCov = (cv::Mat_<float>(4, 4) << 
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0.1f, 0,
        0, 0, 0, 0.1f
    );

    // 初始状态协方差 P
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1e-2));

    // 初始状态设为0（后续在第一次观测时初始化）
    kf.statePost.setTo(cv::Scalar::all(0));
    initialized = false;
}

cv::Point2f ArmorKalmanFilter::correct(const Armor& armor) {
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << armor.center.x, armor.center.y);

    if (!initialized) {
        // 第一次观测：初始化状态
        kf.statePost.at<float>(0) = armor.center.x;
        kf.statePost.at<float>(1) = armor.center.y;
        kf.statePost.at<float>(2) = 0; // 初始速度为0
        kf.statePost.at<float>(3) = 0;
        initialized = true;
    }

    // 预测
    cv::Mat prediction = kf.predict();

    // 更新（校正）
    cv::Mat estimate = kf.correct(measurement);

    return cv::Point2f(estimate.at<float>(0), estimate.at<float>(1));
}

cv::Point2f ArmorKalmanFilter::getPredictedPosition() const {
    if (!initialized) return cv::Point2f(0, 0);
    cv::Mat prediction = kf.statePre; // 上次 predict() 后的结果
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}
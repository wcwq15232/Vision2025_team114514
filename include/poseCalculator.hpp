#ifndef POSE_CALCULATOR
#define POSE_CALCULATOR

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

struct PoseResult {
    cv::Point3f position;  // 位置
    cv::Vec3d rotation;       // 旋转向量
    double distance;          // 距离
    bool valid;
};

class PoseCalculator {
private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Point3f> objectPoints; // 矩形在世界坐标系中的3D点
public:
    PoseCalculator(float fx, float fy, float cx, float cy, float rect_width, float rect_height);
    std::pair<double, double> getOrientationAngles(const std::vector<cv::Point2f>& imagePoints);
    PoseResult getPose(const std::vector<cv::Point2f>& imagePoints);
};

#endif
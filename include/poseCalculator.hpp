#ifndef POSE_CALCULATOR
#define POSE_CALCULATOR

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <basic_types.hpp>

class PoseCalculator {
private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Point3f> objectPoints; // 矩形在世界坐标系中的3D点
public:
    PoseCalculator(float fx, float fy, float cx, float cy, float rect_width, float rect_height);
    PoseResult getPose(const std::vector<cv::Point2f>& imagePoints);
};

#endif
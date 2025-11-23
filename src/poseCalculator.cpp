#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <poseCalculator.hpp>
    
using namespace cv;

PoseCalculator::PoseCalculator(float fx, float fy, float cx, float cy, 
                               float rect_width, float rect_height) {
    cameraMatrix = (Mat_<double>(3, 3) << 
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);
    distCoeffs = Mat::zeros(5, 1, CV_64F);
    
    // 定义矩形在 YOUR 坐标系中：X-right, Y-forward, Z-up
    // 矩形位于 Y=0 平面（正对相机），X 横向，Z 纵向（向上为正）
    objectPoints.push_back(Point3f(-rect_width/2, 0.0f,  rect_height/2));  // 左上
    objectPoints.push_back(Point3f( rect_width/2, 0.0f,  rect_height/2));  // 右上
    objectPoints.push_back(Point3f( rect_width/2, 0.0f, -rect_height/2));  // 右下
    objectPoints.push_back(Point3f(-rect_width/2, 0.0f, -rect_height/2));  // 左下
}

PoseResult PoseCalculator::getPose(const std::vector<Point2f>& imagePoints) {
    PoseResult result;
    result.valid = false;
    
    if (imagePoints.size() != 4) {
        std::cerr << "错误: 需要4个图像点" << std::endl;
        return result;
    }
    
    Mat rvec, tvec;
    bool success = solvePnP(objectPoints, imagePoints, 
                            cameraMatrix, distCoeffs, 
                            rvec, tvec, false, SOLVEPNP_IPPE_SQUARE); // 或 SOLVEPNP_ITERATIVE
    
    if (!success) {
        std::cerr << "PnP求解失败" << std::endl;
        return result;
    }

    // 现在 tvec 已经是在你的坐标系中：X-right, Y-forward, Z-up
    result.position = Point3f(
        static_cast<float>(tvec.at<double>(0)),
        static_cast<float>(tvec.at<double>(2)),
        static_cast<float>(-tvec.at<double>(1))
    );

    // 计算旋转矩阵
    Mat R;
    Rodrigues(rvec, R);

    // 提取 yaw (绕 Z 轴) 和 pitch (绕 X 轴)
    // 假设使用 Yaw-Pitch-Roll (Z-Y'-X'') 顺序，但这里我们只关心 yaw 和 pitch
    // Yaw: 绕 Z 轴（水平旋转）
    result.yaw = static_cast<float>(atan2(R.at<double>(1,0), R.at<double>(0,0)));

    // Pitch: 绕 X 轴（俯仰）
    result.pitch = static_cast<float>(atan2(-R.at<double>(2,1), R.at<double>(2,2)));

    result.distance = static_cast<double>(norm(result.position));
    result.valid = true;
    return result;
}
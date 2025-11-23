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
    
    // 设置矩形的3D世界坐标点

    objectPoints.push_back(Point3f(-rect_width/2, rect_height/2, 0.0f));  // 左上
    objectPoints.push_back(Point3f(rect_width/2, rect_height/2, 0.0f));   // 右上
    objectPoints.push_back(Point3f(rect_width/2, -rect_height/2, 0.0f));  // 右下
    objectPoints.push_back(Point3f(-rect_width/2, -rect_height/2, 0.0f)); // 左下
}
    
// 主要函数：从2D图像点计算矩形中心在相机坐标系中的位置
PoseResult PoseCalculator::getPose(const std::vector<Point2f>& imagePoints) {
    PoseResult result;
    result.valid = false;
    
    if (imagePoints.size() != 4) {
        std::cerr << "错误: 需要4个图像点来计算矩形姿态" << std::endl;
        return result;
    }
    
    Mat rvec, tvec;
    bool success = solvePnP(objectPoints, imagePoints, 
                                cameraMatrix, distCoeffs, 
                                rvec, tvec, true, SOLVEPNP_IPPE);
    
    if (!success) {
        std::cerr << "PnP求解失败" << std::endl;
        return result;
    }
    
    result.position = Point3f(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    result.rotation = Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    result.distance = norm(result.position);
    result.valid = true;
    
    return result;
}
    
// 计算方位角（相对于相机光轴的角度）
std::pair<double, double> PoseCalculator::getOrientationAngles(const std::vector<Point2f>& imagePoints) {
    auto pose = getPose(imagePoints);
    if (!pose.valid) {
        return std::make_pair(0.0, 0.0);
    }
    
    // 将旋转向量转换为旋转矩阵
    Mat rotationMatrix;
    Rodrigues(pose.rotation, rotationMatrix);
    
    // 计算欧拉角（简化版本）
    double pitch = std::atan2(-rotationMatrix.at<double>(2, 0),
                                std::sqrt(rotationMatrix.at<double>(0, 0)*rotationMatrix.at<double>(0, 0) + 
                                        rotationMatrix.at<double>(1, 0)*rotationMatrix.at<double>(1, 0)));
    double yaw = std::atan2(rotationMatrix.at<double>(1, 0), rotationMatrix.at<double>(0, 0));
    
    return std::make_pair(pitch, yaw); // 俯仰角，偏航角（弧度）
}


// // 使用示例
// int main() {
//     // 假设你已经识别出矩形的4个角点 (顺序：左下、右下、右上、左上)
//     std::vector<Point2f> detected_corners = {
//         Point2f(100, 200), // 左下
//         Point2f(300, 200), // 右下
//         Point2f(300, 100), // 右上
//         Point2f(100, 100)  // 左上
//     };
    

//     float fx = 381.4f; // 焦距x (需要根据实际相机标定调整)
//     float fy = 381.4f; // 焦距y (需要根据实际相机标定调整)
//     float cx = 320.0f; // 主点x
//     float cy = 240.0f; // 主点y
    
//     float rect_width = 0.705f;   // 0.2米
//     float rect_height = 0.230f; // 0.15米
    
//     // 使用完整类获取更多信息
//     PoseCalculator estimator(fx, fy, cx, cy, rect_width, rect_height);
//     PoseResult pose = estimator.getPose(detected_corners);
    
//     if (pose.valid) {
//         std::cout << "\n详细姿态信息:" << std::endl;
//         std::cout << "距离: " << pose.distance << " 米" << std::endl;

//         std::cout << "矩形中心在相机坐标系中的位置:" << std::endl;
//         std::cout << "X: " << pose.position.x << " 米" << std::endl;
//         std::cout << "Y: " << pose.position.y << " 米" << std::endl;
//         std::cout << "Z: " << pose.position.z << " 米 (距离)" << std::endl;
        
//         auto angles = estimator.getOrientationAngles(detected_corners);
//         std::cout << "俯仰角: " << angles.first * 180.0 / CV_PI << " 度" << std::endl;
//         std::cout << "偏航角: " << angles.second * 180.0 / CV_PI << " 度" << std::endl;
//     }
    
//     return 0;
// }
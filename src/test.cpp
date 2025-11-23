#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class RectanglePoseEstimator {
private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Point3f> objectPoints; // 矩形在世界坐标系中的3D点

public:
    // 构造函数，设置相机内参
    RectanglePoseEstimator(float fx, float fy, float cx, float cy) {
        // 相机内参矩阵 (需要根据你的相机参数调整)
        cameraMatrix = (cv::Mat_<double>(3, 3) << 
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0);
        
        // 畸变系数 (假设无畸变)
        distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        
        // 设置矩形的3D世界坐标点 (以矩形中心为原点)
        // 假设矩形尺寸为 width x height
        float rect_width = 0.2f;  // 0.2米 (根据你的实际矩形尺寸调整)
        float rect_height = 0.15f; // 0.15米 (根据你的实际矩形尺寸调整)
        
        // 顺序：左下、右下、右上、左上 (对应图像中的点顺序)
        objectPoints.push_back(cv::Point3f(-rect_width/2, -rect_height/2, 0.0f)); // 左下
        objectPoints.push_back(cv::Point3f(rect_width/2, -rect_height/2, 0.0f));  // 右下
        objectPoints.push_back(cv::Point3f(rect_width/2, rect_height/2, 0.0f));   // 右上
        objectPoints.push_back(cv::Point3f(-rect_width/2, rect_height/2, 0.0f));  // 左上
    }
    
    // 重载构造函数，允许自定义矩形尺寸
    RectanglePoseEstimator(float fx, float fy, float cx, float cy, 
                          float rect_width, float rect_height) {
        cameraMatrix = (cv::Mat_<double>(3, 3) << 
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0);
        distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        
        // 设置矩形的3D世界坐标点
        objectPoints.push_back(cv::Point3f(-rect_width/2, -rect_height/2, 0.0f)); // 左下
        objectPoints.push_back(cv::Point3f(rect_width/2, -rect_height/2, 0.0f));  // 右下
        objectPoints.push_back(cv::Point3f(rect_width/2, rect_height/2, 0.0f));   // 右上
        objectPoints.push_back(cv::Point3f(-rect_width/2, rect_height/2, 0.0f));  // 左上
    }
    
    // 主要函数：从2D图像点计算矩形中心在相机坐标系中的位置
    cv::Point3f getRectangleCenterPosition(const std::vector<cv::Point2f>& imagePoints) {
        if (imagePoints.size() != 4) {
            std::cerr << "错误：需要4个图像点来计算矩形姿态" << std::endl;
            return cv::Point3f(0, 0, 0);
        }
        
        // 执行PnP求解
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(objectPoints, imagePoints, 
                                   cameraMatrix, distCoeffs, 
                                   rvec, tvec, true, cv::SOLVEPNP_IPPE); // IPPE适合矩形
        
        if (!success) {
            std::cerr << "PnP求解失败" << std::endl;
            return cv::Point3f(0, 0, 0);
        }
        
        // tvec就是矩形中心在相机坐标系中的位置
        cv::Point3f center_position(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        
        // 验证结果的合理性（可选）
        if (center_position.z < 0.05 || center_position.z > 10.0) { // 距离在0.05-10米之间
            std::cerr << "警告：检测到的距离可能不合理: " << center_position.z << "米" << std::endl;
        }
        
        return center_position;
    }
    
    // 获取完整的姿态信息（旋转和平移）
    struct PoseResult {
        cv::Point3f translation;  // 位置
        cv::Vec3d rotation;       // 旋转向量
        double distance;          // 距离
        bool valid;
    };
    
    PoseResult getFullPose(const std::vector<cv::Point2f>& imagePoints) {
        PoseResult result;
        result.valid = false;
        
        if (imagePoints.size() != 4) {
            std::cerr << "错误：需要4个图像点来计算矩形姿态" << std::endl;
            return result;
        }
        
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(objectPoints, imagePoints, 
                                   cameraMatrix, distCoeffs, 
                                   rvec, tvec, true, cv::SOLVEPNP_IPPE);
        
        if (!success) {
            std::cerr << "PnP求解失败" << std::endl;
            return result;
        }
        
        result.translation = cv::Point3f(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        result.rotation = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
        result.distance = cv::norm(result.translation);
        result.valid = true;
        
        return result;
    }
    
    // 计算方位角（相对于相机光轴的角度）
    std::pair<double, double> getOrientationAngles(const std::vector<cv::Point2f>& imagePoints) {
        auto pose = getFullPose(imagePoints);
        if (!pose.valid) {
            return std::make_pair(0.0, 0.0);
        }
        
        // 将旋转向量转换为旋转矩阵
        cv::Mat rotationMatrix;
        cv::Rodrigues(pose.rotation, rotationMatrix);
        
        // 计算欧拉角（简化版本）
        double pitch = std::atan2(-rotationMatrix.at<double>(2, 0),
                                 std::sqrt(rotationMatrix.at<double>(0, 0)*rotationMatrix.at<double>(0, 0) + 
                                          rotationMatrix.at<double>(1, 0)*rotationMatrix.at<double>(1, 0)));
        double yaw = std::atan2(rotationMatrix.at<double>(1, 0), rotationMatrix.at<double>(0, 0));
        
        return std::make_pair(pitch, yaw); // 俯仰角，偏航角（弧度）
    }
};

// 便捷函数：直接获取矩形中心位置
cv::Point3f estimateRectanglePosition(const std::vector<cv::Point2f>& corners_2d,
                                     float fx, float fy, float cx, float cy,
                                     float rect_width, float rect_height) {
    RectanglePoseEstimator estimator(fx, fy, cx, cy, rect_width, rect_height);
    return estimator.getRectangleCenterPosition(corners_2d);
}

// 使用示例
int main() {
    // 假设你已经识别出矩形的4个角点 (顺序：左下、右下、右上、左上)
    std::vector<cv::Point2f> detected_corners = {
        cv::Point2f(100, 200), // 左下
        cv::Point2f(300, 200), // 右下
        cv::Point2f(300, 100), // 右上
        cv::Point2f(100, 100)  // 左上
    };
    
    // 根据你的Gazebo相机参数设置
    // 从你的URDF文件中可以看到，图像尺寸是640x480，FPS=90
    float fx = 320.0f; // 焦距x (需要根据实际相机标定调整)
    float fy = 320.0f; // 焦距y (需要根据实际相机标定调整)
    float cx = 320.0f; // 主点x
    float cy = 240.0f; // 主点y
    
    // 矩形的实际尺寸 (根据你的实际目标调整)
    float rect_width = 0.2f;   // 0.2米
    float rect_height = 0.15f; // 0.15米
    
    // 计算矩形中心位置
    cv::Point3f center_pos = estimateRectanglePosition(detected_corners, 
                                                      fx, fy, cx, cy, 
                                                      rect_width, rect_height);
    
    std::cout << "矩形中心在相机坐标系中的位置:" << std::endl;
    std::cout << "X: " << center_pos.x << " 米" << std::endl;
    std::cout << "Y: " << center_pos.y << " 米" << std::endl;
    std::cout << "Z: " << center_pos.z << " 米 (距离)" << std::endl;
    
    // 使用完整类获取更多信息
    RectanglePoseEstimator estimator(fx, fy, cx, cy, rect_width, rect_height);
    auto full_pose = estimator.getFullPose(detected_corners);
    
    if (full_pose.valid) {
        std::cout << "\n详细姿态信息:" << std::endl;
        std::cout << "距离: " << full_pose.distance << " 米" << std::endl;
        
        auto angles = estimator.getOrientationAngles(detected_corners);
        std::cout << "俯仰角: " << angles.first * 180.0 / CV_PI << " 度" << std::endl;
        std::cout << "偏航角: " << angles.second * 180.0 / CV_PI << " 度" << std::endl;
    }
    
    return 0;
}
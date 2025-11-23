#ifndef TOOLS_MY
#define TOOLS_MY

#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <deque>

extern std::vector<cv::Point2f> orderPoints(std::vector<cv::Point2f> points);
extern void draw4points(const std::vector<cv::Point2f> &points, cv::Mat &result_image);
extern inline void drawRotatedRect(cv::RotatedRect &rect, cv::Mat img, cv::Scalar color = cv::Scalar(20, 255, 20));
extern inline void drawRotatedRect(std::vector<cv::Point2f> points, cv::Mat img, cv::Scalar color = cv::Scalar(20, 255, 20));
extern void adjustRotatedRect(cv::RotatedRect& rect);

extern std::vector<cv::Scalar> point_colors;
std::vector<cv::Scalar> point_colors = {
    cv::Scalar(255, 0, 0),   // 蓝色 - 1
    cv::Scalar(0, 255, 0),   // 绿色 - 2
    cv::Scalar(0, 255, 255), // 黄色 - 3
    cv::Scalar(255, 0, 255)  // 紫色 - 4
};

void adjustRotatedRect(cv::RotatedRect& rect) {
    // 统一矩形角度所指向的方向
    if (rect.size.width > rect.size.height) {
        // 交换width和height
        float temp = rect.size.width;
        rect.size.width = rect.size.height;
        rect.size.height = temp;
        
        // 角度调整 +90度
        rect.angle += 90.0;
        if (rect.angle >= 180.0) {
            rect.angle -= 180.0;
        }
    }
}

std::vector<cv::Point2f> orderPoints(std::vector<cv::Point2f> points) {
    std::vector<cv::Point2f> ordered(4);
    // 给点排序， 对于矩形是 左下逆时针到左上

    std::vector<float> sum, diff;
    for (const auto& p : points) {
        sum.push_back(p.x + p.y);
        diff.push_back(p.x - p.y);
    }
    
    ordered[3] = points[min_element(sum.begin(), sum.end()) - sum.begin()];
    ordered[2] = points[max_element(diff.begin(), diff.end()) - diff.begin()];
    ordered[1] = points[max_element(sum.begin(), sum.end()) - sum.begin()];
    ordered[0] = points[min_element(diff.begin(), diff.end()) - diff.begin()];

    return ordered;
}

void draw4points(const std::vector<cv::Point2f> &points, cv::Mat &result_image){
    for (int j = 0; j < 4; j++)
    {
        cv::circle(result_image, points[j], 6, point_colors[j], -1);
        cv::circle(result_image, points[j], 6, cv::Scalar(0, 0, 0), 2);

        // 标注序号
        std::string point_text = std::to_string(j + 1);
        cv::putText(
            result_image, point_text,
            cv::Point(points[j].x + 10, points[j].y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
        cv::putText(
            result_image, point_text,
            cv::Point(points[j].x + 10, points[j].y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);
    }
}

inline void connectPoints(const std::deque<cv::Point2f> &points, cv::Mat &result_image){
    for (auto it = points.begin(); it != points.end() - 1; ++it)
        cv::line(result_image, *it, *(it + 1), cv::Scalar(10, 0, 10));
}

inline void drawRotatedRect(cv::RotatedRect &rect, cv::Mat img, cv::Scalar color){
    cv::Point2f points[4];
    rect.points(points);
    for (int i = 0; i < 4; i++)
        line(img, points[i], points[(i + 1) % 4], color, 2);
}

inline void drawRotatedRect(std::vector<cv::Point2f> points, cv::Mat img, cv::Scalar color){
    for (int i = 0; i < 4; i++) line(img, points[i], points[(i + 1) % 4], color, 2);
}

inline double get_hit_angle(double v, double target_x, double target_y, double g) {
    constexpr double eps = 1e-12;

    const double x = target_x;
    const double y = target_y;

    const double v2 = v * v;
    const double gx2_over_2v2 = g * x * x / (2.0 * v2);

    const double a = gx2_over_2v2;
    const double b = -x;
    const double c = y + gx2_over_2v2;

    const double discriminant = b * b - 4.0 * a * c;

    std::cout << v << std::endl;
    std::cout << target_x << std::endl;
    std::cout << target_y << std::endl;
    std::cout << g << std::endl;
    std::cout << discriminant << std::endl;

    if (discriminant < 0.0) {
        return std::numeric_limits<double>::quiet_NaN(); // 无解
    }

    const double sqrtD = std::sqrt(discriminant);
    double T1 = (-b + sqrtD) / (2.0 * a);
    double T2 = (-b - sqrtD) / (2.0 * a);

    double theta1 = std::atan(T1);
    double theta2 = std::atan(T2);

    // 确保角度为正（向上发射）
    if (theta1 < 0) theta1 += M_PI; // 理论上不会出现（x>0, v>0），但保险起见
    if (theta2 < 0) theta2 += M_PI;

    // 只取 0 到 π/2 范围内的有效角度（向上发射）
    double valid_theta1 = (theta1 >= 0 && theta1 <= M_PI/2) ? theta1 : -1;
    double valid_theta2 = (theta2 >= 0 && theta2 <= M_PI/2) ? theta2 : -1;

    // 返回最小的有效角度
    if (valid_theta1 < 0 && valid_theta2 < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (valid_theta1 < 0) {
        return valid_theta2;
    } else if (valid_theta2 < 0) {
        return valid_theta1;
    } else {
        return std::min(valid_theta1, valid_theta2);
    }
}

#endif
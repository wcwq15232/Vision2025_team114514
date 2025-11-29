#ifndef BASIC_TYPES
#define BASIC_TYPES

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

// 本文件定义程序所需的  存储各类目标状态的结构体

// std::vector<std::string> obj_names {
//     "sphere",
//     "armor",
//     "armor_red_2",
//     "armor_red_3",
//     "armor_red_4",
//     "armor_red_5",
//     "rect"
// };
// "rect_move"

struct PoseResult {
    cv::Point3f position;  // 位置
    cv::Vec3d rotation;       // 旋转向量
    double distance;          // 距离
    bool valid;
    float yaw;
    float pitch;
};

struct Circle{
    cv::Point2f center;
    float radius1 = 0;
    float radius2 = 0;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
};

struct Sphere {
    cv::Point2f center;
    float radius = 0;
    std::vector<cv::Point2f> points;
};

struct Armor {
    cv::Point2f center;
    int number;
    float angel;
    float width;
    float height;
    std::vector<cv::Point2f> points;  // 要发送的4个点
    std::vector<cv::Point2f> points2; // 该组点用于提取数字
    PoseResult pose;  // pnp解算结果
    float predit_time;  // 提前量(s)
    float yaw;  // 射击角度
    float pitch;  // 射击角度
    float roll;  // 射击角度
};

struct Light {
    cv::Point2f center;
    cv::Point2f top;
    cv::Point2f button;
    cv::Point2f top2;  // 与armor的points2一致用途
    cv::Point2f button2;
    float angle;
    float width;
    float height;
};

struct Rect_s {
    cv::Point2f center;
    std::vector<cv::Point2f> points;
};

#endif
#ifndef BASIC_TYPES
#define BASIC_TYPES

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

#include "poseCalculator.hpp"

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
    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> points2;
    PoseResult pose;
    float predit_time;
    float yaw;
    float pitch;
    float roll;
};

struct Light {
    cv::Point2f center;
    cv::Point2f top;
    cv::Point2f button;
    cv::Point2f top2;
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
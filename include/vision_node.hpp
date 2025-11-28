#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <deque>

#include <sensor_msgs/image_encodings.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "referee_pkg/msg/race_stage.hpp"
#include <std_msgs/msg/header.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include "referee_pkg/srv/hit_armor.hpp"

#include "poseCalculator.hpp"
#include "kalmanFilater.hpp"
#include "basic_types.hpp"
#include "trace.hpp"

constexpr int OBJECT_CLASS_COUNT = 7;

float ARMOR_WIDTH = 0.705f;
float ARMOR_HEIGHT = 0.230f;
int FPS = 90;

float FX = 554.383f; // 焦距x
float FY = 554.383; // 焦距y
float CX = 320.0f; // 主点x
float CY = 320.0f; // 主点y

Mat kernel_3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
Mat kernel_5 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
Mat kernel_7 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

class TestNode : public rclcpp::Node
{
public:
    TestNode(std::string name);
    ~TestNode();
private:
    int stage;
    ArmorTracker kalman_filter;
    ArmorTracker_time kalman_filter_time;

    std::vector<Mat> num_imgs;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub;
    rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr Target_pub;
    rclcpp::Subscription<referee_pkg::msg::RaceStage>::SharedPtr Stage_sub;
    rclcpp::Service<referee_pkg::srv::HitArmor>::SharedPtr Hit_srv;

    Mat src, hsv, img_result, rectMask, mask;
    std::deque<Point2f> rect_history_points {std::deque<Point2f>(10)};
    int miss_count_rect;

    std::vector<Sphere> sphere_list;
    std::vector<Rect_s> rect_list;
    std::vector<Light>  light_list;
    std::vector<Armor>  armor_list;

    PoseCalculator poseCalculator {FX, FY, CX, CY, ARMOR_WIDTH, ARMOR_HEIGHT};

    std::vector<std::vector<Point2f> > currentDetections{OBJECT_CLASS_COUNT};
    std::vector<Trace> traceList;

    void callback_camera(sensor_msgs::msg::Image::SharedPtr msg);
    void callback_stage_change(referee_pkg::msg::RaceStage::SharedPtr msg);
    void callback_hit_srv(referee_pkg::srv::HitArmor_Request::SharedPtr request, referee_pkg::srv::HitArmor_Response::SharedPtr response);
    void preprocess(Mat &src, Mat &result);

    void getSphere(std::vector<std::vector<Point>> &contours);
    void calculateStableSpherePoints(Sphere &sphere);

    void getRect(std::vector<std::vector<Point>> &contours);
    void predit_hit();

    void getArmor(std::vector<std::vector<Point>> &contours);
    void getLights(std::vector<std::vector<Point>> &contours);
    void matchLights();
    void getNumberImg(Armor &armor, Mat &num_img);
    int matchNum(Mat &num_img);
    void getArmorPose();

    void sendResult(sensor_msgs::msg::Image::SharedPtr msg);
    void showResult();

    void drawTrackObjectID();
};

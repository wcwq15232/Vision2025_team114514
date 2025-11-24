#include <cmath>
#include <cassert>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "referee_pkg/msg/race_stage.hpp"
#include <std_msgs/msg/header.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include "referee_pkg/srv/hit_armor.hpp"

#include <vector>
#include <string>
#include <deque>
#include <map>

#include "basic_types.hpp"

#include "vision_node.hpp"
#include "tools.hpp"

using std::map, std::vector, std::string, std::cin, std::cout, std::endl, std::to_string;
using namespace rclcpp;
using namespace cv;


#include "poseCalculator.hpp"
#include "kalmanFilater.hpp"

int max_history_length = 20;
int abandon_count = 20;
string NUMS_PATH = "install/team_challenge/share/team_challenge/nums/";
double V = 15;

TestNode::TestNode(string name) : Node(name){
    RCLCPP_INFO(this->get_logger(), "Initializing TestNode");

    Image_sub = this->create_subscription<sensor_msgs::msg::Image>("/camera/image_raw", 10, bind(&TestNode::callback_camera, this, std::placeholders::_1));
    Target_pub = this->create_publisher<referee_pkg::msg::MultiObject>("/vision/target", 10);
    Stage_sub = this->create_subscription<referee_pkg::msg::RaceStage>("/referee/race_stage", 10, bind(&TestNode::callback_stage_change, this, std::placeholders::_1));
    Hit_srv = this->create_service<referee_pkg::srv::HitArmor>("/referee/hit_arror", bind(&TestNode::callback_hit_srv, this, std::placeholders::_1, std::placeholders::_2));

    namedWindow("Detection Result", WINDOW_AUTOSIZE);

    RCLCPP_INFO(this->get_logger(), "TestNode initialized successfully");

    for (int i = 0; i < 5; ++i){
        num_imgs.push_back(imread(NUMS_PATH + to_string(i + 1) + ".jpg", IMREAD_REDUCED_GRAYSCALE_4));
        assert(!num_imgs[i].empty());
        cout << "长" << num_imgs[i].cols << "宽" << num_imgs[i].rows << endl;
        RCLCPP_INFO(this->get_logger(), "num%d img read successfully", i + 1);
    }
}

TestNode::~TestNode() { destroyWindow("Detection Result"); }

void TestNode::calculateStableSpherePoints(Sphere &sphere){
    sphere.points.push_back(Point2f(sphere.center.x - sphere.radius, sphere.center.y)); // 左点 (1)
    sphere.points.push_back(Point2f(sphere.center.x, sphere.center.y + sphere.radius)); // 下点 (2)
    sphere.points.push_back(Point2f(sphere.center.x + sphere.radius, sphere.center.y)); // 右点 (3)
    sphere.points.push_back(Point2f(sphere.center.x, sphere.center.y - sphere.radius)); // 上点 (4)
}

void TestNode::callback_camera(sensor_msgs::msg::Image::SharedPtr msg){
    cv_bridge::CvImagePtr cv_ptr;

    if (msg->encoding == "rgb8" || msg->encoding == "R8G8B8"){
        Mat image(msg->height, msg->width, CV_8UC3,
                        const_cast<unsigned char *>(msg->data.data()));
        Mat bgr_image;
        cvtColor(image, bgr_image, COLOR_RGB2BGR);
        cv_ptr = std::make_shared<cv_bridge::CvImage>();
        cv_ptr->header = msg->header;
        cv_ptr->encoding = "bgr8";
        cv_ptr->image = bgr_image;
    }
    else
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    src = cv_ptr->image;

    if (src.empty()){
        RCLCPP_WARN(this->get_logger(), "Received empty image");
        return;
    }

    // 创建结果图像
    img_result = src.clone();
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // 找矩形

    // std::thread thread_([this] () -> void {
        vector<vector<Point>> rect_contours;

        inRange(hsv, Scalar(75, 140, 215), Scalar(100, 255, 255), rectMask);
        morphologyEx(rectMask, rectMask, MORPH_CLOSE, kernel_3);
        morphologyEx(rectMask, rectMask, MORPH_OPEN, kernel_3);
        findContours(rectMask, rect_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        getRect(rect_contours);
    // });


    // 找圆和装甲板，目前俩色差不多所以就共用预处理了
    
    // 示例的预处理
    // Mat mask1, mask2, mask;
    // inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
    // inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
    // mask = mask1 | mask2;

    vector<vector<Point>> red_contours;

    Mat channels[3];
    split(src, channels);
    mask = channels[2] - channels[0];

    // morphologyEx(mask, mask, MORPH_CLOSE, kernel_3);
    // morphologyEx(mask, mask, MORPH_OPEN, kernel_3);

    threshold(mask, mask, 75, 255, THRESH_BINARY);
    dilate(mask, mask, kernel_3);
    // imshow("a", mask);

    findContours(mask, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    getSphere(red_contours);
    getArmor(red_contours);
    getArmorPose();

    // thread_.join();
    // draw_history_points();

    // 展示结果图这一块
    showResult();
    sendResult(msg);
}

void TestNode::callback_stage_change(referee_pkg::msg::RaceStage::SharedPtr msg) {
    stage = msg->stage;
    RCLCPP_INFO(this->get_logger(), "stage changed to %d", stage);
}

void TestNode::callback_hit_srv(referee_pkg::srv::HitArmor_Request::SharedPtr request, referee_pkg::srv::HitArmor_Response::SharedPtr response) {
    vector<Point2f> points(4);

    for (int i = 0; i < 4; ++i){
        points[i].x = request->modelpoint[i].x;
        points[i].y = request->modelpoint[i].y;
    }
    
    PoseResult result = poseCalculator.getPose(points);
    Point3f &tmp = result.position;

    // TODO 卡尔曼滤波做预测 TODO

    response->yaw = result.yaw;
    response->pitch = get_hit_angle(V, sqrt(tmp.x * tmp.x + tmp.y * tmp.y), tmp.z, request->g);
    response->roll = 0;
}


void TestNode::preprocess(Mat &src, Mat &result){
    // 阿巴阿巴，或许以后有用呢（
}

void TestNode::getSphere(vector<vector<Point>> &contours){
    sphere_list.clear();
    cv_bridge::CvImagePtr cv_ptr;

    int valid_spheres = 0;

    for (size_t i = 0; i < contours.size(); i++){
        double area = contourArea(contours[i]);
        if (area < 450)
            continue;

        // 计算最小外接圆
        Point2f center;
        float radius = 0;
        minEnclosingCircle(contours[i], center, radius);

        // 计算圆形度
        double perimeter = arcLength(contours[i], true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);

        if (circularity > 0.7 && radius > 15 && radius < 200){
            Sphere tmp_sphere {center, radius};
            calculateStableSpherePoints(tmp_sphere);

            // 绘制检测到的球体
            circle(img_result, center, static_cast<int>(radius), Scalar(0, 255, 0), 2); // 绿色圆圈
            circle(img_result, center, 3, Scalar(0, 0, 255), -1);                       // 红色圆心

            // 绘制球体上的四个点
            vector<string> point_names = {"左", "下", "右", "上"};
            draw4points(tmp_sphere.points, img_result);

            // 显示半径信息
            string info_text = "R:" + to_string((int)radius);
            putText(
                img_result, info_text, Point(center.x - 15, center.y + 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

            valid_spheres++;
            RCLCPP_INFO(this->get_logger(),
                        "Found sphere: (%.1f, %.1f) R=%.1f C=%.3f", center.x,
                        center.y, radius, circularity);

            // 添加到发送列表
            sphere_list.push_back(tmp_sphere);


            for (int j = 0; j < 4; j++){
                RCLCPP_INFO(this->get_logger(),
                            "Sphere %d, Point %d (%s): (%.1f, %.1f)",
                            valid_spheres + 1, j + 1, point_names[j].c_str(),
                            tmp_sphere.points[j].x, tmp_sphere.points[j].y);
            }

        }
    }
}

void TestNode::getRect(vector<vector<Point>> &contours){
    rect_list.clear();

    int valid_rects = 0;

    cv_bridge::CvImagePtr cv_ptr;

    for (int i = 0; i < contours.size(); ++i)
    {
        float area = contourArea(contours[i]);
        float area2 = 0;

        // 逼近就不用了
        // vector<vector<Point>> conPoly(contours.size());
        vector<RotatedRect> boundRect(contours.size());
        string objectType = "";

        if (area < 50) continue;

        float peri = arcLength(contours[i], true);
        // approxPolyDP(contours[i], conPoly[i], 0.03 * peri, true);
        // cout << contours[i].size() << endl;
        boundRect[i] = minAreaRect(contours[i]);
        area2 = boundRect[i].size.width * boundRect[i].size.height;

        // RCLCPP_INFO(this->get_logger(), to_string(area).c_str());
        // RCLCPP_INFO(this->get_logger(), to_string(area2).c_str());
        if (area / area2 > 0.80){
            Point2f vertices[4];
            boundRect[i].points(vertices);

            vector<Point2f> vertices_vec;
            vector<Point2f> vertices_vec_ordered;
            for (int i = 0; i < 4; i++) {
                vertices_vec.push_back(Point(vertices[i]));
            }

            vector<string> point_names = {"左下", "右下", "右上", "左上"};
            vertices_vec_ordered = orderPoints(vertices_vec);

            Rect_s tmp;
            tmp.points = vertices_vec_ordered;
            tmp.center = boundRect[i].center;

            rect_list.push_back(tmp);

            drawRotatedRect(boundRect[i], img_result);
            draw4points(vertices_vec_ordered, img_result);

            RCLCPP_INFO(this->get_logger(),
                "Found RECT: (%.1f, %.1f) 宽%.1f 高%.3f", boundRect[i].center.x,
                boundRect[i].center.y, boundRect[i].size.width, boundRect[i].size.height);
            ++valid_rects;
            for (int j = 0; j < 4; j++){
                RCLCPP_INFO(this->get_logger(),
                            "Rect %d, Point %d (%s): (%.1f, %.1f)",
                            valid_rects + 1, j + 1, point_names[j].c_str(),
                            vertices_vec[j].x, vertices_vec[j].y);
            }
        }
    }
}

void TestNode::getLights(vector<vector<Point>> &contours)
{
    for (int i = 0; i < contours.size(); ++i)
    {
        int area = contourArea(contours[i]);
        if (area < 30)
            continue;
        // RotatedRect tmp = fitEllipse(contours[i]);
        RotatedRect tmp = minAreaRect(contours[i]);

        float div;

        adjustRotatedRect(tmp);
        div = tmp.size.height / tmp.size.width;
        if (div < 3.5 || div > 18.5 || (75 < tmp.angle && tmp.angle < 105))
            continue;
        
        Light light;
        light.center = tmp.center;
        light.width = tmp.size.width;
        light.height = tmp.size.height;
        light.angle = tmp.angle;
        
        // 按y坐标排序找到上下边的点
        Point2f vertices[4];
        tmp.points(vertices);
        
        vector<Point2f> sortedVertices(vertices, vertices + 4);
        std::sort(sortedVertices.begin(), sortedVertices.end(), 
                [](const Point2f& a, const Point2f& b) {
                    return a.y < b.y;
                });
        
        light.top = (sortedVertices[0] + sortedVertices[1]) * 0.5f;
        light.button = (sortedVertices[2] + sortedVertices[3]) * 0.5f;

        // 这个top2 button2是装甲板的坐标，给数字的变换用的
        // 这个参数可以调 目前效果还行
        tmp.size.height *= 1.95;
        tmp.points(vertices);
        vector<Point2f> sortedVertices2(vertices, vertices + 4);
        std::sort(sortedVertices2.begin(), sortedVertices2.end(), 
                [](const Point2f& a, const Point2f& b) {
                    return a.y < b.y;
                });
        
        light.top2 = (sortedVertices2[0] + sortedVertices2[1]) * 0.5f;
        light.button2 = (sortedVertices2[2] + sortedVertices2[3]) * 0.5f;

        light_list.push_back(light);


        // 自己试试这些玩意的效果  主要是debug用的

        // line(img_result, light.top, light.button, Scalar(255, 0, 0), 3);
        // circle(img_result, tmp.center, 3, Scalar(0, 255, 0), 4);

        // drawRotatedRect(img_result, tmp);            
        // putText(img_result , to_string(tmp.angle), tmp.center, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
        // putText(img_result , to_string(int(tmp.size.height)), tmp.center, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
        // putText(img_result , to_string(int(tmp.angle)) + " " + to_string(tmp.size.height / tmp.size.width), tmp.center, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 0), 2);

    }
}

void TestNode::matchLights()
{   
    if (light_list.size() <= 1)
        return;

    vector<bool> used(light_list.size(), false);

    int valid_armor = 0;
    int i, j;

    // cout << endl << "lllllll" << lights.size() << endl;
    // for (i = 0; i < lights.size(); ++i)
    //     putText(img_result , to_string(i + 1), Point2f(lights[i].center.x, lights[i].center.y + 40), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 0), 2);

    for (i = 0; i < light_list.size() - 1; ++i)
    {
        for (j = i + 1; j < light_list.size(); ++j)
        {  
            // cout << endl << to_string(i + 1) + " " + to_string(j + 1) << endl;
            // cout << abs(lights[i].angle - lights[j].angle) << endl;
            // cout << abs(lights[i].height / lights[j].height - 1) << endl;
            // cout << used[i] << " " << used[j] << endl;

            if (used[i] || used[j]) continue;
            float angel_diff = abs(light_list[i].angle - light_list[j].angle);
            if (angel_diff > 20 && angel_diff < 160) continue;
            if (abs(light_list[i].height / light_list[j].height - 1) > 0.35) continue;


            float distance = std::sqrt(std::pow(light_list[i].center.x - light_list[j].center.x, 2) + std::pow(light_list[i].center.y - light_list[j].center.y, 2));
            float avg_angle = (light_list[i].angle + light_list[j].angle) / 2.0;

            Armor armor;
            
            armor.center = Point2f((light_list[i].center.x + light_list[j].center.x) / 2 , (light_list[i].center.y + light_list[j].center.y) / 2);
            armor.points = {light_list[i].button, light_list[j].button, light_list[j].top, light_list[i].top};
            armor.points2 = {light_list[i].button2, light_list[j].button2, light_list[j].top2, light_list[i].top2};
            
            armor.angel = avg_angle;
            armor.width = distance;
            armor.height = (light_list[i].height + light_list[j].height) / 2;

            Mat num_img;

            getNumberImg(armor, num_img);
            // imshow("num of " + to_string(i + 1) + " " + to_string(j + 1), num_img);

            armor.number = matchNum(num_img);

            if (armor.number == -1) continue;

            ++valid_armor;
            used[i] = true;
            used[j] = true;

            armor_list.push_back(armor);
            drawRotatedRect(armor.points, img_result);
            draw4points(armor.points, img_result);
            
            putText(img_result , to_string(armor.number), Point2f(light_list[j].center.x + 15, light_list[j].center.y - 15), FONT_HERSHEY_SIMPLEX, 2.2, Scalar(0, 0, 0), 4.2);
            putText(img_result , to_string(armor.number), Point2f(light_list[j].center.x + 15, light_list[j].center.y - 15), FONT_HERSHEY_SIMPLEX, 2.2, Scalar(255, 255, 0), 2);

            // imshow("Armor " + to_string(armor_list.size()), num_img);
            vector<string> point_names = {"左下", "右下", "右上", "左上"};
            for (int j = 0; j < 4; j++){
                RCLCPP_INFO(this->get_logger(),
                            "Armor_%d %zu, Point %d (%s): (%.1f, %.1f)",
                            armor.number, armor_list.size(), j + 1, point_names[j].c_str(),
                            armor.points[j].x, armor.points[j].y);
            }
            // line(img_result, center1, center2, Scalar(0, 0, 0), 2);
            // circle(img_result, armor.center, 3, Scalar(0, 255, 0), 4);
        }
    }
}

void TestNode::getArmor(vector<vector<Point>> &contours)
{
    armor_list.clear();
    light_list.clear();

    getLights(contours);
    std::sort(light_list.begin(), light_list.end(), [](const Light& a, const Light& b) { return a.center.x < b.center.x; });
    matchLights();
}

void TestNode::getNumberImg(Armor &armor, Mat &num_img){
    int outWidth = 160, outHeight = 123;

    // draw4points(armor.points2, img_result);

    vector<Point2f> dstPoints {
        Point2f(0, outHeight - 1),
        Point2f(outWidth - 1, outHeight - 1),
        Point2f(outWidth - 1, 0),
        Point2f(0, 0)
    };

    Mat perspectiveMatrix = getPerspectiveTransform(armor.points2, dstPoints);
    warpPerspective(src, num_img, perspectiveMatrix, Size(outWidth, outHeight));
    cvtColor(num_img, num_img, COLOR_BGR2GRAY);
    threshold(num_img, num_img, 200, 255, THRESH_BINARY);


    // imshow("num_img", num_img);
}

int TestNode::matchNum(Mat &num_img){
    // AI generated
    if (num_imgs.size() != 5 || num_img.empty()) {
        std::cerr << "Error: Invalid input" << endl;
        return -1;
    }
    
    Mat result;
    double maxVal = 0;
    int best_match = -1;
    
    // 遍历1-5的模板进行匹配
    for (int i = 0; i < 5; i++) {
        if (num_imgs[i].empty()) {
            std::cerr << "Error: Template " << i << " is empty" << endl;
            continue;
        }

        // 使用标准相关系数匹配方法
        matchTemplate(num_img, num_imgs[i], result, TM_CCOEFF_NORMED);
        
        double minVal, maxVal_temp;
        Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal_temp, &minLoc, &maxLoc);
        
        // cout << "Number " << (i + 1) << " correlation: " << maxVal_temp << endl;
        
        // 找到最大匹配值对应的数字
        if (maxVal_temp > maxVal) {
            maxVal = maxVal_temp;
            best_match = i + 1;  // 数字1-5
        }
    }

    const double THRESHOLD = 0.48;  // 可根据实际情况调整
    if (maxVal > THRESHOLD) {
        // cout << "Matched number: " << best_match << endl;
        // cout << "Confidence: " << maxVal << endl;
        return best_match;
    } else {
        // cout << "No reliable match found. Best correlation: " << maxVal << endl;
        return -1;  // 未找到可靠匹配
    }
    return -1;
}

void TestNode::getArmorPose(){
    for (Armor& armor: armor_list){
        armor.pose = poseCalculator.getPose(armor.points);
        // TODO 卡尔曼滤波做预测 TODO
        Point3f &tmp = armor.pose.position;
        float pitch = get_hit_angle(V, sqrt(tmp.x * tmp.x + tmp.y * tmp.y), tmp.z, 9.8);
        if (pitch == NAN)
            pitch = -1;
        // RCLCPP_INFO(this->get_logger(), "计算击打角度 yaw:%f pitch: %f roll:0", armor.pose.yaw, pitch);
        putText(img_result, "dis:" + to_string(armor.pose.distance) + "m", Point2f(armor.center.x + armor.width / 2 + 10, armor.center.y), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 20, 20));
        putText(img_result, "x:" + to_string(armor.pose.position.x), Point2f(armor.center.x + armor.width / 2 + 10, armor.center.y + 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 20, 20));
        putText(img_result, "y:" + to_string(armor.pose.position.y), Point2f(armor.center.x + armor.width / 2 + 10, armor.center.y + 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 20, 20));
        putText(img_result, "z:" + to_string(armor.pose.position.z), Point2f(armor.center.x + armor.width / 2 + 10, armor.center.y + 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 20, 20));
        putText(img_result, "yaw:" + to_string(armor.pose.yaw), Point2f(armor.center.x + armor.width / 2 + 10, armor.center.y + 80), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 20, 20));
        putText(img_result, "pitch:" + to_string(pitch), Point2f(armor.center.x + armor.width / 2 + 10, armor.center.y + 100), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 20, 20));
    }
}

void TestNode::draw_history_points(){
    if (rect_list.empty()) {
        if (!rect_history_points.empty()) {
            miss_count_rect += 1;
            if (miss_count_rect > 10) {
                rect_history_points.clear();
            } else {
                connectPoints(rect_history_points, img_result);
                circle(img_result, rect_history_points[0], 4, Scalar(0, 0, 0), 2);
            }
        }
    } else {
        miss_count_rect = 0;
        if (rect_history_points.size() == max_history_length)
            rect_history_points.pop_back();
        rect_history_points.push_front(rect_list[0].center);
        connectPoints(rect_history_points, img_result);
    }
}

void TestNode::sendResult(sensor_msgs::msg::Image::SharedPtr msg){
    try{
        referee_pkg::msg::MultiObject msg_object;
        msg_object.header = msg->header;
        msg_object.num_objects = sphere_list.size() + rect_list.size() + armor_list.size();

        for (const Sphere& tmp_sphere: sphere_list){
            referee_pkg::msg::Object obj;
            obj.target_type = "sphere";
            for (int j = 0; j < 4; j++){
                geometry_msgs::msg::Point corner;
                corner.x = tmp_sphere.points[j].x;
                corner.y = tmp_sphere.points[j].y;
                corner.z = 0.0;
                obj.corners.push_back(corner);
            }
            msg_object.objects.push_back(obj);
        }

        for (const Rect_s& tmp_rect: rect_list){
            referee_pkg::msg::Object obj;
            obj.target_type = "rect";
            for (int j = 0; j < 4; j++){
                geometry_msgs::msg::Point corner;
                corner.x = tmp_rect.points[j].x;
                corner.y = tmp_rect.points[j].y;
                corner.z = 0.0;
                obj.corners.push_back(corner);
            }
            msg_object.objects.push_back(obj);
        }

        for (const Armor& tmp_armor: armor_list){
            referee_pkg::msg::Object obj;
            obj.target_type = "armor_red_" + to_string(tmp_armor.number);
            for (int j = 0; j < 4; j++){
                geometry_msgs::msg::Point corner;
                corner.x = tmp_armor.points[j].x;
                corner.y = tmp_armor.points[j].y;
                corner.z = 0.0;
                obj.corners.push_back(corner);
            }
            msg_object.objects.push_back(obj);
        }

        Target_pub->publish(msg_object);
        
        RCLCPP_INFO(this->get_logger(), "Published %zu sphere targets", sphere_list.size());        
        RCLCPP_INFO(this->get_logger(), "Published %zu rect targets", rect_list.size());        
        RCLCPP_INFO(this->get_logger(), "Published %zu armor targets", armor_list.size());
    }
    catch (const cv_bridge::Exception &e){
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
    catch (const std::exception &e){
        RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
}

void TestNode::showResult(){
    imshow("Detection Result", img_result);
    waitKey(1);
}

int main(int argc, char **argv){
    init(argc, argv);
    auto node = std::make_shared<TestNode>("TestNode");
    RCLCPP_INFO(node->get_logger(), "Starting TestNode");
    spin(node);
    shutdown();
    return 0;
}
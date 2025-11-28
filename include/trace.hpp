#ifndef TRACE
#define TRACE

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp> // 包含 KalmanFilter
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace cv;


// 匈牙利算法
// 匈牙利算法/Munkres的核心步骤标记
enum { UNMARKED, STARRED, PRIMED };

class HungarianAlgorithm {
private:
    // Helper function: 寻找一行中的最小元素并进行减法
    void step1(std::vector<std::vector<double>>& cost_matrix) {
        int rows = cost_matrix.size();
        int cols = cost_matrix[0].size();

        for (int i = 0; i < rows; ++i) {
            double min_val = std::numeric_limits<double>::max();
            for (int j = 0; j < cols; ++j) {
                if (cost_matrix[i][j] < min_val) {
                    min_val = cost_matrix[i][j];
                }
            }
            // 从该行的所有元素中减去最小值
            if (min_val != 0 && min_val != std::numeric_limits<double>::max()) {
                for (int j = 0; j < cols; ++j) {
                    if (cost_matrix[i][j] != std::numeric_limits<double>::max()) {
                        cost_matrix[i][j] -= min_val;
                    }
                }
            }
        }
    }

    // Step 2: 标记零元素 (找初始匹配)
    void step2(const std::vector<std::vector<double>>& cost_matrix, std::vector<int>& row_mask, std::vector<int>& col_mask, std::vector<std::pair<int, int>>& star_zeros) {
        int rows = cost_matrix.size();
        int cols = cost_matrix[0].size();

        row_mask.assign(rows, 0); // 0: 未覆盖
        col_mask.assign(cols, 0); // 0: 未覆盖
        star_zeros.clear();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // 找到零元素，且该行和该列未被标记
                if (std::abs(cost_matrix[i][j]) < 1e-9 && row_mask[i] == 0 && col_mask[j] == 0) {
                    star_zeros.emplace_back(i, j); // 标记为 'star' zero
                    row_mask[i] = 1; // 覆盖该行
                    col_mask[j] = 1; // 覆盖该列
                }
            }
        }
    }

    // Step 3: 覆盖所有包含 'star' zero 的列。如果覆盖的列数等于矩阵的维度，则完成
    bool step3(int dim, const std::vector<std::pair<int, int>>& star_zeros, std::vector<int>& col_mask) {
        // 在这一步，col_mask 已经被 step2 初始化。
        int covered_cols = 0;
        for (int mask : col_mask) {
            if (mask == 1) covered_cols++;
        }
        return covered_cols >= dim; // 如果覆盖的列数等于行数/列数，则找到解
    }

    // Step 4: 核心迭代步骤 (找新的零元素，并转换标记)
    void step4(const std::vector<std::vector<double>>& cost_matrix, std::vector<int>& row_mask, std::vector<int>& col_mask,
               std::vector<std::pair<int, int>>& star_zeros, std::vector<std::pair<int, int>>& prime_zeros) {
        int rows = cost_matrix.size();
        int cols = cost_matrix[0].size();

        while (true) {
            // 4.1 寻找未覆盖行中的未覆盖零元素 (Prime zero)
            int prime_r = -1, prime_c = -1;
            for (int i = 0; i < rows; ++i) {
                if (row_mask[i] == 0) { // 只看未覆盖的行
                    for (int j = 0; j < cols; ++j) {
                        if (col_mask[j] == 0) { // 只看未覆盖的列
                            if (std::abs(cost_matrix[i][j]) < 1e-9) {
                                prime_r = i;
                                prime_c = j;
                                prime_zeros.emplace_back(prime_r, prime_c);
                                goto found_prime;
                            }
                        }
                    }
                }
            }

        found_prime:;
            if (prime_r == -1) {
                // 找不到新的 Prime zero，进行矩阵调整 (Step 6)
                return;
            }

            // 4.2 检查 Prime zero 所在行是否有 Star zero
            int star_c = -1;
            for (const auto& star : star_zeros) {
                if (star.first == prime_r) {
                    star_c = star.second;
                    break;
                }
            }

            if (star_c == -1) {
                // 该 Prime zero 所在行没有 Star zero，找到增广路径 (Step 5)
                step5(prime_r, prime_c, star_zeros, prime_zeros, row_mask, col_mask);
                return;
            } else {
                // 找到 Star zero，覆盖 Prime zero 所在的行，取消覆盖 Star zero 所在的列
                row_mask[prime_r] = 1;
                col_mask[star_c] = 0;
            }
        }
    }

    // Step 5: 增广路径转换 (核心匹配调整)
    void step5(int r, int c, std::vector<std::pair<int, int>>& star_zeros, std::vector<std::pair<int, int>>& prime_zeros,
               std::vector<int>& row_mask, std::vector<int>& col_mask) {
        // 构建增广路径：从 (r, c) 开始，Prime zero -> Star zero -> Prime zero -> ...
        std::vector<std::pair<int, int>> path;
        path.emplace_back(r, c);

        while (true) {
            // 寻找 path 中最后一个 Prime zero 对应的 Star zero (如果存在)
            int star_r = -1, star_c = -1;
            for (const auto& star : star_zeros) {
                if (star.second == path.back().second) { // 寻找同一列的 Star zero
                    star_r = star.first;
                    star_c = star.second;
                    break;
                }
            }

            if (star_r == -1) break; // 路径结束

            path.emplace_back(star_r, star_c);

            // 寻找 path 中最后一个 Star zero 对应的 Prime zero
            int prime_r = -1, prime_c = -1;
            for (const auto& prime : prime_zeros) {
                if (prime.first == star_r) { // 寻找同一行的 Prime zero
                    prime_r = prime.first;
                    prime_c = prime.second;
                    break;
                }
            }
            if (prime_r != -1) path.emplace_back(prime_r, prime_c);
        }

        // 转换路径上的标记：Prime 变为 Star，Star 标记被移除
        for (size_t k = 0; k < path.size(); ++k) {
            int pr = path[k].first;
            int pc = path[k].second;

            if (k % 2 == 0) { // Prime zero (Path[0], Path[2], ...)
                // 移除与 Prime zero 所在列相同的 Star zero
                star_zeros.erase(std::remove_if(star_zeros.begin(), star_zeros.end(),
                    [pc](const std::pair<int, int>& s){ return s.second == pc; }), star_zeros.end());
                // 将 Prime zero 添加为 Star zero
                star_zeros.emplace_back(pr, pc);
            }
        }

        // 重置所有标记和覆盖状态，回到 Step 3 (重新覆盖列)
        row_mask.assign(row_mask.size(), 0);
        col_mask.assign(col_mask.size(), 0);
        prime_zeros.clear();

        // Step 2-style 重新覆盖 Star zero 所在的列
        for(const auto& star : star_zeros) {
            col_mask[star.second] = 1;
        }
    }

    // Step 6: 调整矩阵元素 (揭示新的零元素)
    void step6(std::vector<std::vector<double>>& cost_matrix, const std::vector<int>& row_mask, const std::vector<int>& col_mask) {
        int rows = cost_matrix.size();
        int cols = cost_matrix[0].size();

        double min_val = std::numeric_limits<double>::max();
        for (int i = 0; i < rows; ++i) {
            if (row_mask[i] == 0) { // 仅考虑未覆盖的行
                for (int j = 0; j < cols; ++j) {
                    if (col_mask[j] == 0) { // 仅考虑未覆盖的列
                        if (cost_matrix[i][j] < min_val) {
                            min_val = cost_matrix[i][j];
                        }
                    }
                }
            }
        }

        if (min_val == 0 || min_val == std::numeric_limits<double>::max()) return;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (row_mask[i] == 0 && col_mask[j] == 0) {
                    // 减去未覆盖行和未覆盖列的交点 (未覆盖区域)
                    cost_matrix[i][j] -= min_val;
                } else if (row_mask[i] == 1 && col_mask[j] == 1) {
                    // 加上覆盖行和覆盖列的交点 (双覆盖区域)
                    if (cost_matrix[i][j] != std::numeric_limits<double>::max()) {
                        cost_matrix[i][j] += min_val;
                    }
                }
            }
        }
    }

public:
    /**
     * @brief 求解最小化代价分配问题。
     * @param cost_data 代价矩阵 (N_tracks 行, N_detections 列)。
     * @param assignment 输出：分配结果 assignment[track_idx] = detection_idx。
     * @return double 总代价。
     */
    double Solve(std::vector<std::vector<double>>& cost_data, std::vector<int>& assignment) {
        if (cost_data.empty() || cost_data[0].empty()) return 0.0;

        int original_rows = cost_data.size();
        int original_cols = cost_data[0].size();
        int dim = max(original_rows, original_cols);

        std::vector<std::vector<double>> cost_matrix = cost_data;
        if (original_rows != original_cols) {
            cost_matrix.resize(dim, std::vector<double>(dim, 0.0));
        }

        std::vector<int> row_mask(dim);
        std::vector<int> col_mask(dim);
        std::vector<std::pair<int, int>> star_zeros;
        std::vector<std::pair<int, int>> prime_zeros;

        step1(cost_matrix);
        step2(cost_matrix, row_mask, col_mask, star_zeros);

        while (!step3(dim, star_zeros, col_mask)) {
            step4(cost_matrix, row_mask, col_mask, star_zeros, prime_zeros);
            step6(cost_matrix, row_mask, col_mask);
        }

        assignment.assign(original_rows, -1);
        double total_cost = 0.0;

        for (const auto& star : star_zeros) {
            int r = star.first;
            int c = star.second;

            if (r < original_rows && c < original_cols) {
                assignment[r] = c;
                total_cost += cost_data[r][c];
            }
        }

        return total_cost;
    }
};


// 跟踪中的目标结构（现在包含一个卡尔曼滤波器）
struct TrackedObject {
    int id;
    KalmanFilter kf;
    int consecutive_misses = 0;

    // 初始化卡尔曼滤波器
    TrackedObject(int track_id, const Point2f& initial_pos) : id(track_id) {

        kf.init(4, 2, 0);
        setIdentity(kf.transitionMatrix);
        kf.transitionMatrix.at<float>(0, 2) = 1.0f; // x += vx
        kf.transitionMatrix.at<float>(1, 3) = 1.0f; // y += vy

        kf.measurementMatrix.setTo(Scalar(0));
        kf.measurementMatrix.at<float>(0, 0) = 1.0f;
        kf.measurementMatrix.at<float>(1, 1) = 1.0f;

        setIdentity(kf.processNoiseCov, Scalar(1e-2));

        setIdentity(kf.measurementNoiseCov, Scalar(1e-1));

        kf.statePost.at<float>(0) = initial_pos.x;
        kf.statePost.at<float>(1) = initial_pos.y;
        kf.statePost.at<float>(2) = 0.0f;
        kf.statePost.at<float>(3) = 0.0f;

        setIdentity(kf.errorCovPost, Scalar(1.0f));
    }

    Point2f predict() {
        Mat prediction = kf.predict();
        return Point2f(prediction.at<float>(0), prediction.at<float>(1));
    }

    void correct(const Point2f& observation) {
        Mat measurement(2, 1, CV_32F);
        measurement.at<float>(0) = observation.x;
        measurement.at<float>(1) = observation.y;
        kf.correct(measurement);
    }
};
struct Trace {
    // 全局跟踪列表和ID计数器
    std::map<int, TrackedObject> active_tracks;
    int next_track_id = 0;

    // 匹配对象类型
    // 0: 球; 1: 矩形; 2: 装甲板1; 3: 装甲板2; 4: 装甲板3; 5: 装甲板4; 6: 装甲板5
    int class_;
    std::string class_name;
    explicit Trace(const int class_) : class_(class_) {
        switch (this->class_) {
        case 0:
            class_name = "Sphere";
            break;
        case 1:
            class_name = "Rect";
            break;
        case 2:
            class_name = "Armor_1";
            break;
        case 3:
            class_name = "Armor_2";
            break;
        case 4:
            class_name = "Armor_3";
            break;
        case 5:
            class_name = "Armor_4";
            break;
        case 6:
            class_name = "Armor_5";
            break;
        default:
            class_name = "Unknown";
        }
    }

    // 匹配参数
    float MAX_DISTANCE_THRESHOLD = 50.0f; // 匹配阈值
    int MAX_MISSES = 5; // 连续未匹配多少帧后认为目标消失

    Mat build_cost_matrix(const std::vector<Point2f>& current_detections,
                      std::vector<Point2f>& predicted_positions, // 输出：所有跟踪目标的预测位置
                      std::vector<int>& track_ids);
    void match_and_update(const std::vector<Point2f>& current_detections);
    /**
    * @brief 使用匈牙利算法求解最优分配问题。
    * @param cost_matrix 代价矩阵 (N_tracks 行, N_detections 列)，CV_32F类型。
    * @param max_cost 允许的最大匹配代价，超过此值不予匹配。
    * @return std::vector<std::pair<int, int>> 包含最优匹配对的列表。
    * 每个pair是 (Track Index, Detection Index)。
    */
    std::vector<std::pair<int, int>> solve_hungarian(const Mat& cost_matrix, float max_cost);
};

#endif
#include "trace.hpp"

using namespace cv;

std::vector<std::pair<int, int>> Trace::solve_hungarian(const Mat& cost_matrix, float max_cost) {
    if (cost_matrix.empty()) {
        return {};
    }

    int N_tracks = cost_matrix.rows;
    int N_detections = cost_matrix.cols;

    // 转换为 double 类型的 std::vector
    std::vector cost_data(N_tracks, std::vector<double>(N_detections));

    for (int i = 0; i < N_tracks; ++i) {
        for (int j = 0; j < N_detections; ++j) {
            float cost = cost_matrix.at<float>(i, j);

            // 超过阈值的代价，设为算法中的“无限大”以阻止匹配
            if (cost > max_cost) {
                cost_data[i][j] = std::numeric_limits<double>::max();
            } else {
                cost_data[i][j] = static_cast<double>(cost);
            }
        }
    }

    // 调用匈牙利算法求解器
    HungarianAlgorithm solver;
    std::vector<int> assignment; // assignment[track_idx] = detection_idx

    solver.Solve(cost_data, assignment);

    // 结果转换
    std::vector<std::pair<int, int>> matches;

    for (int i = 0; i < N_tracks; ++i) {
        int det_idx = assignment[i];

        // 检查是否匹配成功 (assignment[i] != -1) 并且代价在合理范围内 (虽然 Solve 内部已处理，这里再次确认鲁棒性)
        if (det_idx != -1 && det_idx < N_detections && cost_data[i][det_idx] <= max_cost) {
            matches.emplace_back(i, det_idx);
        }
    }

    return matches;
}

Mat Trace::build_cost_matrix(const std::vector<Point2f>& current_detections,
                      std::vector<Point2f>& predicted_positions, // 输出：所有跟踪目标的预测位置
                      std::vector<int>& track_ids) {

    // 预测下一帧的位置
    for (auto& pair : active_tracks) {
        predicted_positions.push_back(pair.second.predict());
        track_ids.push_back(pair.first);
    }

    int N_tracks = predicted_positions.size();
    int N_detections = current_detections.size();

    // 如果没有跟踪或没有检测，则跳过
    if (N_tracks == 0 || N_detections == 0) {
        return {};
    }

    // 代价矩阵 (N_tracks 行, N_detections 列)
    Mat cost_matrix(N_tracks, N_detections, CV_32F);

    for (int i = 0; i < N_tracks; ++i) {
        for (int j = 0; j < N_detections; ++j) {
            // 代价 = 欧氏距离
            float dist = norm(predicted_positions[i] - current_detections[j]);

            // 如果距离超过阈值，代价设为无穷大，匈牙利算法将不会匹配它
            if (dist > MAX_DISTANCE_THRESHOLD) {
                cost_matrix.at<float>(i, j) = std::numeric_limits<float>::max();
            } else {
                cost_matrix.at<float>(i, j) = dist;
            }
        }
    }
    return cost_matrix;
}

void Trace::match_and_update(const std::vector<Point2f>& current_detections) {
    std::vector<Point2f> predicted_positions;
    std::vector<int> track_ids;
    Mat cost_matrix = build_cost_matrix(current_detections, predicted_positions, track_ids);

    if (cost_matrix.empty()) {
        // 如果没有跟踪或检测，处理新增或消失
        if (active_tracks.empty()) {
            // 全部新增
            for (const auto& p : current_detections) {
                active_tracks.emplace(next_track_id, TrackedObject(next_track_id++, p));
            }
        } else {
            // 全部消失
            for (auto& pair : active_tracks) {
                pair.second.consecutive_misses++;
            }
        }
        // 清理消失的点
        std::vector<int> tracks_to_remove;
        for (const auto& pair : active_tracks) {
            if (pair.second.consecutive_misses > MAX_MISSES) {
                tracks_to_remove.push_back(pair.first);
            }
        }
        for (int id : tracks_to_remove) {
            active_tracks.erase(id);
        }
        return;
    }

    // 这里的返回值是一个包含 (track_index, detection_index) 的匹配对列表
    std::vector<std::pair<int, int>> matches = solve_hungarian(cost_matrix, MAX_DISTANCE_THRESHOLD);

    // 标记匹配状态
    std::vector matched_track(track_ids.size(), false);
    std::vector matched_detection(current_detections.size(), false);

    // 更新匹配成功的跟踪目标
    for (const auto& p : matches) {
        int track_idx = p.first;
        int det_idx = p.second;
        int track_id = track_ids[track_idx];

        // 使用检测结果校正卡尔曼滤波器
        active_tracks.at(track_id).correct(current_detections[det_idx]);
        active_tracks.at(track_id).consecutive_misses = 0; // 重置未匹配计数

        matched_track[track_idx] = true;
        matched_detection[det_idx] = true;
    }

    // 处理未匹配的跟踪目标 (目标消失或遮挡)
    std::vector<int> tracks_to_remove;
    for (size_t i = 0; i < track_ids.size(); ++i) {
        if (!matched_track[i]) {
            int track_id = track_ids[i];
            // 对于未匹配的跟踪，卡尔曼滤波器已经通过 predict() 运行了，
            // 但我们不进行 correct()，只增加未匹配计数。
            active_tracks.at(track_id).consecutive_misses++;

            if (active_tracks.at(track_id).consecutive_misses > MAX_MISSES) {
                tracks_to_remove.push_back(track_id);
            }
        }
    }

    // 移除已消失的跟踪目标
    for (int id : tracks_to_remove) {
        active_tracks.erase(id);
    }

    // 处理未匹配的检测点 (新增目标)
    for (size_t i = 0; i < current_detections.size(); ++i) {
        if (!matched_detection[i]) {
            // 新建跟踪，并初始化卡尔曼滤波器
            active_tracks.emplace(next_track_id, TrackedObject(next_track_id++, current_detections[i]));
        }
    }
}

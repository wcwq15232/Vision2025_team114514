#include "trace.hpp"

// #include <numeric>
// #include <utility>
//
// Trace::Trace(const int maxDisappeared) {
//     this->maxDisappeared = maxDisappeared;
// }
//
// void Trace::register_(const cv::Point& centroid) {
//     objects[nextObjectID] = centroid;
//     disappeared[nextObjectID] = 0;
//     nextObjectID++;
// }
//
// void Trace::deregister(const int objectID) {
//     objects.erase(objectID);
//     disappeared.erase(objectID);
// }
//
// std::map<int, cv::Point> Trace::update(std::vector<cv::Point> centroids) {
//
//     if (objects.empty()) {
//         for (auto [fst, snd] : disappeared) {
//             disappeared[fst]++;
//             if (disappeared[fst] > maxDisappeared) {
//                 deregister(fst);
//                 return objects;
//             }
//         }
//     }
//
//     auto inputCentroids = std::move(centroids);
//
//     if (objects.empty()) {
//
//         for (const auto & inputCentroid : inputCentroids) {
//             register_(inputCentroid);
//         }
//
//     } else {
//
//         auto objectIDs = std::vector<int>(objects.size());
//         auto objectCentroids = std::vector<cv::Point>(objects.size());
//         {
//             int i = 0;
//             for (const auto& [fst, snd] : objects) {
//                 objectIDs[i] = fst;
//                 objectCentroids[i] = snd;
//                 i++;
//             }
//         }
//
//         auto D = std::vector<std::vector<int> >(objectCentroids.size());
//
//         for (int row = 0; row < D.size(); row++) {
//
//             D[row] = std::vector<int>(inputCentroids.size());
//             for (int col = 0; col < D[row].size(); col++) {
//                 D[row][col] = std::sqrt(
//                     std::pow(objectCentroids[row].x + inputCentroids[col].x, 2)
//                     + std::pow(objectCentroids[row].y + inputCentroids[col].y, 2));
//             }
//
//         }
//
//         // 找行最小值的索引
//         // 第一步：对每一行找到最小值的索引 (argmin(axis=1))
//         std::vector<size_t> argmin_indices;
//         argmin_indices.reserve(D.size());
//
//         for (const auto& row : D) {
//             if (row.empty()) {
//                 argmin_indices.push_back(0);
//                 continue;
//             }
//
//             auto min_it = std::min_element(row.begin(), row.end());
//             size_t min_index = std::distance(row.begin(), min_it);
//             argmin_indices.push_back(min_index);
//         }
//
//         // 第二步：对argmin_indices进行argsort排序
//         std::vector<size_t> rows(argmin_indices.size());
//         std::iota(rows.begin(), rows.end(), 0);
//
//         // 根据argmin_indices的值对索引进行排序
//         std::sort(rows.begin(), rows.end(),
//                   [&](size_t i, size_t j) {
//                       return argmin_indices[i] < argmin_indices[j];
//                   });
//
//
//         // 找列最小值的索引
//         std::vector<size_t> cols;
//         cols.reserve(rows.size());
//
//         for (size_t row_idx : rows) {
//             if (row_idx >= D.size()) {
//                 // 处理越界情况，可以根据需求调整
//                 cols.push_back(0);
//                 continue;
//             }
//
//             const auto& row = D[row_idx];
//             if (row.empty()) {
//                 cols.push_back(0);
//                 continue;
//             }
//
//             auto min_it = std::min_element(row.begin(), row.end());
//             size_t min_index = std::distance(row.begin(), min_it);
//             cols.push_back(min_index);
//         }
//
//
//         // 记录已检查的行索引和列索引
//         std::set<int> usedRows, usedCols;
//         for (int row = 0; row < D.size(); row++) {
//
//             for (int col = 0; col < D[rows[row]].size(); col++) {
//                 if (usedRows.find(rows[row]) != usedRows.end() || usedCols.find(cols[col]) != usedCols.end())
//                     continue;
//                 int objectID = objectIDs[rows[row]];
//                 objects[objectID] = inputCentroids[cols[col]];
//                 disappeared[objectID] = 0;
//                 usedRows.insert(rows[row]);
//                 usedCols.insert(cols[col]);
//             }
//
//         }
//
//         // 计算尚未检查的行索引和列索引
//         auto unused = [] (std::set<int> used, const int size) -> std::set<int> {
//             std::set<int> unused_;
//             for (int i = 0; i < size; i++) {
//                 if (used.find(i) == used.end()) {
//                     unused_.insert(i);
//                 }
//             }
//             return unused_;
//         };
//         auto unusedRows = unused(usedRows, D.size());
//         auto unusedCols = unused(usedCols, D[0].size());
//
//         // 若对象质心数大于等于输入质心数
//         // 检查其中某些对象是否已消失
//         if (D.size() >= D[0].size()) {
//             for (auto row : unusedRows) {
//                 int objectID = objectIDs[row];
//                 disappeared[objectID]++;
//                 if (disappeared[objectID] > maxDisappeared) {
//                     deregister(objectID);
//                 }
//             }
//         } else {
//             for (auto col : unusedCols) {
//                 register_(inputCentroids[col]);
//             }
//         }
//     }
//     return objects;
// }

using namespace cv;

std::vector<std::pair<int, int>> Trace::solve_hungarian(const Mat& cost_matrix, float max_cost) {
    if (cost_matrix.empty()) {
        return {};
    }

    int N_tracks = cost_matrix.rows;
    int N_detections = cost_matrix.cols;

    // 1. 转换为 double 类型的 std::vector
    std::vector<std::vector<double>> cost_data(N_tracks, std::vector<double>(N_detections));

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

    // 2. 调用匈牙利算法求解器
    HungarianAlgorithm solver;
    std::vector<int> assignment; // assignment[track_idx] = detection_idx

    solver.Solve(cost_data, assignment);

    // 3. 结果转换
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
        // 清理消失的
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

    // 假设调用匈牙利算法求解器
    // 实际应用中，您需要引入或实现一个匈牙利算法库
    // 这里的返回值是一个包含 (track_index, detection_index) 的匹配对列表
    std::vector<std::pair<int, int>> matches = solve_hungarian(cost_matrix, MAX_DISTANCE_THRESHOLD);

    // 3. 标记匹配状态
    std::vector matched_track(track_ids.size(), false);
    std::vector matched_detection(current_detections.size(), false);

    // 4. 更新匹配成功的跟踪目标
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

    // 5. 处理未匹配的跟踪目标 (目标消失或遮挡)
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

    // 6. 处理未匹配的检测点 (新增目标)
    for (size_t i = 0; i < current_detections.size(); ++i) {
        if (!matched_detection[i]) {
            // 新建跟踪，并初始化卡尔曼滤波器
            active_tracks.emplace(next_track_id, TrackedObject(next_track_id++, current_detections[i]));
        }
    }
}

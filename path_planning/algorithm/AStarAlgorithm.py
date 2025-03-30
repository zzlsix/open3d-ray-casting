import heapq
import numpy as np
import trimesh
from collections import defaultdict
from path_planning.GeometryTools import GeometryTools
from path_planning.algorithm.PathPlanningAlgorithm import PathPlanningAlgorithm


# A*算法实现
class AStarAlgorithm(PathPlanningAlgorithm):
    def find_path(self, start, goal, planner):
        """使用A*算法查找路径"""
        # 处理起点和终点，确保它们在表面外部
        safe_start = GeometryTools.find_nearest_surface_point(planner.mesh, start)
        safe_goal = GeometryTools.find_nearest_surface_point(planner.mesh, goal)

        print(f"处理后的起点: {safe_start}")
        print(f"处理后的终点: {safe_goal}")

        # 调整起点和终点，确保不在障碍物内
        safe_start, start_adjusted = self.adjust_point_if_in_collision(safe_start, planner)
        safe_goal, goal_adjusted = self.adjust_point_if_in_collision(safe_goal, planner)

        # 如果仍然无法找到有效的起点或终点
        if self.is_collision(safe_start, planner):
            print("错误：无法找到有效的起点")
            return None

        if self.is_collision(safe_goal, planner):
            print("错误：无法找到有效的终点")
            return None

        # 转换为网格坐标
        start_grid = planner.to_grid(safe_start)
        goal_grid = planner.to_grid(safe_goal)

        print(f"开始A*搜索，从网格坐标 {start_grid} 到 {goal_grid}")

        # A*算法初始化
        open_set = []
        heapq.heappush(open_set, (0, start_grid))

        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_grid] = 0

        f_score = defaultdict(lambda: float('inf'))
        f_score[start_grid] = GeometryTools.heuristic(start_grid, goal_grid)

        open_set_hash = {start_grid}
        visited_count = 0

        # A*主循环
        while open_set:
            # 获取f值最小的节点
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            visited_count += 1

            # 定期显示进度
            if visited_count % 1000 == 0:
                print(f"已探索 {visited_count} 个节点...")

            # 达到目标
            if current == goal_grid:
                path = []
                while current in came_from:
                    path.append(planner.to_world(current))
                    current = came_from[current]
                path.append(planner.to_world(start_grid))  # 添加起点
                path.reverse()  # 反转路径顺序

                print(f"路径长度: {len(path)}个点")
                print(f"探索节点数: {visited_count}")
                process_path = self.post_process_path(planner.mesh, path, start, goal)
                return process_path

            # 检查所有邻居
            for neighbor_grid in self._get_neighbors(current):
                # 检查邻居点是否合法
                neighbor_world = planner.to_world(neighbor_grid)

                if not planner.is_within_bounds(neighbor_world) or self.is_collision(neighbor_world, planner):
                    continue

                # 计算到达邻居的代价
                move_cost = self._calculate_move_cost(current, neighbor_grid, planner.voxel_size)
                tentative_g = g_score[current] + move_cost

                # 如果找到更好的路径
                if tentative_g < g_score[neighbor_grid]:
                    came_from[neighbor_grid] = current
                    g_score[neighbor_grid] = tentative_g
                    f_score[neighbor_grid] = tentative_g + GeometryTools.heuristic(neighbor_grid, goal_grid)

                    if neighbor_grid not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor_grid], neighbor_grid))
                        open_set_hash.add(neighbor_grid)

        # 如果没有找到路径
        print(f"探索节点数: {visited_count}")
        print(f"没有找到路径")
        return None

    @staticmethod
    def _get_neighbors(current):
        """获取当前网格点的所有邻居"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # 跳过当前点
                    neighbors.append((current[0] + dx, current[1] + dy, current[2] + dz))
        return neighbors

    @staticmethod
    def _calculate_move_cost(current, neighbor, voxel_size):
        """计算从当前点到邻居点的移动代价"""
        dx = abs(current[0] - neighbor[0])
        dy = abs(current[1] - neighbor[1])
        dz = abs(current[2] - neighbor[2])

        if dx + dy + dz == 1:  # 直线移动
            return voxel_size
        elif dx + dy + dz == 2:  # 面对角线
            return 1.414 * voxel_size  # √2
        else:  # 体对角线
            return 1.732 * voxel_size  # √3

    def is_collision(self, point, planner):
        """检查点是否与障碍物碰撞"""
        # 性能不高，暂时优化，但准确率降低
        # # 将点转换为numpy数组以确保兼容性
        # point_array = np.asarray(point)
        # return planner.voxel_grid.is_filled(point_array)

        return planner.voxel_index.is_occupied(point)


    def post_process_path(self, mesh, path, original_start, original_end, offset_distance=0.1):
        """将路径点投影到表面附近"""
        if path is None:
            return None

        # 保留原始起点和终点
        processed_path = [original_start]

        # 处理中间点 - 将它们投影到距离表面更近的位置
        for i in range(1, len(path) - 1):
            point = path[i]

            # 计算点到网格的最近点
            closest_point, distance, triangle_id = trimesh.proximity.closest_point(mesh, [point])
            closest_point = closest_point[0]

            # 计算表面法线
            normal = mesh.face_normals[triangle_id[0]]

            # 将点移动到表面附近但不在表面上
            # 小偏移，让路径非常接近表面
            projected_point = closest_point + normal * offset_distance

            processed_path.append(projected_point)

        # 添加原始终点
        processed_path.append(original_end)

        return processed_path

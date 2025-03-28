import heapq
import time
from collections import defaultdict

import numpy as np
import trimesh

from path_planning.GeometryTools import GeometryTools


# 路径规划模块
class PathPlanner:
    def __init__(self, mesh, occupied_voxels, voxel_size, bounds=None):
        self.mesh = mesh
        self.occupied_voxels = occupied_voxels
        self.voxel_size = voxel_size
        self.bounds = bounds
        self.geometry_tools = GeometryTools()

    def to_grid(self, point):
        """将连续坐标转换为离散网格坐标"""
        return tuple(np.round(np.array(point) / self.voxel_size).astype(int))

    def to_world(self, grid_point):
        """将离散网格坐标转换为连续坐标"""
        return tuple(np.array(grid_point) * self.voxel_size)

    def is_within_bounds(self, point):
        """检查点是否在边界内"""
        if self.bounds is None:
            return True
        # 增加一点余量，因为点可能在边界外一点点
        margin = self.voxel_size * 2
        return all(self.bounds[0][i] - margin <= point[i] <= self.bounds[1][i] + margin for i in range(3))

    def is_collision(self, point):
        """检查点是否与障碍物碰撞"""
        return GeometryTools.is_point_in_voxels(point, self.occupied_voxels, self.voxel_size)

    def adjust_point_if_in_collision(self, point):
        """如果点在障碍物内，尝试调整"""
        if not self.is_collision(point):
            return point, False

        print(f"警告：点在障碍物内，尝试小幅调整...")
        for offset in [
            (self.voxel_size, 0, 0), (-self.voxel_size, 0, 0),
            (0, self.voxel_size, 0), (0, -self.voxel_size, 0),
            (0, 0, self.voxel_size), (0, 0, -self.voxel_size)
        ]:
            new_point = tuple(np.array(point) + np.array(offset))
            if not self.is_collision(new_point):
                print(f"点已调整为: {new_point}")
                return new_point, True

        return point, False

    def find_path(self, start, goal):
        """使用A*算法寻找从起点到终点的路径"""
        # 处理起点和终点，确保它们在表面外部
        safe_start = GeometryTools.find_nearest_surface_point(self.mesh, start)
        safe_goal = GeometryTools.find_nearest_surface_point(self.mesh, goal)

        print(f"处理后的起点: {safe_start}")
        print(f"处理后的终点: {safe_goal}")

        # 调整起点和终点，确保不在障碍物内
        safe_start, start_adjusted = self.adjust_point_if_in_collision(safe_start)
        safe_goal, goal_adjusted = self.adjust_point_if_in_collision(safe_goal)

        # 如果仍然无法找到有效的起点或终点
        if self.is_collision(safe_start):
            print("错误：无法找到有效的起点")
            return None

        if self.is_collision(safe_goal):
            print("错误：无法找到有效的终点")
            return None

        start_grid = self.to_grid(safe_start)
        goal_grid = self.to_grid(safe_goal)

        print(f"开始A*搜索，从 {safe_start} 到 {safe_goal}")
        start_time = time.time()

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
                    path.append(self.to_world(current))
                    current = came_from[current]
                path.append(safe_start)
                path.reverse()

                # 在路径末尾添加真正的目标点
                path.append(goal)

                end_time = time.time()
                print(f"找到路径！用时: {end_time - start_time:.2f}秒")
                print(f"路径长度: {len(path)}个点")
                print(f"探索节点数: {visited_count}")
                return path

            # 检查所有邻居
            for neighbor_grid in [
                (current[0] + dx, current[1] + dy, current[2] + dz)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dz in [-1, 0, 1]
                if not (dx == 0 and dy == 0 and dz == 0)
            ]:
                # 检查邻居点是否合法
                neighbor_world = self.to_world(neighbor_grid)

                if not self.is_within_bounds(neighbor_world) or self.is_collision(neighbor_world):
                    continue

                # 计算到达邻居的代价
                dx, dy, dz = abs(current[0] - neighbor_grid[0]), abs(current[1] - neighbor_grid[1]), abs(
                    current[2] - neighbor_grid[2])

                if dx + dy + dz == 1:  # 直线移动
                    move_cost = self.voxel_size
                elif dx + dy + dz == 2:  # 面对角线
                    move_cost = 1.414 * self.voxel_size  # √2
                else:  # 体对角线
                    move_cost = 1.732 * self.voxel_size  # √3

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
        end_time = time.time()
        print(f"没有找到路径。用时: {end_time - start_time:.2f}秒")
        print(f"探索节点数: {visited_count}")
        return None

    def post_process_path(self, path, original_start, original_end):
        """将路径点投影到表面附近"""
        if path is None:
            return None

        processed_path = []

        # 保留原始起点和终点
        processed_path.append(original_start)

        # 处理中间点 - 将它们投影到距离表面更近的位置
        for i in range(1, len(path) - 1):
            point = path[i]

            # 计算点到网格的最近点
            closest_point, distance, triangle_id = trimesh.proximity.closest_point(self.mesh, [point])
            closest_point = closest_point[0]

            # 计算表面法线
            normal = self.mesh.face_normals[triangle_id[0]]

            # 将点移动到表面附近但不在表面上
            offset_distance = 0.2  # 小偏移，让路径非常接近表面
            projected_point = closest_point + normal * offset_distance

            processed_path.append(projected_point)

        # 添加原始终点
        processed_path.append(original_end)

        return processed_path

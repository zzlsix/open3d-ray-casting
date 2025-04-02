import heapq
import numpy as np
import trimesh
from collections import defaultdict
from path_planning.GeometryTools import GeometryTools
from path_planning.algorithm.PathPlanningAlgorithm import PathPlanningAlgorithm
from path_planning.visualization.ProcessVisualizer import AStarProcessVisualizer


# A*算法实现
class AStarAlgorithm(PathPlanningAlgorithm):
    def __init__(self):
        self.visualizer = AStarProcessVisualizer()

    def find_path(self, start, goal, planner, show_process=True):
        """使用A*算法查找路径"""
        # 处理起点和终点，确保它们在表面外部
        safe_start = GeometryTools.find_nearest_surface_point(planner.mesh, start)
        safe_goal = GeometryTools.find_nearest_surface_point(planner.mesh, goal)

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
        f_score[start_grid] = self.heuristic(start_grid, goal_grid)

        open_set_hash = {start_grid}
        visited_points = []
        open_set_points = [planner.to_world(start_grid)]

        # 初始化可视化
        if show_process:
            self.visualizer.show(planner.mesh, start, goal, None)

        visited_count = 0

        # A*主循环
        while open_set:
            # 获取f值最小的节点
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            current_world = planner.to_world(current)
            visited_points.append(current_world)
            visited_count += 1

            # 构建当前路径用于可视化
            temp_path = []
            temp_current = current
            while temp_current in came_from:
                temp_path.append(planner.to_world(temp_current))
                temp_current = came_from[temp_current]
            temp_path.append(planner.to_world(start_grid))
            temp_path.reverse()

            # 更新开放集点的世界坐标
            open_set_world_points = [planner.to_world(node) for _, node in open_set]
            if show_process:
                self.visualizer.update(current_world, open_set_world_points, visited_points, came_from, temp_path)

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

                # 显示最终路径
                if show_process:
                    self.visualizer.update_path(process_path)
                    self.visualizer.vis_o3d.run()  # 保持窗口打开直到用户关闭
                    self.visualizer.vis_o3d.destroy_window()

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
                    f_score[neighbor_grid] = tentative_g + self.heuristic(neighbor_grid, goal_grid)

                    if neighbor_grid not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor_grid], neighbor_grid))
                        open_set_hash.add(neighbor_grid)
                        open_set_points.append(neighbor_world)

        # 如果没有找到路径
        print(f"探索节点数: {visited_count}")
        print(f"没有找到路径")

        # 关闭可视化窗口
        if show_process:
            self.visualizer.vis_o3d.destroy_window()

        return None

    @staticmethod
    def heuristic(a, b):
        """计算两点间的欧几里得距离"""
        return np.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))) * 1.5

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
